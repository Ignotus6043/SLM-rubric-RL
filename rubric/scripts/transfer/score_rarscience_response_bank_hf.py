#!/usr/bin/env python3
"""Score a RaR-Science response bank with direct HF generation, matching Bench."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """
You are an impartial rubric grader.

Instruction:
{question}

Candidate Response:
{response}

Rubric Rules ({num_rules} total):
{rubric_rules}

Task:
For each rubric rule, decide whether the response satisfies it.

Output format requirement (strict):
- Return exactly {num_rules} lines.
- Each line must follow this format exactly:
  Rule i: <short justification>. Verdict: Yes/No
- Use i = 1..{num_rules} in order.
- Keep each justification brief (1 sentence).
- Verdict must be exactly `Yes` or `No`.

Example for 6 rules:
Rule 1: [Justification for rule 1]. Verdict: Yes
Rule 2: [Justification for rule 2]. Verdict: No
...
Rule 6: [Justification for rule 6]. Verdict: Yes

Do not output JSON, markdown, extra headings, or any text outside those rule lines.
""".strip()

VERDICT_ONLY_PROMPT_TEMPLATE = """
You are an impartial rubric grader.

Instruction:
{question}

Candidate Response:
{response}

Rubric Rules ({num_rules} total):
{rubric_rules}

Task:
For each rubric rule, decide whether the response satisfies it.

Output format requirement (strict):
- Return exactly {num_rules} lines.
- Each line must follow this format exactly:
  Rule i: Yes/No
- Use i = 1..{num_rules} in order.
- Use only Yes or No after the colon.

Do not output reasoning, JSON, markdown, headings, or any text outside those rule lines.
""".strip()

SAFETY_BUFFER_GB = 12.0
MAX_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-style", choices=("bench", "verdict_only"), default="bench")
    parser.add_argument(
        "--decomposition-mode",
        choices=("vector", "per_rule", "grouped"),
        default="vector",
        help=(
            "Rubric decomposition ablation. vector=score all rules in one call; "
            "per_rule=one judge call per rubric item; grouped=score contiguous groups of --group-size rules."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=3,
        help="Rules per call when --decomposition-mode grouped. Ignored for vector/per_rule.",
    )
    parser.add_argument(
        "--rubric-format",
        choices=("full", "compact", "description_only", "title_only"),
        default="full",
        help=(
            "Rubric text ablation. full=title+description; compact strips category boilerplate; "
            "description_only/title_only test whether RaR drop is driven by long rubric text."
        ),
    )
    parser.add_argument(
        "--max-rubric-rule-chars",
        type=int,
        default=0,
        help="Optional per-rule character cap after rubric formatting. 0 disables truncation.",
    )
    parser.add_argument(
        "--candidate-ids",
        nargs="*",
        default=None,
        help="Optional candidate_id allowlist. Example: --candidate-ids reference_answer",
    )
    parser.add_argument(
        "--question-ids-file",
        default="",
        help="Optional held-out question id filter. Supports one id per line or JSON with heldout_question_ids/question_ids.",
    )
    return parser.parse_args()


def load_response_bank(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_question_ids(path: str) -> set[str]:
    if not path:
        return set()
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    if raw[0] in "[{":
        data = json.loads(raw)
        if isinstance(data, list):
            return {str(item) for item in data}
        if isinstance(data, dict):
            for key in ("heldout_question_ids", "question_ids", "eval_question_ids"):
                values = data.get(key)
                if isinstance(values, list):
                    return {str(item) for item in values}
        raise ValueError(f"Could not find question ids in JSON file: {path}")
    return {line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_criteria_boilerplate(text: str) -> str:
    return re.sub(
        r"\b(?:Essential|Important|Optional|Pitfall)\s+Criteria\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )


def maybe_truncate_rule(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def format_rubric_rules(
    rubric: list[dict[str, Any]],
    rubric_format: str = "full",
    max_rule_chars: int = 0,
) -> str:
    lines = []
    for idx, item in enumerate(rubric, start=1):
        title = normalize_space(str(item.get("title", "") or ""))
        description = normalize_space(str(item.get("description", "") or ""))
        if rubric_format == "title_only":
            text = title or description
        elif rubric_format == "description_only":
            text = description or title
        else:
            if title and description:
                text = f"{title}: {description}"
            else:
                text = title or description
            if rubric_format == "compact":
                text = strip_criteria_boilerplate(text)
        text = maybe_truncate_rule(normalize_space(text), max_rule_chars)
        lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def build_prompt(
    question: str,
    rubric: list[dict[str, Any]],
    response: str,
    prompt_style: str,
    rubric_format: str = "full",
    max_rule_chars: int = 0,
) -> str:
    template = VERDICT_ONLY_PROMPT_TEMPLATE if prompt_style == "verdict_only" else PROMPT_TEMPLATE
    return template.format(
        question=question,
        response=response,
        rubric_rules=format_rubric_rules(rubric, rubric_format, max_rule_chars),
        num_rules=len(rubric),
    )


def parse_verdict_vector(text: str, num_rules: int, prompt_style: str) -> list[int] | None:
    if not text:
        return None
    if prompt_style == "verdict_only":
        patterns = [
            re.compile(r"^\s*Rule\s*(\d+)\s*:\s*(Yes|No|Y|N)\s*$", flags=re.IGNORECASE),
            re.compile(r"^\s*Rule\s*(\d+)\s*[:.)-]\s*(Yes|No|Y|N)\b.*$", flags=re.IGNORECASE),
            re.compile(
                r"^\s*Rule\s*(\d+)\s*:\s*.*?\bVerdict\s*:\s*(Yes|No|Y|N)\s*\.?\s*$",
                flags=re.IGNORECASE,
            ),
        ]
        verdict_by_rule: dict[int, int] = {}
        expected_rule = 1
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = None
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    break
            if not match:
                if verdict_by_rule:
                    verdict_by_rule = {}
                    expected_rule = 1
                continue
            rule_id = int(match.group(1))
            token = match.group(2).strip().lower()
            verdict = 1 if token in {"yes", "y"} else 0
            if rule_id == expected_rule:
                verdict_by_rule[rule_id] = verdict
                expected_rule += 1
                if expected_rule == num_rules + 1:
                    return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]
            elif rule_id == 1:
                verdict_by_rule = {1: verdict}
                expected_rule = 2
            else:
                verdict_by_rule = {}
                expected_rule = 1
        return None
    else:
        verdict_by_rule: dict[int, int] = {}
        patterns = [
            re.compile(
                r"^\s*Rule\s*(\d+)\s*:\s*.*?\bVerdict\s*:\s*(Yes|No|Y|N)\s*$",
                flags=re.IGNORECASE,
            )
        ]
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = None
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    break
            if not match:
                continue
            rule_id = int(match.group(1))
            token = match.group(2).strip().lower()
            verdict_by_rule[rule_id] = 1 if token in {"yes", "y"} else 0
        if len(verdict_by_rule) != num_rules:
            return None
        if set(verdict_by_rule) != set(range(1, num_rules + 1)):
            return None
        return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]


def calculate_auto_batch_size(model_config: Any, max_prompt_length: int, max_new_tokens: int) -> int:
    if not torch.cuda.is_available():
        return 1
    free_mem, _ = torch.cuda.mem_get_info()
    usable_mem = free_mem - (SAFETY_BUFFER_GB * 1024**3)
    if usable_mem <= 0:
        return 1
    n_layers = getattr(model_config, "num_hidden_layers", 24)
    n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 8))
    n_heads = max(1, getattr(model_config, "num_attention_heads", 32))
    head_dim = getattr(model_config, "hidden_size", 4096) // n_heads
    bytes_per_token = 4 * n_layers * n_kv_heads * head_dim
    denom = (max_prompt_length + max_new_tokens) * max(1, bytes_per_token)
    auto_bs = usable_mem // denom
    if auto_bs <= 0:
        return 1
    return min(MAX_BATCH_SIZE, max(1, 2 ** int(math.log2(auto_bs))))


def load_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except AttributeError as exc:
        message = str(exc)
        if "'list' object has no attribute 'keys'" not in message:
            raise
        config_path = Path(model_path) / "tokenizer_config.json"
        if not config_path.is_file():
            raise
        config = json.loads(config_path.read_text(encoding="utf-8"))
        extra_special_tokens = config.get("extra_special_tokens")
        if not isinstance(extra_special_tokens, list):
            raise
        print(
            "[WARN] tokenizer_config.json has legacy list-valued extra_special_tokens; "
            "retrying tokenizer load with extra_special_tokens={}.",
            flush=True,
        )
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            extra_special_tokens={},
        )


def aggregate_score(rubric: list[dict[str, Any]], verdicts: list[int]) -> tuple[float, list[float]]:
    weights = [abs(float(item.get("weight", 1.0) if item.get("weight", 1.0) is not None else 1.0)) for item in rubric]
    weighted_sum = sum(verdict * weight for verdict, weight in zip(verdicts, weights, strict=True))
    total_positive_weight = sum(weight for weight in weights if weight > 0)
    return (weighted_sum / total_positive_weight if total_positive_weight > 0 else 0.0), weights


def make_reward_result(rubric: list[dict[str, Any]], completion: str, prompt_style: str) -> dict[str, Any]:
    verdicts = parse_verdict_vector(completion, len(rubric), prompt_style)
    if verdicts is None:
        weights = [abs(float(item.get("weight", 1.0) if item.get("weight", 1.0) is not None else 1.0)) for item in rubric]
        return {
            "score": 0.0,
            "criterion_scores": [0.0] * len(rubric),
            "criterion_weights": weights,
            "criterion_parse_ok": [False] * len(rubric),
            "num_parse_failures": len(rubric),
            "raw_num_parse_failures": len(rubric),
            "raw_parse_success": False,
            "fallback_used": False,
            "judge_attempts": 1,
            "parse_source": "hf_direct_parse_failed",
            "raw_judge_response": completion,
        }
    score, weights = aggregate_score(rubric, verdicts)
    return {
        "score": score,
        "criterion_scores": [float(item) for item in verdicts],
        "criterion_weights": weights,
        "criterion_parse_ok": [True] * len(rubric),
        "num_parse_failures": 0,
        "raw_num_parse_failures": 0,
        "raw_parse_success": True,
        "fallback_used": False,
        "judge_attempts": 1,
        "parse_source": "hf_direct",
        "raw_judge_response": completion,
    }


def flatten(
    response_bank: list[dict[str, Any]],
    candidate_ids: set[str] | None = None,
    prompt_style: str = "bench",
    rubric_format: str = "full",
    max_rule_chars: int = 0,
) -> list[dict[str, Any]]:
    flat = []
    for q_index, question in enumerate(response_bank):
        for c_index, candidate in enumerate(question["candidates"]):
            if candidate_ids is not None and candidate["candidate_id"] not in candidate_ids:
                continue
            flat.append(
                {
                    "question_index": q_index,
                    "candidate_index": c_index,
                    "question_id": question["question_id"],
                    "candidate_id": candidate["candidate_id"],
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "rubric": question["rubric"],
                    "response": candidate["response"],
                    "prompt": build_prompt(
                        question["question"],
                        rubric=question["rubric"],
                        response=candidate["response"],
                        prompt_style=prompt_style,
                        rubric_format=rubric_format,
                        max_rule_chars=max_rule_chars,
                    ),
                }
            )
    return flat


def rubric_groups(rubric: list[dict[str, Any]], decomposition_mode: str, group_size: int) -> list[tuple[int, int]]:
    if decomposition_mode == "vector":
        return [(0, len(rubric))]
    if decomposition_mode == "per_rule":
        return [(idx, idx + 1) for idx in range(len(rubric))]
    if group_size <= 0:
        raise ValueError("--group-size must be positive when --decomposition-mode grouped")
    return [(start, min(start + group_size, len(rubric))) for start in range(0, len(rubric), group_size)]


def build_judge_requests(
    sample_items: list[dict[str, Any]],
    decomposition_mode: str,
    group_size: int,
    prompt_style: str,
    rubric_format: str,
    max_rule_chars: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(sample_items):
        groups = rubric_groups(sample["rubric"], decomposition_mode, group_size)
        for group_index, (start, end) in enumerate(groups):
            rubric_slice = sample["rubric"][start:end]
            requests.append(
                {
                    "sample_index": sample_index,
                    "group_index": group_index,
                    "group_start": start,
                    "group_end": end,
                    "rubric": rubric_slice,
                    "prompt": build_prompt(
                        sample["question"],
                        rubric=rubric_slice,
                        response=sample["response"],
                        prompt_style=prompt_style,
                        rubric_format=rubric_format,
                        max_rule_chars=max_rule_chars,
                    ),
                }
            )
    return requests


def combine_group_reward_results(
    sample: dict[str, Any],
    group_results: list[dict[str, Any]],
    group_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    full_rubric = sample["rubric"]
    scores: list[float | None] = [None] * len(full_rubric)
    parse_ok: list[bool] = [False] * len(full_rubric)
    raw_responses: list[dict[str, Any]] = []
    raw_parse_success = True
    total_parse_failures = 0
    total_raw_parse_failures = 0

    for request, result in sorted(zip(group_requests, group_results), key=lambda item: item[0]["group_start"]):
        start = int(request["group_start"])
        group_scores = result.get("criterion_scores", [])
        group_parse_ok = result.get("criterion_parse_ok", [])
        for offset, score in enumerate(group_scores):
            target = start + offset
            if 0 <= target < len(scores):
                scores[target] = float(score)
        for offset, ok in enumerate(group_parse_ok):
            target = start + offset
            if 0 <= target < len(parse_ok):
                parse_ok[target] = bool(ok)
        raw_parse_success = raw_parse_success and bool(result.get("raw_parse_success", False))
        total_parse_failures += int(result.get("num_parse_failures", 0))
        total_raw_parse_failures += int(result.get("raw_num_parse_failures", result.get("num_parse_failures", 0)))
        raw_responses.append(
            {
                "group_index": request["group_index"],
                "group_start": request["group_start"],
                "group_end": request["group_end"],
                "parse_source": result.get("parse_source", ""),
                "raw_judge_response": result.get("raw_judge_response", ""),
            }
        )

    final_scores = [float(score) if score is not None else 0.0 for score in scores]
    score, weights = aggregate_score(full_rubric, [int(round(item)) for item in final_scores])
    return {
        "score": score,
        "criterion_scores": final_scores,
        "criterion_weights": weights,
        "criterion_parse_ok": parse_ok,
        "num_parse_failures": total_parse_failures,
        "raw_num_parse_failures": total_raw_parse_failures,
        "raw_parse_success": raw_parse_success and total_raw_parse_failures == 0,
        "fallback_used": False,
        "judge_attempts": len(group_results),
        "parse_source": "hf_direct_decomposed" if len(group_results) > 1 else "hf_direct",
        "raw_judge_response": "\n\n".join(item["raw_judge_response"] for item in raw_responses),
        "raw_judge_responses_by_group": raw_responses,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    response_bank = load_response_bank(args.response_bank)
    question_id_filter = load_question_ids(args.question_ids_file)
    if question_id_filter:
        response_bank = [record for record in response_bank if str(record.get("question_id", "")) in question_id_filter]
        if not response_bank:
            raise SystemExit(f"ERROR: no questions selected by --question-ids-file {args.question_ids_file}")
    candidate_id_filter = set(args.candidate_ids) if args.candidate_ids else None
    sample_items = flatten(
        response_bank,
        candidate_id_filter,
        args.prompt_style,
        args.rubric_format,
        args.max_rubric_rule_chars,
    )
    if not sample_items:
        raise SystemExit("ERROR: no candidates selected from response bank.")
    judge_requests = build_judge_requests(
        sample_items,
        args.decomposition_mode,
        args.group_size,
        args.prompt_style,
        args.rubric_format,
        args.max_rubric_rule_chars,
    )
    if not judge_requests:
        raise SystemExit("ERROR: no judge requests built from selected candidates.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_bank.jsonl"
    summary_path = output_dir / "summary.json"

    tokenizer = load_tokenizer(args.judge_model, args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    batch_size = args.batch_size or calculate_auto_batch_size(
        model.config,
        args.max_prompt_length,
        args.max_new_tokens,
    )
    print(f"[INFO] judge_model={args.judge_model}")
    print(f"[INFO] samples={len(sample_items)} judge_requests={len(judge_requests)} batch_size={batch_size}")

    started_at = time.perf_counter()
    group_reward_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(judge_requests), batch_size):
            batch = judge_requests[start : start + batch_size]
            inputs = tokenizer(
                [item["prompt"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
            input_width = inputs.input_ids.shape[1]
            for row_idx, item in enumerate(batch):
                completion = tokenizer.decode(outputs[row_idx][input_width:], skip_special_tokens=True)
                group_reward_results.append(make_reward_result(item["rubric"], completion, args.prompt_style))

    sample_group_results: list[list[dict[str, Any]]] = [[] for _ in sample_items]
    sample_group_requests: list[list[dict[str, Any]]] = [[] for _ in sample_items]
    for request, result in zip(judge_requests, group_reward_results):
        sample_index = int(request["sample_index"])
        sample_group_requests[sample_index].append(request)
        sample_group_results[sample_index].append(result)

    reward_results = [
        combine_group_reward_results(sample, sample_group_results[idx], sample_group_requests[idx])
        for idx, sample in enumerate(sample_items)
    ]

    result_iter = iter(reward_results)
    enriched_records = []
    per_candidate_scores: dict[str, list[float]] = defaultdict(list)
    samples_parse_ok = 0
    total_parse_failures = 0
    total_judged_criteria = 0

    for question in response_bank:
        scored_candidates = []
        for candidate in question["candidates"]:
            if candidate_id_filter is not None and candidate["candidate_id"] not in candidate_id_filter:
                continue
            reward_result = next(result_iter)
            score = float(reward_result["score"])
            parse_failures = int(reward_result["num_parse_failures"])
            judged_items = len(reward_result["criterion_scores"]) or 1
            samples_parse_ok += int(parse_failures == 0)
            total_parse_failures += parse_failures
            total_judged_criteria += judged_items
            per_candidate_scores[candidate["candidate_id"]].append(score)
            scored_candidates.append(
                {
                    **candidate,
                    "score": score,
                    "num_parse_failures": parse_failures,
                    "reward_result": reward_result,
                }
            )
        enriched_records.append({**question, "candidates": scored_candidates})

    with scored_path.open("w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "label": args.label or args.judge_model,
        "judge_model": args.judge_model,
        "judge_backend": "hf_direct_bench_style",
        "response_bank": args.response_bank,
        "question_ids_file": args.question_ids_file or None,
        "question_ids_filter_count": len(question_id_filter),
        "candidate_ids_filter": sorted(candidate_id_filter) if candidate_id_filter else None,
        "num_questions": len(response_bank),
        "num_candidates_scored": len(sample_items),
        "num_judge_requests": len(judge_requests),
        "decomposition_mode": args.decomposition_mode,
        "group_size": args.group_size,
        "batch_size": batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
        "rubric_format": args.rubric_format,
        "max_rubric_rule_chars": args.max_rubric_rule_chars,
        "samples_parse_ok": samples_parse_ok,
        "samples_parse_failed": len(sample_items) - samples_parse_ok,
        "sample_parse_success_rate": samples_parse_ok / len(sample_items) if sample_items else 1.0,
        "raw_samples_parse_ok": samples_parse_ok,
        "raw_sample_parse_success_rate": samples_parse_ok / len(sample_items) if sample_items else 1.0,
        "fallback_used": 0,
        "fallback_rate": 0.0,
        "mean_judge_attempts": len(judge_requests) / len(sample_items) if sample_items else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_candidate": elapsed_seconds / len(sample_items) if sample_items else 0.0,
        "seconds_per_judge_request": elapsed_seconds / len(judge_requests) if judge_requests else 0.0,
        "total_parse_failures": total_parse_failures,
        "parse_success_rate": (
            1.0 - total_parse_failures / total_judged_criteria if total_judged_criteria else 1.0
        ),
        "per_candidate_summary": {
            candidate_id: {
                "mean_score": mean(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "num_examples": len(scores),
            }
            for candidate_id, scores in sorted(per_candidate_scores.items())
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("=======================================")
    print(f"Judge label:           {summary['label']}")
    print(f"Judge backend:         {summary['judge_backend']}")
    print(f"Candidates scored:     {summary['num_candidates_scored']}")
    print(f"Raw sample parse:      {summary['raw_sample_parse_success_rate']:.4f}")
    print(f"Parse success:         {summary['parse_success_rate']:.4f}")
    print(f"Scored bank:           {scored_path}")
    print(f"Summary:               {summary_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
