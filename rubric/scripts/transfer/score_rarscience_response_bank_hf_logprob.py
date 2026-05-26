#!/usr/bin/env python3
"""Score RaR-Science rubric criteria with forced-choice Yes/No logprobs.

This is a decoding-free judge interface: for each criterion, compute the
teacher-forced continuation likelihood of " Yes" and " No" after a criterion
prompt, then convert the logprob margin into a binary verdict and a soft score.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGPROB_PROMPT_TEMPLATE = """
You are an impartial rubric grader.

Instruction:
{question}

Candidate Response:
{response}

Rubric Criterion:
{criterion}

Question:
Does the candidate response satisfy this criterion?

Answer with exactly one word: Yes or No.

Answer:
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--rubric-format",
        choices=("full", "compact", "description_only", "title_only", "canonical"),
        default="canonical",
        help=(
            "Criterion text format. canonical rewrites category/polarity into a direct satisfy/avoid statement."
        ),
    )
    parser.add_argument("--max-rubric-rule-chars", type=int, default=0)
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--question-ids-file", default="")
    parser.add_argument(
        "--yes-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to P(Yes) for binary criterion_scores.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("probability", "binary"),
        default="probability",
        help="Use soft probabilities or thresholded verdicts for aggregate candidate score.",
    )
    parser.add_argument(
        "--choice-normalization",
        choices=("sum", "mean"),
        default="sum",
        help="How to normalize multi-token verbalizer logprobs.",
    )
    parser.add_argument(
        "--verbalizer-style",
        choices=("space", "multi"),
        default="space",
        help="space uses one continuation per class; multi log-sum-exp aggregates several verbalizers.",
    )
    return parser.parse_args()


def load_response_bank(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def category_for_rule(rule: dict[str, Any]) -> str:
    text = f"{rule.get('title', '')} {rule.get('description', '')}".lower()
    for category in ("essential", "important", "optional", "pitfall"):
        if category in text:
            return category
    return "unknown"


def format_criterion(rule: dict[str, Any], rubric_format: str, max_rule_chars: int) -> str:
    title = normalize_space(str(rule.get("title", "") or ""))
    description = normalize_space(str(rule.get("description", "") or ""))
    category = category_for_rule(rule)
    if rubric_format == "title_only":
        text = title or description
    elif rubric_format == "description_only":
        text = description or title
    else:
        if title and description:
            text = f"{title}: {description}"
        else:
            text = title or description
        text = strip_criteria_boilerplate(text) if rubric_format in {"compact", "canonical"} else text
        if rubric_format == "canonical":
            if category == "pitfall":
                text = f"The response avoids this pitfall: {text}"
            else:
                text = f"The response satisfies this criterion: {text}"
    return maybe_truncate_rule(normalize_space(text), max_rule_chars)


def build_prompt(question: str, response: str, criterion: str) -> str:
    return LOGPROB_PROMPT_TEMPLATE.format(question=question, response=response, criterion=criterion)


def load_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        config_path = Path(model_path) / "tokenizer_config.json"
        if not config_path.is_file():
            raise
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config.get("extra_special_tokens"), list):
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


def verbalizers(style: str) -> dict[str, list[str]]:
    if style == "multi":
        return {
            "yes": [" Yes", " yes", "Yes", "YES"],
            "no": [" No", " no", "No", "NO"],
        }
    return {"yes": [" Yes"], "no": [" No"]}


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def aggregate_score(rubric: list[dict[str, Any]], values: list[float]) -> tuple[float, list[float]]:
    weights = [abs(float(item.get("weight", 1.0) if item.get("weight", 1.0) is not None else 1.0)) for item in rubric]
    total_positive_weight = sum(weight for weight in weights if weight > 0)
    if total_positive_weight <= 0:
        return 0.0, weights
    return sum(value * weight for value, weight in zip(values, weights)) / total_positive_weight, weights


def flatten(
    response_bank: list[dict[str, Any]],
    candidate_ids: set[str] | None,
    rubric_format: str,
    max_rule_chars: int,
) -> list[dict[str, Any]]:
    flat = []
    for q_index, question in enumerate(response_bank):
        for c_index, candidate in enumerate(question.get("candidates", [])):
            if candidate_ids is not None and candidate.get("candidate_id") not in candidate_ids:
                continue
            for r_index, rule in enumerate(question.get("rubric", [])):
                criterion = format_criterion(rule, rubric_format, max_rule_chars)
                flat.append(
                    {
                        "question_index": q_index,
                        "candidate_index": c_index,
                        "rule_index": r_index,
                        "question_id": question.get("question_id"),
                        "candidate_id": candidate.get("candidate_id"),
                        "prompt": build_prompt(question.get("question", ""), candidate.get("response", ""), criterion),
                        "criterion": criterion,
                    }
                )
    return flat


def build_expanded_rows(
    rule_items: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
    verbalizer_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    expanded = []
    for item_index, item in enumerate(rule_items):
        prompt_ids = tokenizer(
            item["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        for label, choices in verbalizer_map.items():
            for verbalizer_index, choice in enumerate(choices):
                choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
                if not choice_ids:
                    raise ValueError(f"Empty verbalizer tokenization: {choice!r}")
                expanded.append(
                    {
                        "item_index": item_index,
                        "label": label,
                        "verbalizer_index": verbalizer_index,
                        "input_ids": prompt_ids + choice_ids,
                        "prompt_len": len(prompt_ids),
                        "choice_len": len(choice_ids),
                        "choice": choice,
                    }
                )
    return expanded


def model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def left_pad_contexts(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(row) for row in rows)
    input_tensor = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long)
    attn_tensor = torch.zeros((len(rows), max_len), dtype=torch.long)
    for row_index, row in enumerate(rows):
        row_tensor = torch.tensor(row, dtype=torch.long)
        input_tensor[row_index, -len(row) :] = row_tensor
        attn_tensor[row_index, -len(row) :] = 1
    return input_tensor, attn_tensor


def final_token_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        logits = output.logits
        if logits.ndim == 3:
            return logits[:, -1, :]
        return logits
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return output.logits[:, -1, :]


def score_expanded_rows(
    expanded_rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    choice_normalization: str,
) -> list[float]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_device = model_input_device(model)
    row_scores = [0.0] * len(expanded_rows)
    for start in range(0, len(expanded_rows), batch_size):
        batch = expanded_rows[start : start + batch_size]
        max_choice_len = max(int(row["choice_len"]) for row in batch)
        totals = [0.0 for _ in batch]
        for rel_pos in range(max_choice_len):
            context_rows: list[list[int]] = []
            target_ids: list[int] = []
            active_offsets: list[int] = []
            for offset, row in enumerate(batch):
                choice_len = int(row["choice_len"])
                if rel_pos >= choice_len:
                    continue
                prompt_len = int(row["prompt_len"])
                ids = row["input_ids"]
                context_rows.append(ids[: prompt_len + rel_pos])
                target_ids.append(int(ids[prompt_len + rel_pos]))
                active_offsets.append(offset)
            if not context_rows:
                continue
            input_tensor, mask_tensor = left_pad_contexts(context_rows, int(pad_id))
            input_tensor = input_tensor.to(input_device)
            mask_tensor = mask_tensor.to(input_device)
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=input_device)
            logits = final_token_logits(model, input_tensor, mask_tensor)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_scores = log_probs[torch.arange(len(target_ids), device=input_device), target_tensor]
            for offset, token_score in zip(active_offsets, token_scores.detach().cpu().tolist()):
                totals[offset] += float(token_score)
            del input_tensor, mask_tensor, target_tensor, logits, log_probs, token_scores
        for offset, row in enumerate(batch):
            total = totals[offset]
            if choice_normalization == "mean":
                total /= max(1, int(row["choice_len"]))
            row_scores[start + offset] = total
    return row_scores


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
    rule_items = flatten(response_bank, candidate_id_filter, args.rubric_format, args.max_rubric_rule_chars)
    if not rule_items:
        raise SystemExit("ERROR: no rubric criteria selected from response bank.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_bank.jsonl"
    summary_path = output_dir / "summary.json"

    tokenizer = load_tokenizer(args.judge_model, args.trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    verbalizer_map = verbalizers(args.verbalizer_style)
    expanded_rows = build_expanded_rows(rule_items, tokenizer, args.max_prompt_length, verbalizer_map)

    started_at = time.perf_counter()
    print(f"[INFO] judge_model={args.judge_model}")
    print(f"[INFO] criteria={len(rule_items)} expanded_choices={len(expanded_rows)} batch_size={args.batch_size}")
    with torch.inference_mode():
        expanded_scores = score_expanded_rows(
            expanded_rows,
            model,
            tokenizer,
            args.batch_size,
            args.choice_normalization,
        )

    per_item_label_scores: list[dict[str, list[float]]] = [
        {"yes": [], "no": []} for _ in rule_items
    ]
    per_item_choices: list[dict[str, list[dict[str, Any]]]] = [
        {"yes": [], "no": []} for _ in rule_items
    ]
    for row, score in zip(expanded_rows, expanded_scores):
        item_index = int(row["item_index"])
        label = row["label"]
        per_item_label_scores[item_index][label].append(score)
        per_item_choices[item_index][label].append(
            {"choice": row["choice"], "logprob": score}
        )

    rule_results = []
    for item, label_scores, choices in zip(rule_items, per_item_label_scores, per_item_choices):
        yes_logprob = logsumexp(label_scores["yes"])
        no_logprob = logsumexp(label_scores["no"])
        margin = yes_logprob - no_logprob
        yes_prob = sigmoid(margin)
        verdict = 1.0 if yes_prob >= args.yes_threshold else 0.0
        rule_results.append(
            {
                **item,
                "yes_logprob": yes_logprob,
                "no_logprob": no_logprob,
                "logprob_margin": margin,
                "yes_prob": yes_prob,
                "verdict": verdict,
                "verbalizer_scores": choices,
            }
        )

    grouped_results: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for result in rule_results:
        grouped_results[(int(result["question_index"]), int(result["candidate_index"]))].append(result)

    enriched_records = []
    per_candidate_scores: dict[str, list[float]] = defaultdict(list)
    total_criteria = 0
    for q_index, question in enumerate(response_bank):
        scored_candidates = []
        for c_index, candidate in enumerate(question.get("candidates", [])):
            if candidate_id_filter is not None and candidate.get("candidate_id") not in candidate_id_filter:
                continue
            candidate_rule_results = sorted(grouped_results[(q_index, c_index)], key=lambda row: int(row["rule_index"]))
            rubric = question.get("rubric", [])
            if len(candidate_rule_results) != len(rubric):
                raise ValueError(
                    f"Internal error: expected {len(rubric)} rule results for "
                    f"{question.get('question_id')} / {candidate.get('candidate_id')}, "
                    f"found {len(candidate_rule_results)}"
                )
            verdicts = [float(row["verdict"]) for row in candidate_rule_results]
            yes_probs = [float(row["yes_prob"]) for row in candidate_rule_results]
            score_values = yes_probs if args.score_mode == "probability" else verdicts
            score, weights = aggregate_score(rubric, score_values)
            binary_score, _ = aggregate_score(rubric, verdicts)
            total_criteria += len(candidate_rule_results)
            per_candidate_scores[str(candidate.get("candidate_id"))].append(score)
            reward_result = {
                "score": score,
                "binary_score": binary_score,
                "criterion_scores": verdicts,
                "criterion_weights": weights,
                "criterion_parse_ok": [True] * len(verdicts),
                "num_parse_failures": 0,
                "raw_num_parse_failures": 0,
                "raw_parse_success": True,
                "fallback_used": False,
                "judge_attempts": len(verdicts),
                "parse_source": "hf_forced_choice_logprob",
                "score_mode": args.score_mode,
                "yes_threshold": args.yes_threshold,
                "criterion_yes_probs": yes_probs,
                "criterion_logprob_margins": [float(row["logprob_margin"]) for row in candidate_rule_results],
                "criterion_yes_logprobs": [float(row["yes_logprob"]) for row in candidate_rule_results],
                "criterion_no_logprobs": [float(row["no_logprob"]) for row in candidate_rule_results],
                "criterion_text": [row["criterion"] for row in candidate_rule_results],
                "verbalizer_style": args.verbalizer_style,
                "choice_normalization": args.choice_normalization,
            }
            scored_candidates.append(
                {
                    **candidate,
                    "score": score,
                    "num_parse_failures": 0,
                    "reward_result": reward_result,
                }
            )
        enriched_records.append({**question, "candidates": scored_candidates})

    with scored_path.open("w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed_seconds = time.perf_counter() - started_at
    num_candidates = sum(len(record["candidates"]) for record in enriched_records)
    summary = {
        "label": args.label or args.judge_model,
        "judge_model": args.judge_model,
        "judge_backend": "hf_forced_choice_logprob",
        "response_bank": args.response_bank,
        "question_ids_file": args.question_ids_file or None,
        "question_ids_filter_count": len(question_id_filter),
        "candidate_ids_filter": sorted(candidate_id_filter) if candidate_id_filter else None,
        "num_questions": len(response_bank),
        "num_candidates_scored": num_candidates,
        "num_criteria_scored": total_criteria,
        "num_choice_forwards": len(expanded_rows),
        "batch_size": args.batch_size,
        "max_prompt_length": args.max_prompt_length,
        "rubric_format": args.rubric_format,
        "max_rubric_rule_chars": args.max_rubric_rule_chars,
        "score_mode": args.score_mode,
        "yes_threshold": args.yes_threshold,
        "choice_normalization": args.choice_normalization,
        "verbalizer_style": args.verbalizer_style,
        "samples_parse_ok": num_candidates,
        "samples_parse_failed": 0,
        "sample_parse_success_rate": 1.0,
        "raw_samples_parse_ok": num_candidates,
        "raw_sample_parse_success_rate": 1.0,
        "fallback_used": 0,
        "fallback_rate": 0.0,
        "mean_judge_attempts": total_criteria / num_candidates if num_candidates else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_candidate": elapsed_seconds / num_candidates if num_candidates else 0.0,
        "seconds_per_criterion": elapsed_seconds / total_criteria if total_criteria else 0.0,
        "total_parse_failures": 0,
        "parse_success_rate": 1.0,
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
    print(f"Criteria scored:       {summary['num_criteria_scored']}")
    print(f"Parse success:         {summary['parse_success_rate']:.4f}")
    print(f"Elapsed seconds:       {elapsed_seconds:.2f}")
    print(f"Scored bank:           {scored_path}")
    print(f"Summary:               {summary_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
