#!/usr/bin/env python3
"""Score OpenRubricBench criteria with forced-choice Yes/No logprobs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


HELPER_PATH = Path(__file__).with_name("score_bench_hf_probe.py")
HELPER_SPEC = importlib.util.spec_from_file_location("bench_probe_helper", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load Bench helper from {HELPER_PATH}")
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HELPER
HELPER_SPEC.loader.exec_module(HELPER)

BenchSample = HELPER.BenchSample
compute_metrics = HELPER.compute_metrics
flatten_samples = HELPER.flatten_samples
format_criterion = HELPER.format_criterion
safe_div = HELPER.safe_div
split_question_ids = HELPER.split_question_ids
vector_to_yn = HELPER.vector_to_yn
verdicts_to_score = HELPER.verdicts_to_score
write_csv = HELPER.write_csv


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
    parser.add_argument("--bench-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--train-questions", type=int, default=350)
    parser.add_argument("--dev-questions", type=int, default=50)
    parser.add_argument("--eval-split", choices=("all", "heldout"), default="heldout")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--rubric-format", choices=("canonical", "full"), default="canonical")
    parser.add_argument("--yes-threshold", type=float, default=0.5)
    parser.add_argument("--score-mode", choices=("probability", "binary"), default="binary")
    parser.add_argument("--choice-normalization", choices=("sum", "mean"), default="sum")
    parser.add_argument("--verbalizer-style", choices=("space", "multi"), default="space")
    return parser.parse_args()


def load_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    cache_dir = os.environ.get("HF_HOME")
    token = os.environ.get("HF_TOKEN")
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        config_path = Path(model_path) / "tokenizer_config.json"
        if not config_path.is_file():
            raise
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config.get("extra_special_tokens"), list):
            raise
        kwargs["extra_special_tokens"] = {}
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_model(model_path: str, trust_remote_code: bool) -> Any:
    cache_dir = os.environ.get("HF_HOME")
    token = os.environ.get("HF_TOKEN")
    kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def build_prompt(question: str, response: str, criterion: str) -> str:
    return LOGPROB_PROMPT_TEMPLATE.format(question=question, response=response, criterion=criterion)


def build_criterion_items(
    samples: list[BenchSample],
    question_ids: set[str],
    split: str,
    rubric_format: str,
) -> tuple[list[dict[str, Any]], list[BenchSample]]:
    selected = [sample for sample in samples if sample.question_id in question_ids]
    items = []
    for sample in selected:
        for rule_index, rule in enumerate(sample.rules):
            criterion = format_criterion(rule, rubric_format)
            items.append(
                {
                    "split": split,
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "answer_id": sample.answer_id,
                    "rule_index": rule_index,
                    "prompt": build_prompt(sample.question, sample.response, criterion),
                    "label": int(sample.gold_verdicts[rule_index]),
                    "criterion": criterion,
                    "rule_type": "hard" if rule.get("type") == "hard" else "soft",
                }
            )
    return items, selected


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


def model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def encode_choice_sequences(
    tokenizer: Any,
    prompts: list[str],
    choices: list[str],
    max_prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    encoded_rows = []
    spans = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for prompt, choice in zip(prompts, choices):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if not choice_ids:
            raise ValueError(f"Empty verbalizer tokenization for choice={choice!r}")
        input_ids = prompt_ids + choice_ids
        encoded_rows.append(input_ids)
        spans.append((len(prompt_ids), len(choice_ids)))
    max_len = max(len(row) for row in encoded_rows)
    input_tensor = torch.full((len(encoded_rows), max_len), int(pad_id), dtype=torch.long)
    attn_tensor = torch.zeros((len(encoded_rows), max_len), dtype=torch.long)
    for row_index, row in enumerate(encoded_rows):
        input_tensor[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attn_tensor[row_index, : len(row)] = 1
    return input_tensor, attn_tensor, spans


def encode_choice_parts(
    tokenizer: Any,
    prompts: list[str],
    choices: list[str],
    max_prompt_length: int,
) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for prompt, choice in zip(prompts, choices):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if not choice_ids:
            raise ValueError(f"Empty verbalizer tokenization for choice={choice!r}")
        encoded.append((prompt_ids, choice_ids))
    return encoded


def left_pad_contexts(
    rows: list[list[int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def score_choice_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    choices: list[str],
    max_prompt_length: int,
    choice_normalization: str,
) -> list[float]:
    encoded = encode_choice_parts(tokenizer, prompts, choices, max_prompt_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_device = model_input_device(model)
    scores = [0.0 for _ in encoded]
    choice_lengths = [len(choice_ids) for _, choice_ids in encoded]
    max_choice_len = max(choice_lengths)
    for offset in range(max_choice_len):
        context_rows: list[list[int]] = []
        target_ids: list[int] = []
        active_indices: list[int] = []
        for row_index, (prompt_ids, choice_ids) in enumerate(encoded):
            if offset >= len(choice_ids):
                continue
            context_rows.append(prompt_ids + choice_ids[:offset])
            target_ids.append(int(choice_ids[offset]))
            active_indices.append(row_index)
        if not context_rows:
            continue
        input_ids, attention_mask = left_pad_contexts(context_rows, int(pad_id))
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=input_device)
        with torch.inference_mode():
            logits = final_token_logits(model, input_ids, attention_mask)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_scores = log_probs[torch.arange(len(target_ids), device=input_device), target_tensor]
        for active_index, token_score in zip(active_indices, token_scores.detach().cpu().tolist()):
            scores[active_index] += float(token_score)
        del input_ids, attention_mask, target_tensor, logits, log_probs, token_scores
    if choice_normalization == "mean":
        scores = [score / float(choice_len) for score, choice_len in zip(scores, choice_lengths)]
    return scores


def score_items_logprob(
    items: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_prompt_length: int,
    choice_normalization: str,
    verbalizer_style: str,
) -> list[dict[str, Any]]:
    class_verbalizers = verbalizers(verbalizer_style)
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts: list[str] = []
        choices: list[str] = []
        owner: list[tuple[int, str]] = []
        for local_index, item in enumerate(batch):
            for class_name, class_choices in class_verbalizers.items():
                for choice in class_choices:
                    prompts.append(item["prompt"])
                    choices.append(choice)
                    owner.append((local_index, class_name))
        flat_scores = score_choice_batch(
            model,
            tokenizer,
            prompts,
            choices,
            max_prompt_length,
            choice_normalization,
        )
        grouped = [{"yes": [], "no": []} for _ in batch]
        for (local_index, class_name), score in zip(owner, flat_scores):
            grouped[local_index][class_name].append(score)
        for item, class_scores in zip(batch, grouped):
            yes_logprob = logsumexp(class_scores["yes"])
            no_logprob = logsumexp(class_scores["no"])
            yes_prob = float(1.0 / (1.0 + math.exp(no_logprob - yes_logprob)))
            outputs.append(
                {
                    **item,
                    "yes_logprob": yes_logprob,
                    "no_logprob": no_logprob,
                    "yes_prob": yes_prob,
                }
            )
        if torch.cuda.is_available() and start % max(batch_size * 20, 1) == 0:
            torch.cuda.empty_cache()
    return outputs


def build_prediction_rows(
    eval_samples: list[BenchSample],
    scored_items: list[dict[str, Any]],
    threshold: float,
    label: str,
    score_mode: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored_items:
        verdict = 1 if float(item["yes_prob"]) >= threshold else 0
        grouped[item["sample_id"]].append({**item, "verdict": verdict})

    rows = []
    for sample in eval_samples:
        rule_rows = sorted(grouped[sample.sample_id], key=lambda row: int(row["rule_index"]))
        pred_verdicts = [int(row["verdict"]) for row in rule_rows]
        score_values = [float(row["yes_prob"]) if score_mode == "probability" else int(row["verdict"]) for row in rule_rows]
        if score_mode == "probability":
            pred_total = float(sum(value * weight for value, weight in zip(score_values, sample.rule_weights)))
        else:
            pred_total = verdicts_to_score(pred_verdicts, sample.rule_weights)
        rule_matches = sum(int(a == b) for a, b in zip(sample.gold_verdicts, pred_verdicts))
        weighted_rule_matches = sum(
            weight for gold, pred, weight in zip(sample.gold_verdicts, pred_verdicts, sample.rule_weights) if gold == pred
        )
        rows.append(
            {
                "model": label,
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "answer_id": sample.answer_id,
                "num_rules": len(sample.rules),
                "weight_sum": int(sum(sample.rule_weights)),
                "gold_verdicts": sample.gold_verdicts,
                "pred_verdicts": pred_verdicts,
                "gold_vector_yn": vector_to_yn(sample.gold_verdicts),
                "pred_vector_yn": vector_to_yn(pred_verdicts),
                "gold_total_score": sample.gold_total_score,
                "pred_total_score": pred_total,
                "parse_ok": True,
                "vector_exact": pred_verdicts == sample.gold_verdicts,
                "rule_matches": int(rule_matches),
                "weighted_rule_matches": int(weighted_rule_matches),
                "score_exact": pred_total == sample.gold_total_score,
                "raw_output": "hf_logprob",
                "first_line_output": "hf_logprob",
                "criterion_yes_probs": [float(row["yes_prob"]) for row in rule_rows],
                "criterion_yes_logprobs": [float(row["yes_logprob"]) for row in rule_rows],
                "criterion_no_logprobs": [float(row["no_logprob"]) for row in rule_rows],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started_at = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_data = json.loads(Path(args.bench_json).read_text(encoding="utf-8"))
    samples = flatten_samples(bench_data)
    if not samples:
        raise SystemExit("ERROR: no Bench samples found.")

    train_ids, dev_ids, heldout_ids = split_question_ids(samples, args.train_questions, args.dev_questions, args.seed)
    eval_ids = {sample.question_id for sample in samples} if args.eval_split == "all" else heldout_ids
    eval_items, eval_samples = build_criterion_items(samples, eval_ids, args.eval_split, args.rubric_format)
    if not eval_items:
        raise SystemExit("ERROR: no eval items found.")

    tokenizer = load_tokenizer(args.judge_model, args.trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.judge_model, args.trust_remote_code)
    model.eval()

    print(f"[INFO] judge_model={args.judge_model}", flush=True)
    print(
        f"[INFO] split train/dev/eval_questions={len(train_ids)}/{len(dev_ids)}/{len(eval_ids)} "
        f"criteria={len(eval_items)}",
        flush=True,
    )
    scored_items = score_items_logprob(
        eval_items,
        model,
        tokenizer,
        args.batch_size,
        args.max_prompt_length,
        args.choice_normalization,
        args.verbalizer_style,
    )
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    label = args.label or args.judge_model
    pred_rows = build_prediction_rows(scored_items=scored_items, eval_samples=eval_samples, threshold=args.yes_threshold, label=label, score_mode=args.score_mode)
    metrics = compute_metrics(pred_rows)
    elapsed_seconds = time.perf_counter() - started_at
    metric_row = {
        "model": label,
        **metrics,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": safe_div(elapsed_seconds, len(eval_samples)),
        "judge_model": args.judge_model,
        "judge_backend": "hf_logprob",
        "train_questions": args.train_questions,
        "dev_questions": args.dev_questions,
        "eval_split": args.eval_split,
        "rubric_format": args.rubric_format,
        "yes_threshold": args.yes_threshold,
        "score_mode": args.score_mode,
        "choice_normalization": args.choice_normalization,
        "verbalizer_style": args.verbalizer_style,
    }

    write_csv(output_dir / "report.csv", [metric_row])
    write_csv(output_dir / "overall_metrics.csv", [metric_row])
    write_csv(output_dir / "per_sample_predictions.csv", pred_rows)
    (output_dir / "summary.json").write_text(json.dumps(metric_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Label:                {label}")
    print(f"Bench samples scored: {len(eval_samples)}")
    print(f"Parse success:        {metrics['parse_success_rate']:.4f}")
    print(f"Weighted acc:         {metrics['rule_weighted_accuracy_all_samples']:.4f}")
    print(f"Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"Pairwise acc:         {metrics['pairwise_order_accuracy_all_pairs']:.4f}")
    print(f"Report CSV:           {output_dir / 'report.csv'}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
