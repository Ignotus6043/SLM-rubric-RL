#!/usr/bin/env python3
"""Train a frozen-HF representation probe for Bench rubric judging.

This mirrors the RaR-Science probe interface: freeze the decoder LM, extract a
hidden state for each answer/criterion pair, train a tiny binary classifier on a
question-level train/dev split, and write Bench-compatible report/prediction
CSVs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROBE_PROMPT_TEMPLATE = """
You are an impartial rubric grader.

Instruction:
{question}

Candidate Response:
{response}

Rubric Criterion:
{criterion}

Question:
Does the candidate response satisfy this criterion?

Answer:
""".strip()


@dataclass
class BenchSample:
    sample_id: str
    question_id: str
    answer_id: int
    question: str
    response: str
    rules: list[dict[str, Any]]
    rule_weights: list[int]
    gold_verdicts: list[int]
    gold_total_score: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--train-questions", type=int, default=350)
    parser.add_argument("--dev-questions", type=int, default=50)
    parser.add_argument("--eval-split", choices=("all", "heldout"), default="all")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--rubric-format", choices=("canonical", "full"), default="canonical")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-2)
    parser.add_argument("--probe-classifier", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe-dropout", type=float, default=0.0)
    parser.add_argument(
        "--threshold-metric",
        choices=("macro_f1", "balanced_accuracy"),
        default="macro_f1",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return safe_div(num, den_x * den_y)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson_corr(average_ranks(xs), average_ranks(ys))


def verdicts_to_score(verdicts: list[int], rule_weights: list[int]) -> int:
    return int(sum(v * w for v, w in zip(verdicts, rule_weights)))


def vector_to_yn(vec: list[int] | None) -> str:
    if vec is None:
        return ""
    return "".join("Y" if int(v) == 1 else "N" for v in vec)


def flatten_samples(data: list[dict[str, Any]]) -> list[BenchSample]:
    samples = []
    for question_row in data:
        question_id = str(question_row.get("question_id", ""))
        question = str(question_row.get("question", ""))
        refined = question_row.get("refined_rubric", []) or []
        answers = question_row.get("answers", []) or []
        rule_ids = [str(rule.get("rule_id", i + 1)) for i, rule in enumerate(refined)]
        rule_weights = [3 if rule.get("type") == "hard" else 1 for rule in refined]
        if not refined:
            continue
        for answer in answers:
            answer_id = int(answer.get("answer_id", -1))
            verdict_map = {}
            for item in answer.get("grade", []) or []:
                rid = str(item.get("rule_id", ""))
                verdict = str(item.get("verdict", "")).strip().lower()
                verdict_map[rid] = 1 if verdict == "yes" else 0
            gold_verdicts = [verdict_map.get(rid, 0) for rid in rule_ids]
            computed_total = verdicts_to_score(gold_verdicts, rule_weights)
            samples.append(
                BenchSample(
                    sample_id=f"{question_id}_{answer_id}",
                    question_id=question_id,
                    answer_id=answer_id,
                    question=question,
                    response=str(answer.get("answer_text", "")),
                    rules=refined,
                    rule_weights=rule_weights,
                    gold_verdicts=gold_verdicts,
                    gold_total_score=int(answer.get("total_score", computed_total)),
                )
            )
    return samples


def split_question_ids(
    samples: list[BenchSample],
    train_questions: int,
    dev_questions: int,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    qids = sorted({sample.question_id for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(qids)
    if train_questions + dev_questions >= len(qids):
        raise ValueError(
            f"train_questions + dev_questions must be < number of questions. "
            f"Got {train_questions}+{dev_questions} for {len(qids)}."
        )
    train_ids = set(qids[:train_questions])
    dev_ids = set(qids[train_questions : train_questions + dev_questions])
    heldout_ids = set(qids[train_questions + dev_questions :])
    return train_ids, dev_ids, heldout_ids


def format_criterion(rule: dict[str, Any], rubric_format: str) -> str:
    rule_type = "hard" if rule.get("type") == "hard" else "soft"
    rubric = " ".join(str(rule.get("rubric", "")).split())
    if rubric_format == "full":
        return f"{rubric} [{rule_type.title()} Rule]"
    return f"The response satisfies this {rule_type} rule: {rubric}"


def build_prompt(question: str, response: str, criterion: str) -> str:
    return PROBE_PROMPT_TEMPLATE.format(question=question, response=response, criterion=criterion)


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


def load_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    cache_dir = os.environ.get("HF_HOME")
    token = os.environ.get("HF_TOKEN")
    kwargs = {"trust_remote_code": trust_remote_code}
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


def parse_layers(layer_spec: str, num_layers: int) -> list[int]:
    if layer_spec == "last":
        return [num_layers]
    if layer_spec == "all":
        return list(range(1, num_layers + 1))
    if layer_spec == "auto":
        candidates = [max(1, num_layers // 4), max(1, num_layers // 2), max(1, (3 * num_layers) // 4), num_layers]
    else:
        candidates = []
        for chunk in layer_spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            value = int(chunk)
            if value < 0:
                value = num_layers + 1 + value
            candidates.append(value)
    layers = []
    for layer in candidates:
        if layer < 1 or layer > num_layers:
            raise ValueError(f"Layer {layer} is out of range for model with {num_layers} layers.")
        if layer not in layers:
            layers.append(layer)
    return layers


def model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def extract_features(
    items: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    layers: list[int],
    batch_size: int,
    max_prompt_length: int,
) -> dict[int, torch.Tensor]:
    input_device = model_input_device(model)
    feature_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        attention_mask = encoded["attention_mask"]
        positions = attention_mask.sum(dim=1) - 1
        for layer in layers:
            hidden = hidden_states[layer]
            row_idx = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[row_idx, positions.to(hidden.device)].detach().float().cpu()
            feature_chunks[layer].append(selected)
        del outputs, hidden_states, encoded
        if torch.cuda.is_available() and start % max(batch_size * 20, 1) == 0:
            torch.cuda.empty_cache()
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in feature_chunks.items()}


def labels_tensor(items: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([float(row["label"]) for row in items], dtype=torch.float32)


def binary_metrics(y_true: torch.Tensor, probs: torch.Tensor, threshold: float) -> dict[str, float]:
    y = y_true.int()
    pred = (probs >= threshold).int()
    tp = int(((y == 1) & (pred == 1)).sum().item())
    fp = int(((y == 0) & (pred == 1)).sum().item())
    fn = int(((y == 1) & (pred == 0)).sum().item())
    tn = int(((y == 0) & (pred == 0)).sum().item())

    def div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    yes_precision = div(tp, tp + fp)
    yes_recall = div(tp, tp + fn)
    yes_f1 = div(2 * yes_precision * yes_recall, yes_precision + yes_recall)
    no_precision = div(tn, tn + fn)
    no_recall = div(tn, tn + fp)
    no_f1 = div(2 * no_precision * no_recall, no_precision + no_recall)
    return {
        "threshold": float(threshold),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
        "macro_f1": (yes_f1 + no_f1) / 2.0,
        "balanced_accuracy": (yes_recall + no_recall) / 2.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def select_threshold(y_true: torch.Tensor, probs: torch.Tensor, metric: str) -> tuple[float, dict[str, float]]:
    thresholds = {0.5}
    thresholds.update(i / 100.0 for i in range(5, 96))
    thresholds.update(float(value) for value in probs.tolist())
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, probs, best_threshold)
    for threshold in sorted(thresholds):
        metrics = binary_metrics(y_true, probs, threshold)
        if (
            metrics[metric],
            metrics["balanced_accuracy"],
            metrics["accuracy"],
        ) > (
            best_metrics[metric],
            best_metrics["balanced_accuracy"],
            best_metrics["accuracy"],
        ):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def standardize(
    train_features: torch.Tensor,
    dev_features: torch.Tensor,
    eval_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mean_vec = train_features.mean(dim=0, keepdim=True)
    std_vec = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (
        (train_features - mean_vec) / std_vec,
        (dev_features - mean_vec) / std_vec,
        (eval_features - mean_vec) / std_vec,
        {"mean": mean_vec, "std": std_vec},
    )


def build_probe_classifier(input_dim: int, classifier: str, hidden_dim: int, dropout: float) -> nn.Module:
    if classifier == "linear":
        return nn.Linear(input_dim, 1)
    if hidden_dim <= 0:
        raise ValueError("--probe-hidden-dim must be positive for mlp probes.")
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_probe_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    classifier: str,
    hidden_dim: int,
    dropout: float,
) -> tuple[nn.Module, dict[str, float]]:
    torch.manual_seed(seed)
    model = build_probe_classifier(train_features.shape[1], classifier, hidden_dim, dropout)
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives > 0 else torch.tensor([1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_dev_loss = float("inf")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features).squeeze(-1)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_logits = model(dev_features).squeeze(-1)
            dev_loss = F.binary_cross_entropy_with_logits(dev_logits, dev_labels).item()
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_dev_loss": best_dev_loss,
        "pos_weight": float(pos_weight.item()),
        "probe_trainable_params": count_trainable_params(model),
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_samples = len(rows)
    parsed_rows = [row for row in rows if row.get("parse_ok")]
    n_parsed = len(parsed_rows)
    total_rules_all = sum(row["num_rules"] for row in rows)
    total_weight_sum_all = sum(row["weight_sum"] for row in rows)
    total_rules = sum(row["num_rules"] for row in parsed_rows)
    matched_rules = sum(row["rule_matches"] for row in parsed_rows)
    weighted_matches = sum(row["weighted_rule_matches"] for row in parsed_rows)
    weighted_totals = sum(row["weight_sum"] for row in parsed_rows)
    vector_exact = sum(1 for row in parsed_rows if row["vector_exact"])
    score_exact = sum(1 for row in parsed_rows if row["score_exact"])
    score_abs_err = sum(abs(row["pred_total_score"] - row["gold_total_score"]) for row in parsed_rows)
    score_sq_err = sum((row["pred_total_score"] - row["gold_total_score"]) ** 2 for row in parsed_rows)
    score_bias = sum((row["pred_total_score"] - row["gold_total_score"]) for row in parsed_rows)
    score_within_1 = sum(1 for row in parsed_rows if abs(row["pred_total_score"] - row["gold_total_score"]) <= 1)
    score_within_2 = sum(1 for row in parsed_rows if abs(row["pred_total_score"] - row["gold_total_score"]) <= 2)

    tp = fp = fn = tn = 0
    for row in parsed_rows:
        for gt, pr in zip(row["gold_verdicts"], row["pred_verdicts"]):
            if gt == 1 and pr == 1:
                tp += 1
            elif gt == 0 and pr == 1:
                fp += 1
            elif gt == 1 and pr == 0:
                fn += 1
            else:
                tn += 1
    precision_yes = safe_div(tp, tp + fp)
    recall_yes = safe_div(tp, tp + fn)
    f1_yes = safe_div(2 * precision_yes * recall_yes, precision_yes + recall_yes)
    precision_no = safe_div(tn, tn + fn)
    recall_no = safe_div(tn, tn + fp)
    f1_no = safe_div(2 * precision_no * recall_no, precision_no + recall_no)
    macro_f1 = (f1_yes + f1_no) / 2.0
    balanced_accuracy = (recall_yes + recall_no) / 2.0
    mcc = safe_div((tp * tn) - (fp * fn), math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        by_question[row["question_id"]].append(row)
    rows_by_question_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_question_all[row["question_id"]].append(row)

    pair_correct = 0
    pair_total = 0
    pair_total_all = 0
    top1_correct = 0
    bottom1_correct = 0
    q_count = 0
    questions_total_with_multiple_answers = 0
    questions_fully_parseable = 0
    for qid, q_rows_all in rows_by_question_all.items():
        if len(q_rows_all) < 2:
            continue
        questions_total_with_multiple_answers += 1
        pair_total_all += math.comb(len(q_rows_all), 2)
        if all(row.get("parse_ok") for row in q_rows_all):
            questions_fully_parseable += 1
        q_rows = by_question.get(qid, [])
        if len(q_rows) < 2:
            continue
        q_count += 1
        for left, right in combinations(q_rows, 2):
            gold_diff = left["gold_total_score"] - right["gold_total_score"]
            pred_diff = left["pred_total_score"] - right["pred_total_score"]
            pair_total += 1
            if (
                (gold_diff == 0 and pred_diff == 0)
                or (gold_diff > 0 and pred_diff > 0)
                or (gold_diff < 0 and pred_diff < 0)
            ):
                pair_correct += 1
        gold_max = max(row["gold_total_score"] for row in q_rows)
        pred_max = max(row["pred_total_score"] for row in q_rows)
        if {row["answer_id"] for row in q_rows if row["gold_total_score"] == gold_max} & {
            row["answer_id"] for row in q_rows if row["pred_total_score"] == pred_max
        }:
            top1_correct += 1
        gold_min = min(row["gold_total_score"] for row in q_rows)
        pred_min = min(row["pred_total_score"] for row in q_rows)
        if {row["answer_id"] for row in q_rows if row["gold_total_score"] == gold_min} & {
            row["answer_id"] for row in q_rows if row["pred_total_score"] == pred_min
        }:
            bottom1_correct += 1

    gold_scores = [float(row["gold_total_score"]) for row in parsed_rows]
    pred_scores = [float(row["pred_total_score"]) for row in parsed_rows]
    return {
        "samples_total": n_samples,
        "samples_parse_ok": n_parsed,
        "parse_success_rate": safe_div(n_parsed, n_samples),
        "samples_parse_failed": n_samples - n_parsed,
        "rule_exact_match_rate": safe_div(vector_exact, n_parsed),
        "rule_exact_match_rate_all_samples": safe_div(vector_exact, n_samples),
        "rule_hamming_accuracy": safe_div(matched_rules, total_rules),
        "rule_hamming_accuracy_all_samples": safe_div(matched_rules, total_rules_all),
        "rule_weighted_accuracy": safe_div(weighted_matches, weighted_totals),
        "rule_weighted_accuracy_all_samples": safe_div(weighted_matches, total_weight_sum_all),
        "yes_precision": precision_yes,
        "yes_recall": recall_yes,
        "yes_f1": f1_yes,
        "no_precision": precision_no,
        "no_recall": recall_no,
        "no_f1": f1_no,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "jaccard_yes": safe_div(tp, tp + fp + fn),
        "mcc": mcc,
        "score_exact_match_rate": safe_div(score_exact, n_parsed),
        "score_exact_match_rate_all_samples": safe_div(score_exact, n_samples),
        "score_mae": safe_div(score_abs_err, n_parsed),
        "score_rmse": math.sqrt(safe_div(score_sq_err, n_parsed)) if n_parsed else 0.0,
        "score_mean_error_bias": safe_div(score_bias, n_parsed),
        "score_within_1_rate": safe_div(score_within_1, n_parsed),
        "score_within_1_rate_all_samples": safe_div(score_within_1, n_samples),
        "score_within_2_rate": safe_div(score_within_2, n_parsed),
        "score_within_2_rate_all_samples": safe_div(score_within_2, n_samples),
        "score_pearson": pearson_corr(gold_scores, pred_scores),
        "score_spearman": spearman_corr(gold_scores, pred_scores),
        "pairwise_order_accuracy": safe_div(pair_correct, pair_total),
        "pairwise_order_accuracy_all_pairs": safe_div(pair_correct, pair_total_all),
        "pairwise_parseable_coverage": safe_div(pair_total, pair_total_all),
        "top1_set_accuracy": safe_div(top1_correct, q_count),
        "top1_set_accuracy_all_questions": safe_div(top1_correct, questions_total_with_multiple_answers),
        "bottom1_set_accuracy": safe_div(bottom1_correct, q_count),
        "bottom1_set_accuracy_all_questions": safe_div(bottom1_correct, questions_total_with_multiple_answers),
        "questions_with_parseable_answers": q_count,
        "questions_total_with_multiple_answers": questions_total_with_multiple_answers,
        "question_parseable_coverage": safe_div(q_count, questions_total_with_multiple_answers),
        "questions_fully_parseable": questions_fully_parseable,
        "question_full_parse_rate": safe_div(questions_fully_parseable, questions_total_with_multiple_answers),
    }


def build_prediction_rows(
    eval_samples: list[BenchSample],
    eval_items: list[dict[str, Any]],
    probs: torch.Tensor,
    threshold: float,
    label: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item, prob in zip(eval_items, probs.tolist()):
        grouped[item["sample_id"]].append(
            {
                **item,
                "yes_prob": float(prob),
                "verdict": 1 if float(prob) >= threshold else 0,
            }
        )

    rows = []
    for sample in eval_samples:
        rule_rows = sorted(grouped[sample.sample_id], key=lambda row: int(row["rule_index"]))
        pred_verdicts = [int(row["verdict"]) for row in rule_rows]
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
                "raw_output": "hf_representation_probe",
                "first_line_output": "hf_representation_probe",
                "criterion_yes_probs": [float(row["yes_prob"]) for row in rule_rows],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started_at = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_artifact_path = output_dir / "probe_artifact.pt"
    bench_data = json.loads(Path(args.bench_json).read_text(encoding="utf-8"))
    samples = flatten_samples(bench_data)
    if not samples:
        raise SystemExit("ERROR: no Bench samples found.")

    train_ids, dev_ids, heldout_ids = split_question_ids(samples, args.train_questions, args.dev_questions, args.seed)
    eval_ids = {sample.question_id for sample in samples} if args.eval_split == "all" else heldout_ids

    train_items, _ = build_criterion_items(samples, train_ids, "train", args.rubric_format)
    dev_items, _ = build_criterion_items(samples, dev_ids, "dev", args.rubric_format)
    eval_items, eval_samples = build_criterion_items(samples, eval_ids, args.eval_split, args.rubric_format)
    if not train_items or not dev_items or not eval_items:
        raise SystemExit(
            f"ERROR: empty split. train_items={len(train_items)} dev_items={len(dev_items)} eval_items={len(eval_items)}"
        )

    tokenizer = load_tokenizer(args.judge_model, args.trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.judge_model, args.trust_remote_code)
    model.eval()
    layers = parse_layers(args.layers, int(getattr(model.config, "num_hidden_layers")))

    all_items = train_items + dev_items + eval_items
    train_end = len(train_items)
    dev_end = train_end + len(dev_items)
    print(f"[INFO] judge_model={args.judge_model}", flush=True)
    print(f"[INFO] layers={layers}", flush=True)
    print(
        f"[INFO] criteria train/dev/eval={len(train_items)}/{len(dev_items)}/{len(eval_items)} "
        f"eval_split={args.eval_split}",
        flush=True,
    )
    with torch.inference_mode():
        features_by_layer = extract_features(
            all_items,
            model,
            tokenizer,
            layers,
            args.batch_size,
            args.max_prompt_length,
        )
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    y_train = labels_tensor(train_items)
    y_dev = labels_tensor(dev_items)
    y_eval = labels_tensor(eval_items)
    best = None
    layer_summaries = []
    for layer in layers:
        features = features_by_layer[layer]
        x_train_raw = features[:train_end]
        x_dev_raw = features[train_end:dev_end]
        x_eval_raw = features[dev_end:]
        x_train, x_dev, x_eval, scaler = standardize(x_train_raw, x_dev_raw, x_eval_raw)
        probe, train_info = train_probe_classifier(
            x_train,
            y_train,
            x_dev,
            y_dev,
            args.probe_epochs,
            args.probe_lr,
            args.probe_weight_decay,
            args.seed + layer,
            args.probe_classifier,
            args.probe_hidden_dim,
            args.probe_dropout,
        )
        probe.eval()
        with torch.no_grad():
            dev_logits = probe(x_dev).squeeze(-1)
            eval_logits = probe(x_eval).squeeze(-1)
            dev_probs = torch.sigmoid(dev_logits)
            eval_probs = torch.sigmoid(eval_logits)
        threshold, dev_metrics = select_threshold(y_dev, dev_probs, args.threshold_metric)
        eval_metrics = binary_metrics(y_eval, eval_probs, threshold)
        row = {
            "layer": layer,
            "threshold": threshold,
            "dev": dev_metrics,
            "eval": eval_metrics,
            **train_info,
        }
        layer_summaries.append(row)
        rank_key = (
            dev_metrics[args.threshold_metric],
            dev_metrics["balanced_accuracy"],
            dev_metrics["accuracy"],
            -train_info["best_dev_loss"],
        )
        if best is None or rank_key > best["rank_key"]:
            best = {
                "rank_key": rank_key,
                "layer": layer,
                "threshold": threshold,
                "eval_probs": eval_probs,
                "probe": probe,
                "scaler": scaler,
                "summary": row,
            }
        print(
            f"[INFO] layer={layer} dev_{args.threshold_metric}={dev_metrics[args.threshold_metric]:.4f} "
            f"eval_{args.threshold_metric}={eval_metrics[args.threshold_metric]:.4f}",
            flush=True,
        )

    if best is None:
        raise RuntimeError("No probe layer was trained.")

    label = args.label or args.judge_model
    best_probe = best["probe"]
    torch.save(
        {
            "artifact_type": "bench_hf_representation_probe",
            "format_version": 1,
            "label": label,
            "judge_model": args.judge_model,
            "rubric_format": args.rubric_format,
            "probe_classifier": args.probe_classifier,
            "probe_hidden_dim": int(args.probe_hidden_dim),
            "probe_dropout": float(args.probe_dropout),
            "probe_trainable_params": int(best["summary"]["probe_trainable_params"]),
            "best_layer": int(best["layer"]),
            "best_threshold": float(best["threshold"]),
            "threshold_metric": args.threshold_metric,
            "layers_requested": args.layers,
            "feature_dim": int(best["scaler"]["mean"].shape[-1]),
            "state_dict": {key: value.detach().cpu() for key, value in best_probe.state_dict().items()},
            "scaler": {
                "mean": best["scaler"]["mean"].detach().cpu(),
                "std": best["scaler"]["std"].detach().cpu(),
            },
            "train_info": {
                "bench_json": args.bench_json,
                "train_questions": args.train_questions,
                "dev_questions": args.dev_questions,
                "eval_split": args.eval_split,
                "seed": args.seed,
                "probe_epochs": args.probe_epochs,
                "probe_lr": args.probe_lr,
                "probe_weight_decay": args.probe_weight_decay,
            },
            "best_dev_metrics": best["summary"]["dev"],
            "best_eval_criterion_metrics": best["summary"]["eval"],
        },
        probe_artifact_path,
    )
    pred_rows = build_prediction_rows(eval_samples, eval_items, best["eval_probs"], float(best["threshold"]), label)
    metrics = compute_metrics(pred_rows)
    elapsed_seconds = time.perf_counter() - started_at
    metric_row = {
        "model": label,
        **metrics,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": safe_div(elapsed_seconds, len(eval_samples)),
        "judge_model": args.judge_model,
        "judge_backend": "hf_representation_probe",
        "train_questions": args.train_questions,
        "dev_questions": args.dev_questions,
        "eval_split": args.eval_split,
        "rubric_format": args.rubric_format,
        "layers_requested": args.layers,
        "best_layer": int(best["layer"]),
        "best_threshold": float(best["threshold"]),
        "probe_classifier": args.probe_classifier,
        "probe_hidden_dim": args.probe_hidden_dim,
        "probe_dropout": args.probe_dropout,
        "probe_trainable_params": int(best["summary"]["probe_trainable_params"]),
        "probe_artifact": str(probe_artifact_path),
    }

    write_csv(output_dir / "report.csv", [metric_row])
    write_csv(output_dir / "overall_metrics.csv", [metric_row])
    write_csv(output_dir / "per_sample_predictions.csv", pred_rows)
    (output_dir / "probe_summary.json").write_text(
        json.dumps(
            {
                "label": label,
                "judge_model": args.judge_model,
                "bench_json": args.bench_json,
                "train_questions": args.train_questions,
                "dev_questions": args.dev_questions,
                "eval_split": args.eval_split,
                "rubric_format": args.rubric_format,
                "probe_classifier": args.probe_classifier,
                "probe_hidden_dim": args.probe_hidden_dim,
                "probe_dropout": args.probe_dropout,
                "best_layer": int(best["layer"]),
                "best_threshold": float(best["threshold"]),
                "probe_trainable_params": int(best["summary"]["probe_trainable_params"]),
                "probe_artifact": str(probe_artifact_path),
                "layers": layer_summaries,
                "metrics": metric_row,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=======================================")
    print(f"Label:                {label}")
    print(f"Bench samples scored: {len(eval_samples)}")
    print(f"Parse success:        {metrics['parse_success_rate']:.4f}")
    print(f"Weighted acc:         {metrics['rule_weighted_accuracy_all_samples']:.4f}")
    print(f"Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"Pairwise acc:         {metrics['pairwise_order_accuracy_all_pairs']:.4f}")
    print(f"Best layer:           {int(best['layer'])}")
    print(f"Probe params:         {int(best['summary']['probe_trainable_params'])}")
    print(f"Report CSV:           {output_dir / 'report.csv'}")
    print(f"Probe artifact:       {probe_artifact_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
