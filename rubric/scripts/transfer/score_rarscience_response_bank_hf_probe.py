#!/usr/bin/env python3
"""Train a frozen-HF representation probe for RaR-Science criterion judging.

This tests whether a small decoder LM contains rubric-judging signal in its
hidden states even when generated verdicts are weak. The model is frozen. For
each criterion, we extract a hidden state from a prompt ending at "Answer:",
train a tiny linear binary classifier on GPT-labeled train/dev split items, and
emit a heldout scored bank compatible with evaluate_rarscience_judge_alignment.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-scored-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--candidate-id", default="reference_answer")
    parser.add_argument(
        "--candidate-ids",
        nargs="*",
        default=None,
        help=(
            "Optional candidate_id allowlist. When provided, the probe is trained "
            "and evaluated on all listed candidates jointly. Overrides --candidate-id."
        ),
    )
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--train-size", type=int, default=350)
    parser.add_argument("--train-question-ids-file", default="")
    parser.add_argument("--dev-question-ids-file", default="")
    parser.add_argument("--heldout-question-ids-file", default="")
    parser.add_argument(
        "--eval-split",
        choices=("heldout", "all"),
        default="heldout",
        help="Question set to score after fitting on train and selecting threshold/layer on dev.",
    )
    parser.add_argument(
        "--eval-question-ids-file",
        default="",
        help="Optional explicit evaluation question ids. Overrides --eval-split.",
    )
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--rubric-format",
        choices=("full", "compact", "description_only", "title_only", "canonical"),
        default="canonical",
    )
    parser.add_argument("--max-rubric-rule-chars", type=int, default=0)
    parser.add_argument(
        "--layers",
        default="auto",
        help="Layer sweep: auto, last, all, or comma-separated layer ids. Positive ids use hidden_states[id]; -1 means final layer.",
    )
    parser.add_argument(
        "--pooling",
        choices=("last", "mean", "last4mean"),
        default="last",
        help="How to pool token hidden states before fitting the probe.",
    )
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--probe-classifier",
        choices=("linear", "mlp"),
        default="linear",
        help="Classifier head trained on frozen LM representations.",
    )
    parser.add_argument(
        "--probe-hidden-dim",
        type=int,
        default=64,
        help="Hidden width for --probe-classifier mlp.",
    )
    parser.add_argument("--probe-dropout", type=float, default=0.0)
    parser.add_argument(
        "--score-mode",
        choices=("probability", "binary"),
        default="probability",
        help="Use probe probabilities or thresholded verdicts for aggregate candidate score.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=("macro_f1", "balanced_accuracy"),
        default="macro_f1",
        help="Dev-set metric used to choose the Yes threshold for each layer.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_question_ids_file(path: str) -> set[str]:
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
            for key in ("train_question_ids", "dev_question_ids", "heldout_question_ids", "question_ids"):
                values = data.get(key)
                if isinstance(values, list):
                    return {str(item) for item in values}
        raise ValueError(f"Could not find question ids in JSON file: {path}")
    return {line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")}


def load_split_ids(args: argparse.Namespace) -> tuple[set[str], set[str], set[str]]:
    train_ids = load_question_ids_file(args.train_question_ids_file)
    dev_ids = load_question_ids_file(args.dev_question_ids_file)
    heldout_ids = load_question_ids_file(args.heldout_question_ids_file)

    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        split = (manifest.get("splits") or {}).get(str(args.train_size))
        if split is None:
            raise ValueError(f"Train size {args.train_size} not found in split manifest: {args.split_manifest}")
        if not train_ids:
            train_ids = {str(item) for item in split.get("train_question_ids", [])}
        if not dev_ids:
            dev_ids = {str(item) for item in split.get("dev_question_ids", [])}
        if not heldout_ids:
            heldout_ids = {str(item) for item in split.get("heldout_question_ids", [])}

    require_heldout = args.eval_split == "heldout" and not args.eval_question_ids_file
    if not train_ids or not dev_ids or (require_heldout and not heldout_ids):
        raise ValueError(
            "Missing split ids. Provide --split-manifest or explicit train/dev/heldout question id files."
        )
    overlap = (train_ids & dev_ids) | (train_ids & heldout_ids) | (dev_ids & heldout_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Split question ids overlap; first overlaps: {preview}")
    return train_ids, dev_ids, heldout_ids


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
    return PROBE_PROMPT_TEMPLATE.format(question=question, response=response, criterion=criterion)


def candidate_by_id(record: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    for candidate in record.get("candidates", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def criterion_scores(candidate: dict[str, Any]) -> list[int]:
    reward = candidate.get("reward_result") or {}
    scores = reward.get("criterion_scores") if isinstance(reward, dict) else None
    if not isinstance(scores, list):
        return []
    return [1 if float(item) >= 0.5 else 0 for item in scores]


def candidate_parse_ok(candidate: dict[str, Any], rubric_len: int, require_labels: bool = True) -> bool:
    if rubric_len <= 0:
        return False
    if not require_labels:
        return True
    reward = candidate.get("reward_result") or {}
    failures = int(candidate.get("num_parse_failures", reward.get("num_parse_failures", 0)))
    return failures == 0 and len(criterion_scores(candidate)) == rubric_len and rubric_len > 0


def build_criterion_items(
    records: list[dict[str, Any]],
    question_ids: set[str],
    candidate_ids: list[str],
    split: str,
    rubric_format: str,
    max_rule_chars: int,
    require_labels: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = []
    selected_records = []
    skipped = 0
    for record in records:
        qid = str(record.get("question_id", ""))
        if qid not in question_ids:
            continue
        rubric = record.get("rubric") or []
        record_selected = False
        for candidate_id in candidate_ids:
            candidate = candidate_by_id(record, candidate_id)
            if candidate is None or not candidate_parse_ok(candidate, len(rubric), require_labels=require_labels):
                skipped += 1
                continue
            labels = criterion_scores(candidate) if require_labels else [0] * len(rubric)
            if not record_selected:
                selected_records.append(record)
                record_selected = True
            for rule_index, (rule, label) in enumerate(zip(rubric, labels)):
                criterion = format_criterion(rule, rubric_format, max_rule_chars)
                items.append(
                    {
                        "split": split,
                        "question_id": qid,
                        "candidate_id": candidate_id,
                        "rule_index": rule_index,
                        "prompt": build_prompt(record.get("question", ""), candidate.get("response", ""), criterion),
                        "criterion": criterion,
                        "category": category_for_rule(rule),
                        "label": int(label),
                        "weight": float(rule.get("weight", 1.0) if rule.get("weight", 1.0) is not None else 1.0),
                    }
                )
    if skipped:
        print(f"[WARN] skipped {skipped} {split} records due to missing candidate/labels.", flush=True)
    return items, selected_records


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
    pooling: str = "last",
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
            pos = positions.to(hidden.device)
            if pooling == "last":
                row_idx = torch.arange(hidden.shape[0], device=hidden.device)
                selected = hidden[row_idx, pos]
            elif pooling == "mean":
                mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
                selected = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            elif pooling == "last4mean":
                selected_rows = []
                for row_idx, end_pos in enumerate(pos.tolist()):
                    start_pos = max(0, int(end_pos) - 3)
                    selected_rows.append(hidden[row_idx, start_pos : int(end_pos) + 1].mean(dim=0))
                selected = torch.stack(selected_rows, dim=0)
            else:
                raise ValueError(f"Unsupported pooling mode: {pooling}")
            selected = selected.detach().float().cpu()
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

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    yes_precision = safe_div(tp, tp + fp)
    yes_recall = safe_div(tp, tp + fn)
    yes_f1 = safe_div(2 * yes_precision * yes_recall, yes_precision + yes_recall)
    no_precision = safe_div(tn, tn + fn)
    no_recall = safe_div(tn, tn + fp)
    no_f1 = safe_div(2 * no_precision * no_recall, no_precision + no_recall)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "threshold": float(threshold),
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
        "macro_f1": (yes_f1 + no_f1) / 2.0,
        "balanced_accuracy": (yes_recall + no_recall) / 2.0,
        "yes_f1": yes_f1,
        "no_f1": no_f1,
        "yes_precision": yes_precision,
        "yes_recall": yes_recall,
        "no_precision": no_precision,
        "no_recall": no_recall,
        "mcc": safe_div(tp * tn - fp * fn, den),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def select_threshold(y_true: torch.Tensor, probs: torch.Tensor, metric: str) -> tuple[float, dict[str, float]]:
    thresholds = {0.5}
    thresholds.update(i / 100.0 for i in range(5, 96))
    for value in probs.tolist():
        thresholds.add(float(value))
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
    heldout_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mean_vec = train_features.mean(dim=0, keepdim=True)
    std_vec = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (
        (train_features - mean_vec) / std_vec,
        (dev_features - mean_vec) / std_vec,
        (heldout_features - mean_vec) / std_vec,
        {"mean": mean_vec, "std": std_vec},
    )


def build_probe_classifier(
    input_dim: int,
    classifier: str,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if classifier == "linear":
        return nn.Linear(input_dim, 1)
    if classifier == "mlp":
        if hidden_dim <= 0:
            raise ValueError("--probe-hidden-dim must be positive for mlp probes.")
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    raise ValueError(f"Unsupported probe classifier: {classifier}")


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
        "probe_classifier": classifier,
        "probe_hidden_dim": int(hidden_dim) if classifier == "mlp" else 0,
        "probe_dropout": float(dropout),
        "probe_trainable_params": count_trainable_params(model),
    }


def aggregate_score(rubric: list[dict[str, Any]], values: list[float]) -> tuple[float, list[float]]:
    weights = [abs(float(item.get("weight", 1.0) if item.get("weight", 1.0) is not None else 1.0)) for item in rubric]
    total_positive_weight = sum(weight for weight in weights if weight > 0)
    if total_positive_weight <= 0:
        return 0.0, weights
    return sum(value * weight for value, weight in zip(values, weights)) / total_positive_weight, weights


def build_scored_heldout_bank(
    heldout_records: list[dict[str, Any]],
    heldout_items: list[dict[str, Any]],
    probs: torch.Tensor,
    logits: torch.Tensor,
    threshold: float,
    candidate_ids: list[str],
    score_mode: str,
    label: str,
    judge_model: str,
    rubric_format: str,
    layer: int,
    probe_classifier: str,
    probe_hidden_dim: int,
    probe_dropout: float,
    probe_trainable_params: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item, prob, logit in zip(heldout_items, probs.tolist(), logits.tolist()):
        grouped[(item["question_id"], item["candidate_id"])].append(
            {
                **item,
                "yes_prob": float(prob),
                "probe_logit": float(logit),
                "verdict": 1.0 if float(prob) >= threshold else 0.0,
            }
        )

    output_records = []
    for record in heldout_records:
        qid = str(record.get("question_id", ""))
        rubric = record.get("rubric", [])
        scored_candidates = []
        for candidate_id in candidate_ids:
            source_candidate = candidate_by_id(record, candidate_id)
            if source_candidate is None:
                continue
            candidate_rule_results = sorted(grouped[(qid, candidate_id)], key=lambda row: int(row["rule_index"]))
            if len(candidate_rule_results) != len(rubric):
                raise ValueError(
                    f"Internal error: expected {len(rubric)} heldout rule predictions for {qid}/{candidate_id}, "
                    f"found {len(candidate_rule_results)}"
                )
            verdicts = [float(row["verdict"]) for row in candidate_rule_results]
            yes_probs = [float(row["yes_prob"]) for row in candidate_rule_results]
            logits_list = [float(row["probe_logit"]) for row in candidate_rule_results]
            score_values = yes_probs if score_mode == "probability" else verdicts
            score, weights = aggregate_score(rubric, score_values)
            binary_score, _ = aggregate_score(rubric, verdicts)
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
                "parse_source": "hf_representation_probe",
                "score_mode": score_mode,
                "probe_threshold": float(threshold),
                "criterion_yes_probs": yes_probs,
                "criterion_probe_logits": logits_list,
                "criterion_text": [row["criterion"] for row in candidate_rule_results],
                "criterion_categories": [row["category"] for row in candidate_rule_results],
                "judge_model": judge_model,
                "judge_label": label,
                "rubric_format": rubric_format,
                "probe_layer": layer,
                "probe_classifier": probe_classifier,
                "probe_hidden_dim": int(probe_hidden_dim),
                "probe_dropout": float(probe_dropout),
                "probe_trainable_params": int(probe_trainable_params),
            }
            scored_candidates.append(
                {
                    **source_candidate,
                    "score": score,
                    "num_parse_failures": 0,
                    "reward_result": reward_result,
                }
            )
        if scored_candidates:
            output_records.append({**record, "candidates": scored_candidates})
    return output_records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    candidate_ids = list(dict.fromkeys(args.candidate_ids or [args.candidate_id]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_bank.jsonl"
    summary_path = output_dir / "summary.json"
    probe_summary_path = output_dir / "probe_summary.json"
    probe_artifact_path = output_dir / "probe_artifact.pt"

    records = load_jsonl(args.gpt_scored_bank)
    train_ids, dev_ids, heldout_ids = load_split_ids(args)
    train_items, _ = build_criterion_items(
        records, train_ids, candidate_ids, "train", args.rubric_format, args.max_rubric_rule_chars
    )
    dev_items, _ = build_criterion_items(
        records, dev_ids, candidate_ids, "dev", args.rubric_format, args.max_rubric_rule_chars
    )
    if args.eval_question_ids_file:
        eval_ids = load_question_ids_file(args.eval_question_ids_file)
        eval_split_name = Path(args.eval_question_ids_file).stem
    elif args.eval_split == "all":
        eval_ids = {str(record.get("question_id", "")) for record in records if record.get("question_id", "")}
        eval_split_name = "all"
    else:
        eval_ids = heldout_ids
        eval_split_name = "heldout"

    eval_items, eval_records = build_criterion_items(
        records, eval_ids, candidate_ids, eval_split_name, args.rubric_format, args.max_rubric_rule_chars
    )
    if not train_items or not dev_items or not eval_items:
        raise SystemExit(
            f"ERROR: empty probe split. train_items={len(train_items)} dev_items={len(dev_items)} eval_items={len(eval_items)}"
        )

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
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layers = parse_layers(args.layers, num_layers)

    all_items = train_items + dev_items + eval_items
    train_end = len(train_items)
    dev_end = train_end + len(dev_items)

    started_at = time.perf_counter()
    print(f"[INFO] judge_model={args.judge_model}", flush=True)
    print(f"[INFO] layers={layers}", flush=True)
    print(f"[INFO] pooling={args.pooling}", flush=True)
    print(
        f"[INFO] criteria train/dev/eval={len(train_items)}/{len(dev_items)}/{len(eval_items)} "
        f"eval_split={eval_split_name} "
        f"batch_size={args.batch_size}",
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
            args.pooling,
        )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    y_train = labels_tensor(train_items)
    y_dev = labels_tensor(dev_items)
    y_eval = labels_tensor(eval_items)
    layer_summaries = []
    best = None

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
            train_logits = probe(x_train).squeeze(-1)
            dev_logits = probe(x_dev).squeeze(-1)
            eval_logits = probe(x_eval).squeeze(-1)
            train_probs = torch.sigmoid(train_logits)
            dev_probs = torch.sigmoid(dev_logits)
            eval_probs = torch.sigmoid(eval_logits)

        threshold, dev_metrics = select_threshold(y_dev, dev_probs, args.threshold_metric)
        train_metrics = binary_metrics(y_train, train_probs, threshold)
        eval_metrics = binary_metrics(y_eval, eval_probs, threshold)
        row = {
            "layer": layer,
            "threshold": threshold,
            "train": train_metrics,
            "dev": dev_metrics,
            "eval": eval_metrics,
            "heldout": eval_metrics,
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
                "probe": probe,
                "x_eval": x_eval,
                "eval_logits": eval_logits,
                "eval_probs": eval_probs,
                "scaler": scaler,
                "summary": row,
            }
        print(
            f"[INFO] layer={layer} dev_{args.threshold_metric}={dev_metrics[args.threshold_metric]:.4f} "
            f"dev_bal_acc={dev_metrics['balanced_accuracy']:.4f} "
            f"eval_{args.threshold_metric}={eval_metrics[args.threshold_metric]:.4f}",
            flush=True,
        )

    if best is None:
        raise RuntimeError("No probe layer was trained.")

    scored_records = build_scored_heldout_bank(
        heldout_records=eval_records,
        heldout_items=eval_items,
        probs=best["eval_probs"],
        logits=best["eval_logits"],
        threshold=float(best["threshold"]),
        candidate_ids=candidate_ids,
        score_mode=args.score_mode,
        label=args.label or args.judge_model,
        judge_model=args.judge_model,
        rubric_format=args.rubric_format,
        layer=int(best["layer"]),
        probe_classifier=args.probe_classifier,
        probe_hidden_dim=args.probe_hidden_dim,
        probe_dropout=args.probe_dropout,
        probe_trainable_params=int(best["summary"]["probe_trainable_params"]),
    )
    write_jsonl(scored_path, scored_records)

    best_probe = best["probe"]
    probe_artifact = {
        "artifact_type": "rarscience_hf_representation_probe",
        "format_version": 1,
        "label": args.label or args.judge_model,
        "judge_model": args.judge_model,
        "candidate_id": candidate_ids[0] if len(candidate_ids) == 1 else None,
        "candidate_ids": candidate_ids,
        "rubric_format": args.rubric_format,
        "max_rubric_rule_chars": args.max_rubric_rule_chars,
        "score_mode": args.score_mode,
        "probe_classifier": args.probe_classifier,
        "probe_hidden_dim": int(args.probe_hidden_dim),
        "probe_dropout": float(args.probe_dropout),
        "pooling": args.pooling,
        "probe_trainable_params": int(best["summary"]["probe_trainable_params"]),
        "best_layer": int(best["layer"]),
        "best_threshold": float(best["threshold"]),
        "threshold_metric": args.threshold_metric,
        "layers_requested": args.layers,
        "layers_evaluated": layers,
        "feature_dim": int(best["scaler"]["mean"].shape[-1]),
        "state_dict": {key: value.detach().cpu() for key, value in best_probe.state_dict().items()},
        "scaler": {
            "mean": best["scaler"]["mean"].detach().cpu(),
            "std": best["scaler"]["std"].detach().cpu(),
        },
        "train_info": {
            "split_manifest": args.split_manifest or None,
            "train_size": args.train_size,
            "num_train_questions": len(train_ids),
            "num_dev_questions": len(dev_ids),
            "num_heldout_questions": len(heldout_ids),
            "num_train_criteria": len(train_items),
            "num_dev_criteria": len(dev_items),
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "probe_weight_decay": args.probe_weight_decay,
            "seed": args.seed,
        },
        "best_dev_metrics": best["summary"]["dev"],
        "best_eval_criterion_metrics": best["summary"]["eval"],
    }
    torch.save(probe_artifact, probe_artifact_path)

    elapsed_seconds = time.perf_counter() - started_at
    num_candidates = sum(len(row.get("candidates", [])) for row in scored_records)
    total_criteria = len(eval_items)
    summary = {
        "label": args.label or args.judge_model,
        "judge_model": args.judge_model,
        "judge_backend": "hf_representation_probe",
        "gpt_scored_bank": args.gpt_scored_bank,
        "candidate_id": candidate_ids[0] if len(candidate_ids) == 1 else None,
        "candidate_ids": candidate_ids,
        "split_manifest": args.split_manifest or None,
        "train_size": args.train_size,
        "num_train_questions": len(train_ids),
        "num_dev_questions": len(dev_ids),
        "num_heldout_questions": len(heldout_ids),
        "eval_split": eval_split_name,
        "num_eval_questions_requested": len(eval_ids),
        "num_eval_questions_scored": len(eval_records),
        "num_train_criteria": len(train_items),
        "num_dev_criteria": len(dev_items),
        "num_heldout_criteria": len(eval_items),
        "num_eval_criteria": len(eval_items),
        "num_questions": len(scored_records),
        "num_candidates_scored": num_candidates,
        "num_criteria_scored": total_criteria,
        "batch_size": args.batch_size,
        "max_prompt_length": args.max_prompt_length,
        "rubric_format": args.rubric_format,
        "max_rubric_rule_chars": args.max_rubric_rule_chars,
        "score_mode": args.score_mode,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "probe_classifier": args.probe_classifier,
        "probe_hidden_dim": args.probe_hidden_dim,
        "probe_dropout": args.probe_dropout,
        "pooling": args.pooling,
        "probe_trainable_params": int(best["summary"]["probe_trainable_params"]),
        "probe_artifact": str(probe_artifact_path),
        "threshold_metric": args.threshold_metric,
        "layers_requested": args.layers,
        "layers_evaluated": layers,
        "best_layer": int(best["layer"]),
        "best_threshold": float(best["threshold"]),
        "best_dev_metrics": best["summary"]["dev"],
        "best_eval_criterion_metrics": best["summary"]["eval"],
        "best_heldout_criterion_metrics": best["summary"]["eval"],
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
                "num_examples": sum(
                    1
                    for row in scored_records
                    for candidate in row.get("candidates", [])
                    if candidate.get("candidate_id") == candidate_id
                ),
                "mean_score": mean(
                    [
                        float(candidate["score"])
                        for row in scored_records
                        for candidate in row.get("candidates", [])
                        if candidate.get("candidate_id") == candidate_id
                    ]
                )
                if any(
                    candidate.get("candidate_id") == candidate_id
                    for row in scored_records
                    for candidate in row.get("candidates", [])
                )
                else 0.0,
            }
            for candidate_id in candidate_ids
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    probe_summary_path.write_text(
        json.dumps(
            {
                "label": args.label or args.judge_model,
                "judge_model": args.judge_model,
                "best_layer": int(best["layer"]),
                "best_threshold": float(best["threshold"]),
                "probe_classifier": args.probe_classifier,
                "probe_hidden_dim": args.probe_hidden_dim,
                "probe_dropout": args.probe_dropout,
                "pooling": args.pooling,
                "probe_trainable_params": int(best["summary"]["probe_trainable_params"]),
                "probe_artifact": str(probe_artifact_path),
                "eval_split": eval_split_name,
                "layers": layer_summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=======================================")
    print(f"Judge label:           {summary['label']}")
    print(f"Judge backend:         {summary['judge_backend']}")
    print(f"Probe classifier:     {summary['probe_classifier']}")
    print(f"Probe params:         {summary['probe_trainable_params']}")
    print(f"Eval split:           {summary['eval_split']}")
    print(f"Best layer:            {summary['best_layer']}")
    print(f"Best threshold:        {summary['best_threshold']:.4f}")
    print(f"Candidates scored:     {summary['num_candidates_scored']}")
    print(f"Criteria scored:       {summary['num_criteria_scored']}")
    print(f"Parse success:         {summary['parse_success_rate']:.4f}")
    print(f"Elapsed seconds:       {elapsed_seconds:.2f}")
    print(f"Scored bank:           {scored_path}")
    print(f"Summary:               {summary_path}")
    print(f"Probe summary:         {probe_summary_path}")
    print(f"Probe artifact:        {probe_artifact_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
