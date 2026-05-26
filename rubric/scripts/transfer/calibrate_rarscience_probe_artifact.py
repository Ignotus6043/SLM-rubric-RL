#!/usr/bin/env python3
"""Fit Platt/temperature calibration for a saved RaR-Science probe artifact.

The saved probe is left frozen. This script reuses the artifact's selected
layer and linear head, computes logits on the GPT-labeled dev split, fits only
calibration scalars, and writes a copy of the artifact with calibration
metadata. The reward server applies that metadata as:

    p_cal = sigmoid(a * logit + b)

for probability-mode rewards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HELPER_PATH = REPO_ROOT / "scripts" / "transfer" / "score_rarscience_response_bank_hf_probe.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("rarscience_probe_helper", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load helper from {_HELPER_PATH}")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPER
_HELPER_SPEC.loader.exec_module(_HELPER)

build_criterion_items = _HELPER.build_criterion_items
build_probe_classifier = _HELPER.build_probe_classifier
extract_features = _HELPER.extract_features
labels_tensor = _HELPER.labels_tensor
load_jsonl = _HELPER.load_jsonl
load_split_ids = _HELPER.load_split_ids
load_tokenizer = _HELPER.load_tokenizer
binary_metrics = _HELPER.binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-artifact", required=True)
    parser.add_argument("--gpt-scored-bank", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--candidate-id", default="reference_answer")
    parser.add_argument("--train-size", type=int, default=450)
    parser.add_argument("--rubric-format", default="canonical")
    parser.add_argument("--max-rubric-rule-chars", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--method", choices=("platt", "temperature"), default="platt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact at {path}")
    if artifact.get("probe_classifier") != "linear":
        raise ValueError(f"Only linear probe artifacts are supported; got {artifact.get('probe_classifier')}")
    for key in ("judge_model", "best_layer", "state_dict", "scaler", "feature_dim"):
        if key not in artifact:
            raise KeyError(f"Probe artifact missing required key: {key}")
    return artifact


def expected_calibration_error(labels: torch.Tensor, probs: torch.Tensor, bins: int = 10) -> float:
    ece = 0.0
    total = float(labels.numel())
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum().item())
        if not count:
            continue
        confidence = float(probs[mask].mean().item())
        accuracy = float(labels[mask].mean().item())
        ece += (count / total) * abs(confidence - accuracy)
    return float(ece)


def probability_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    return {
        "nll": float(F.binary_cross_entropy_with_logits(logits, labels).item()),
        "brier": float(torch.mean((probs - labels) ** 2).item()),
        "ece_10": expected_calibration_error(labels, probs, bins=10),
        "mean_prob": float(probs.mean().item()),
        "mean_prob_positive": float(probs[labels == 1].mean().item()) if bool((labels == 1).any()) else 0.0,
        "mean_prob_negative": float(probs[labels == 0].mean().item()) if bool((labels == 0).any()) else 0.0,
        **{f"binary_{key}": value for key, value in binary_metrics(labels, probs, 0.5).items()},
    }


def fit_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    raw_a = torch.nn.Parameter(torch.tensor(0.0))
    b = torch.nn.Parameter(torch.tensor(0.0))
    params = [raw_a] if method == "temperature" else [raw_a, b]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best = (1.0, 0.0)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        a = F.softplus(raw_a) + 1e-6
        bias = torch.tensor(0.0) if method == "temperature" else b
        calibrated_logits = logits * a + bias
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, labels)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best = (float(a.detach().item()), float(bias.detach().item()))
    return best


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output_artifact = Path(args.output_artifact)
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_summary) if args.output_summary else output_artifact.with_name("calibration_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.probe_artifact)
    judge_model = str(artifact["judge_model"])
    layer = int(artifact["best_layer"])
    feature_dim = int(artifact["feature_dim"])

    split_args = argparse.Namespace(
        train_question_ids_file="",
        dev_question_ids_file="",
        heldout_question_ids_file="",
        split_manifest=args.split_manifest,
        train_size=args.train_size,
        # Calibration only fits on the dev split. The proberl manifest is often
        # train/dev with heldout_count=0, so do not ask the shared loader to
        # require heldout ids here.
        eval_split="all",
        eval_question_ids_file="",
    )
    _, dev_ids, _ = load_split_ids(split_args)
    records = load_jsonl(args.gpt_scored_bank)
    dev_items, _ = build_criterion_items(
        records,
        dev_ids,
        args.candidate_id,
        "dev",
        args.rubric_format,
        args.max_rubric_rule_chars,
    )
    if not dev_items:
        raise SystemExit("ERROR: empty dev split for calibration.")

    tokenizer = load_tokenizer(judge_model, args.trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    print(f"[INFO] artifact={args.probe_artifact}", flush=True)
    print(f"[INFO] judge_model={judge_model}", flush=True)
    print(f"[INFO] calibration_items={len(dev_items)} layer={layer}", flush=True)
    with torch.inference_mode():
        features = extract_features(dev_items, model, tokenizer, [layer], args.batch_size, args.max_prompt_length)[layer]

    scaler = artifact["scaler"]
    mean = scaler["mean"].float()
    std = scaler["std"].float().clamp_min(1e-6)
    if features.shape[-1] != feature_dim:
        raise ValueError(f"Feature dim mismatch: target={features.shape[-1]} artifact={feature_dim}")
    x_dev = (features - mean) / std
    probe = build_probe_classifier(feature_dim, "linear", 0, 0.0)
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    labels = labels_tensor(dev_items)
    with torch.no_grad():
        logits = probe(x_dev).squeeze(-1)

    a, b = fit_calibrator(logits, labels, args.method, args.epochs, args.lr, args.weight_decay, args.seed)
    calibrated_logits = logits * a + b
    before = probability_metrics(labels, logits)
    after = probability_metrics(labels, calibrated_logits)

    calibrated_artifact = dict(artifact)
    calibrated_artifact["format_version"] = max(int(calibrated_artifact.get("format_version", 1)), 2)
    calibrated_artifact["calibration"] = {
        "method": args.method,
        "a": float(a),
        "b": float(b),
        "threshold": 0.5,
        "fit_split": "dev",
        "fit_items": len(dev_items),
        "gpt_scored_bank": args.gpt_scored_bank,
        "split_manifest": args.split_manifest,
        "train_size": args.train_size,
        "rubric_format": args.rubric_format,
        "source_artifact": args.probe_artifact,
    }
    torch.save(calibrated_artifact, output_artifact)

    summary = {
        "source_artifact": args.probe_artifact,
        "output_artifact": str(output_artifact),
        "judge_model": judge_model,
        "layer": layer,
        "method": args.method,
        "a": float(a),
        "b": float(b),
        "threshold": 0.5,
        "num_dev_items": len(dev_items),
        "positive_rate": float(labels.mean().item()),
        "before": before,
        "after": after,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=======================================")
    print(f"Method:          {args.method}")
    print(f"a, b:            {a:.6f}, {b:.6f}")
    print(f"NLL before/after {before['nll']:.6f} -> {after['nll']:.6f}")
    print(f"Brier before/after {before['brier']:.6f} -> {after['brier']:.6f}")
    print(f"ECE before/after {before['ece_10']:.6f} -> {after['ece_10']:.6f}")
    print(f"Artifact:        {output_artifact}")
    print(f"Summary:         {summary_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
