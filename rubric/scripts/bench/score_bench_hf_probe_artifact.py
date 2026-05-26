#!/usr/bin/env python3
"""Apply a saved HF representation probe artifact to Bench.

This is for cross-dataset probing: train the tiny head elsewhere, then score
Bench without fitting on Bench labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_BENCH_HELPER_PATH = REPO_ROOT / "scripts" / "bench" / "score_bench_hf_probe.py"
_BENCH_HELPER_SPEC = importlib.util.spec_from_file_location("bench_probe_helper", _BENCH_HELPER_PATH)
if _BENCH_HELPER_SPEC is None or _BENCH_HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load Bench probe helper from {_BENCH_HELPER_PATH}")
_BENCH_HELPER = importlib.util.module_from_spec(_BENCH_HELPER_SPEC)
sys.modules[_BENCH_HELPER_SPEC.name] = _BENCH_HELPER
_BENCH_HELPER_SPEC.loader.exec_module(_BENCH_HELPER)

build_criterion_items = _BENCH_HELPER.build_criterion_items
build_prediction_rows = _BENCH_HELPER.build_prediction_rows
build_probe_classifier = _BENCH_HELPER.build_probe_classifier
compute_metrics = _BENCH_HELPER.compute_metrics
extract_features = _BENCH_HELPER.extract_features
flatten_samples = _BENCH_HELPER.flatten_samples
load_model = _BENCH_HELPER.load_model
load_tokenizer = _BENCH_HELPER.load_tokenizer
split_question_ids = _BENCH_HELPER.split_question_ids
write_csv = _BENCH_HELPER.write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-json", required=True)
    parser.add_argument("--probe-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--eval-split", choices=("all", "heldout"), default="all")
    parser.add_argument("--train-questions", type=int, default=350)
    parser.add_argument("--dev-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rubric-format", choices=("canonical", "full"), default="canonical")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact at {path}")
    if artifact.get("probe_classifier") != "linear":
        raise ValueError(
            f"Only linear probe artifacts are supported in this cross scorer; got {artifact.get('probe_classifier')}"
        )
    for key in ("judge_model", "best_layer", "best_threshold", "state_dict", "scaler", "feature_dim"):
        if key not in artifact:
            raise KeyError(f"Probe artifact missing required key: {key}")
    return artifact


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.probe_artifact)
    judge_model = str(artifact["judge_model"])
    label = args.label or f"{artifact.get('label', Path(args.probe_artifact).parent.name)}_to_bench"
    layer = int(artifact["best_layer"])
    threshold = float(artifact["best_threshold"])
    feature_dim = int(artifact["feature_dim"])

    bench_data = json.loads(Path(args.bench_json).read_text(encoding="utf-8"))
    samples = flatten_samples(bench_data)
    _, _, heldout_ids = split_question_ids(samples, args.train_questions, args.dev_questions, args.seed)
    eval_ids = {sample.question_id for sample in samples} if args.eval_split == "all" else heldout_ids
    eval_items, eval_samples = build_criterion_items(samples, eval_ids, args.eval_split, args.rubric_format)
    if not eval_items:
        raise SystemExit("ERROR: no Bench eval items selected.")

    tokenizer = load_tokenizer(judge_model, args.trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(judge_model, args.trust_remote_code)
    model.eval()

    print(f"[INFO] artifact={args.probe_artifact}", flush=True)
    print(f"[INFO] judge_model={judge_model}", flush=True)
    print(f"[INFO] target=Bench eval_split={args.eval_split} items={len(eval_items)}", flush=True)
    with torch.inference_mode():
        features = extract_features(eval_items, model, tokenizer, [layer], args.batch_size, args.max_prompt_length)[layer]

    scaler = artifact["scaler"]
    mean = scaler["mean"].float()
    std = scaler["std"].float().clamp_min(1e-6)
    if features.shape[-1] != feature_dim:
        raise ValueError(f"Feature dim mismatch: target={features.shape[-1]} artifact={feature_dim}")
    x_eval = (features - mean) / std

    probe = build_probe_classifier(feature_dim, "linear", 0, 0.0)
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    with torch.no_grad():
        probs = torch.sigmoid(probe(x_eval).squeeze(-1))

    pred_rows = build_prediction_rows(eval_samples, eval_items, probs, threshold, label)
    metrics = compute_metrics(pred_rows)
    elapsed_seconds = time.perf_counter() - started_at
    metric_row = {
        "model": label,
        **metrics,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / len(eval_samples) if eval_samples else 0.0,
        "judge_model": judge_model,
        "judge_backend": "hf_representation_probe_artifact",
        "source_artifact": str(args.probe_artifact),
        "source_artifact_type": str(artifact.get("artifact_type", "")),
        "target_dataset": "bench",
        "eval_split": args.eval_split,
        "rubric_format": args.rubric_format,
        "best_layer": layer,
        "best_threshold": threshold,
        "probe_classifier": "linear",
        "probe_trainable_params": int(artifact.get("probe_trainable_params", feature_dim + 1)),
    }
    write_csv(output_dir / "report.csv", [metric_row])
    write_csv(output_dir / "overall_metrics.csv", [metric_row])
    write_csv(output_dir / "per_sample_predictions.csv", pred_rows)
    artifact_metadata = {key: value for key, value in artifact.items() if key not in {"state_dict", "scaler"}}
    (output_dir / "artifact_application_summary.json").write_text(
        json.dumps({"artifact": artifact_metadata, "metrics": metric_row}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Label:          {label}")
    print(f"Target:         Bench {args.eval_split}")
    print(f"Parse success:  {metrics['parse_success_rate']:.4f}")
    print(f"Weighted acc:   {metrics['rule_weighted_accuracy_all_samples']:.4f}")
    print(f"Pearson:        {metrics['score_pearson']:.4f}")
    print(f"Report CSV:     {output_dir / 'report.csv'}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
