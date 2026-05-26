#!/usr/bin/env python3
"""Apply a saved HF representation probe artifact to RaR-Science.

This is for cross-dataset probing: train the tiny head elsewhere, then score
RaR-Science without fitting on RaR labels.
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

_RAR_HELPER_PATH = REPO_ROOT / "scripts" / "transfer" / "score_rarscience_response_bank_hf_probe.py"
_RAR_HELPER_SPEC = importlib.util.spec_from_file_location("rarscience_probe_helper", _RAR_HELPER_PATH)
if _RAR_HELPER_SPEC is None or _RAR_HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load RaR-Science probe helper from {_RAR_HELPER_PATH}")
_RAR_HELPER = importlib.util.module_from_spec(_RAR_HELPER_SPEC)
sys.modules[_RAR_HELPER_SPEC.name] = _RAR_HELPER
_RAR_HELPER_SPEC.loader.exec_module(_RAR_HELPER)

build_criterion_items = _RAR_HELPER.build_criterion_items
build_probe_classifier = _RAR_HELPER.build_probe_classifier
build_scored_heldout_bank = _RAR_HELPER.build_scored_heldout_bank
extract_features = _RAR_HELPER.extract_features
load_jsonl = _RAR_HELPER.load_jsonl
load_question_ids_file = _RAR_HELPER.load_question_ids_file
load_tokenizer = _RAR_HELPER.load_tokenizer
write_jsonl = _RAR_HELPER.write_jsonl
from transformers import AutoModelForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt-scored-bank",
        "--response-bank",
        dest="gpt_scored_bank",
        required=True,
        help="Input RaR-Science bank. May be a raw response bank or a pre-scored bank.",
    )
    parser.add_argument("--probe-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--candidate-id", default="reference_answer")
    parser.add_argument("--eval-split", choices=("all", "ids"), default="all")
    parser.add_argument("--eval-question-ids-file", default="")
    parser.add_argument("--rubric-format", choices=("full", "compact", "description_only", "title_only", "canonical"), default="canonical")
    parser.add_argument("--max-rubric-rule-chars", type=int, default=0)
    parser.add_argument("--score-mode", choices=("probability", "binary"), default="probability")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
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
    scored_path = output_dir / "scored_bank.jsonl"
    summary_path = output_dir / "summary.json"

    artifact = load_artifact(args.probe_artifact)
    judge_model = str(artifact["judge_model"])
    label = args.label or f"{artifact.get('label', Path(args.probe_artifact).parent.name)}_to_rarscience"
    layer = int(artifact["best_layer"])
    threshold = float(artifact["best_threshold"])
    feature_dim = int(artifact["feature_dim"])
    pooling = str(artifact.get("pooling", "last"))

    records = load_jsonl(args.gpt_scored_bank)
    if args.eval_split == "ids":
        if not args.eval_question_ids_file:
            raise SystemExit("ERROR: --eval-question-ids-file is required for --eval-split ids")
        eval_ids = load_question_ids_file(args.eval_question_ids_file)
        eval_split_name = Path(args.eval_question_ids_file).stem
    else:
        eval_ids = {str(record.get("question_id", "")) for record in records if record.get("question_id", "")}
        eval_split_name = "all"

    eval_items, eval_records = build_criterion_items(
        records,
        eval_ids,
        args.candidate_id,
        eval_split_name,
        args.rubric_format,
        args.max_rubric_rule_chars,
        require_labels=False,
    )
    if not eval_items:
        raise SystemExit("ERROR: no RaR-Science eval items selected.")

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
    print(f"[INFO] target=RaR-Science eval_split={eval_split_name} items={len(eval_items)}", flush=True)
    with torch.inference_mode():
        features = extract_features(eval_items, model, tokenizer, [layer], args.batch_size, args.max_prompt_length, pooling)[layer]

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
        logits = probe(x_eval).squeeze(-1)
        probs = torch.sigmoid(logits)

    scored_records = build_scored_heldout_bank(
        heldout_records=eval_records,
        heldout_items=eval_items,
        probs=probs,
        logits=logits,
        threshold=threshold,
        candidate_id=args.candidate_id,
        score_mode=args.score_mode,
        label=label,
        judge_model=judge_model,
        rubric_format=args.rubric_format,
        layer=layer,
        probe_classifier="linear",
        probe_hidden_dim=0,
        probe_dropout=0.0,
        probe_trainable_params=int(artifact.get("probe_trainable_params", feature_dim + 1)),
    )
    write_jsonl(scored_path, scored_records)

    elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "label": label,
        "judge_model": judge_model,
        "judge_backend": "hf_representation_probe_artifact",
        "source_artifact": str(args.probe_artifact),
        "source_artifact_type": str(artifact.get("artifact_type", "")),
        "target_dataset": "rarscience",
        "input_bank": args.gpt_scored_bank,
        "gpt_scored_bank": args.gpt_scored_bank,
        "candidate_id": args.candidate_id,
        "eval_split": eval_split_name,
        "num_questions": len(scored_records),
        "num_candidates_scored": len(scored_records),
        "num_eval_questions_scored": len(scored_records),
        "num_eval_criteria": len(eval_items),
        "rubric_format": args.rubric_format,
        "max_rubric_rule_chars": args.max_rubric_rule_chars,
        "score_mode": args.score_mode,
        "best_layer": layer,
        "best_threshold": threshold,
        "pooling": pooling,
        "probe_classifier": "linear",
        "probe_trainable_params": int(artifact.get("probe_trainable_params", feature_dim + 1)),
        "samples_parse_ok": len(scored_records),
        "samples_parse_failed": 0,
        "sample_parse_success_rate": 1.0,
        "raw_samples_parse_ok": len(scored_records),
        "raw_sample_parse_success_rate": 1.0,
        "fallback_used": 0,
        "fallback_rate": 0.0,
        "mean_judge_attempts": len(eval_items) / len(scored_records) if scored_records else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_candidate": elapsed_seconds / len(scored_records) if scored_records else 0.0,
        "seconds_per_criterion": elapsed_seconds / len(eval_items) if eval_items else 0.0,
        "total_parse_failures": 0,
        "parse_success_rate": 1.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "artifact_metadata.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in artifact.items()
                if key not in {"state_dict", "scaler"}
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Label:          {label}")
    print(f"Target:         RaR-Science {eval_split_name}")
    print(f"Scored bank:    {scored_path}")
    print(f"Summary:        {summary_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
