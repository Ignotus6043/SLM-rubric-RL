#!/usr/bin/env python3
"""Summarize the RaR-Science dev static robustness suite."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VARIANT_ORDER = {
    "anchor": 0,
    "verdict": 1,
    "verdict_vector": 1,
    "verdict_item": 2,
    "logprob": 3,
    "probe": 4,
    "canonical": 4,
    "loose": 5,
    "max384": 6,
    "sft": 7,
}
MODEL_ORDER = {
    "qwen3_0p6b": 0,
    "qwen3_0p6b_sft": 1,
    "qwen3_0p6b_sft_vector": 1,
    "qwen3_0p6b_sft_item": 2,
    "qwen3_1p7b": 3,
    "qwen3_1p7b_sft": 4,
    "qwen3_1p7b_sft_vector": 4,
    "qwen3_1p7b_sft_item": 5,
    "qwen3_4b": 6,
    "qwen3_4b_sft": 7,
    "qwen3_4b_sft_vector": 7,
    "qwen3_4b_sft_item": 8,
    "qwen3_8b": 9,
    "qwen3_8b_sft": 10,
    "qwen3_8b_sft_vector": 10,
    "qwen3_8b_sft_item": 11,
    "qwen3_8b_anchor": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_candidate_label(label: str) -> tuple[str, str]:
    if label.endswith("_anchor"):
        return label, "anchor"
    if label.endswith("_sft"):
        return "qwen3_1p7b_sft", "sft"
    for variant in ("verdict_vector", "verdict_item", "verdict", "logprob", "probe", "canonical", "loose", "max384"):
        suffix = f"_{variant}"
        if label.endswith(suffix):
            return label[: -len(suffix)], variant
    return label, "other"


def normalize_candidate_label(dirname: str) -> str:
    label = dirname[len("judge_") :]
    label = re.sub(r"_seed\d+_(?:dev|test)\d+$", "", label)
    label = re.sub(r"_heldout\d+$", "", label)
    return label


def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        MODEL_ORDER.get(row["model_key"], 999),
        VARIANT_ORDER.get(row["variant"], 999),
        row["candidate_label"],
    )


def format_metric(value: Any) -> str:
    if value is None:
      return "NA"
    if isinstance(value, float):
      return f"{value:.4f}"
    return str(value)


def main() -> int:
    args = parse_args()
    transfer_root = Path(args.transfer_root).resolve()
    if not transfer_root.is_dir():
        raise SystemExit(f"ERROR: transfer root not found: {transfer_root}")

    bank_candidates = sorted(transfer_root.glob("response_bank_seed*_dev*/response_bank_summary.json"))
    bank_summary = load_json(bank_candidates[0]) if bank_candidates else None

    rows: list[dict[str, Any]] = []
    anchor_summary = None

    for summary_path in sorted(transfer_root.glob("judge_*/summary.json")):
        candidate_dir = summary_path.parent
        scoring_summary = load_json(summary_path)
        candidate_label = normalize_candidate_label(candidate_dir.name)
        model_key, variant = split_candidate_label(candidate_label)

        alignment_paths = sorted(candidate_dir.glob("alignment_vs_*/alignment_summary.json"))
        alignment_summary = load_json(alignment_paths[0]) if alignment_paths else None

        row = {
            "candidate_label": candidate_label,
            "model_key": model_key,
            "variant": variant,
            "judge_model": scoring_summary.get("judge_model", ""),
            "judge_label": scoring_summary.get("label", ""),
            "num_questions": scoring_summary.get("num_questions", 0),
            "num_candidates_scored": scoring_summary.get("num_candidates_scored", 0),
            "sample_parse_success_rate": scoring_summary.get("sample_parse_success_rate"),
            "raw_sample_parse_success_rate": scoring_summary.get("raw_sample_parse_success_rate"),
            "fallback_rate": scoring_summary.get("fallback_rate"),
            "fallback_used": scoring_summary.get("fallback_used"),
            "mean_judge_attempts": scoring_summary.get("mean_judge_attempts"),
            "paired_sample_parse_success_rate": (
                alignment_summary.get("parse_success_rate")
                if alignment_summary is not None
                else None
            ),
            "candidate_parse_success_rate": (
                alignment_summary.get("candidate_parse_success_rate")
                if alignment_summary is not None
                else scoring_summary.get("parse_success_rate")
            ),
            "samples_parse_ok": scoring_summary.get("samples_parse_ok"),
            "paired_samples_parse_ok": alignment_summary.get("samples_parse_ok") if alignment_summary else None,
            "rule_exact_match_rate": alignment_summary.get("rule_exact_match_rate") if alignment_summary else None,
            "rule_hamming_accuracy": alignment_summary.get("rule_hamming_accuracy") if alignment_summary else None,
            "rule_weighted_accuracy_abs": (
                alignment_summary.get("rule_weighted_accuracy_abs") if alignment_summary else None
            ),
            "macro_f1": alignment_summary.get("macro_f1") if alignment_summary else None,
            "balanced_accuracy": alignment_summary.get("balanced_accuracy") if alignment_summary else None,
            "mcc": alignment_summary.get("mcc") if alignment_summary else None,
            "score_mae": alignment_summary.get("score_mae") if alignment_summary else None,
            "score_pearson": alignment_summary.get("score_pearson") if alignment_summary else None,
            "score_spearman": alignment_summary.get("score_spearman") if alignment_summary else None,
            "parse_success_rate_raw": scoring_summary.get("parse_success_rate"),
            "summary_path": str(summary_path),
            "alignment_summary_path": str(alignment_paths[0]) if alignment_paths else "",
        }
        rows.append(row)
        if variant == "anchor":
            anchor_summary = row

    rows.sort(key=sort_key)

    output_json = transfer_root / "static_suite_summary.json"
    output_md = transfer_root / "static_suite_summary.md"

    summary = {
        "transfer_root": str(transfer_root),
        "bank_summary_path": str(bank_candidates[0]) if bank_candidates else "",
        "bank_summary": bank_summary,
        "anchor_summary": anchor_summary,
        "results": rows,
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RaR-Science Dev Static Suite Summary",
        "",
        f"- Transfer root: `{transfer_root}`",
        f"- Bank summary: `{summary['bank_summary_path'] or 'MISSING'}`",
        f"- Anchor: `{anchor_summary['candidate_label'] if anchor_summary else 'MISSING'}`",
        "- Note: `Sample Parse` is effective full-vector label coverage after retry/fallback. `Raw Parse` excludes fallback. `Paired Parse` additionally requires the anchor to have an effective label. `Item Parse` is per-rule effective coverage.",
        "",
        "| Candidate | Variant | Sample Parse | Raw Parse | Fallback | Attempts | Paired Parse | Item Parse | Rule Exact | Rule Acc | Weighted Acc | Macro F1 | Bal Acc | Score MAE | Pearson | Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_label']} | {row['variant']} | "
            f"{format_metric(row['sample_parse_success_rate'])} | "
            f"{format_metric(row['raw_sample_parse_success_rate'])} | "
            f"{format_metric(row['fallback_rate'])} | "
            f"{format_metric(row['mean_judge_attempts'])} | "
            f"{format_metric(row['paired_sample_parse_success_rate'])} | "
            f"{format_metric(row['candidate_parse_success_rate'])} | "
            f"{format_metric(row['rule_exact_match_rate'])} | "
            f"{format_metric(row['rule_hamming_accuracy'])} | "
            f"{format_metric(row['rule_weighted_accuracy_abs'])} | "
            f"{format_metric(row['macro_f1'])} | "
            f"{format_metric(row['balanced_accuracy'])} | "
            f"{format_metric(row['score_mae'])} | "
            f"{format_metric(row['score_pearson'])} | "
            f"{format_metric(row['score_spearman'])} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Transfer root:          {transfer_root}")
    print(f"Bank summary:           {summary['bank_summary_path'] or 'MISSING'}")
    print(f"Candidates summarized:  {len(rows)}")
    print(f"Summary JSON:           {output_json}")
    print(f"Summary MD:             {output_md}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
