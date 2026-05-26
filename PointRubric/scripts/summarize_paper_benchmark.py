#!/usr/bin/env python3
"""Summarize Bench static benchmark runs into paper-ready tables."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CORE_COLUMNS = [
    "model",
    "parse_success_rate",
    "rule_exact_match_rate_all_samples",
    "rule_weighted_accuracy_all_samples",
    "score_exact_match_rate_all_samples",
    "score_mae",
    "pairwise_order_accuracy_all_pairs",
    "top1_set_accuracy_all_questions",
    "bottom1_set_accuracy_all_questions",
    "question_full_parse_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-json", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def parse_vector(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, list):
        return [int(x) for x in parsed]
    return None


def load_rule_metadata(bench_json_path: str) -> tuple[dict[str, list[str]], dict[str, Any]]:
    payload = json.loads(Path(bench_json_path).read_text(encoding="utf-8"))
    rule_types_by_question: dict[str, list[str]] = {}
    rubric_lengths = Counter()
    hard_counts = Counter()
    soft_counts = Counter()
    answers_total = 0

    for question in payload:
        question_id = str(question.get("question_id", ""))
        refined = question.get("refined_rubric", [])
        rule_types = []
        for rule in refined:
            rule_type = "hard" if rule.get("type") == "hard" else "soft"
            rule_types.append(rule_type)
        rule_types_by_question[question_id] = rule_types
        rubric_lengths[len(rule_types)] += 1
        hard_counts[rule_types.count("hard")] += 1
        soft_counts[rule_types.count("soft")] += 1
        answers_total += len(question.get("answers", []))

    dataset_summary = {
        "questions_total": len(payload),
        "answers_total": answers_total,
        "rubric_length_distribution": dict(sorted(rubric_lengths.items())),
        "hard_rule_count_distribution": dict(sorted(hard_counts.items())),
        "soft_rule_count_distribution": dict(sorted(soft_counts.items())),
        "rubric_length_bucketing_degenerate": len(rubric_lengths) == 1,
        "rubric_length_bucketing_note": (
            "All questions have the same refined rubric length; rubric-length buckets do not separate the current Bench split."
            if len(rubric_lengths) == 1
            else "Multiple refined rubric lengths are present."
        ),
    }
    return rule_types_by_question, dataset_summary


def compute_rule_type_metrics(
    predictions_df: pd.DataFrame,
    rule_types_by_question: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in predictions_df.to_dict(orient="records"):
        model = str(row["model"])
        question_id = str(row["question_id"])
        rule_types = rule_types_by_question.get(question_id)
        if not rule_types:
            continue

        gold = parse_vector(row.get("gold_verdicts")) or []
        pred = parse_vector(row.get("pred_verdicts"))
        parse_ok = bool(row.get("parse_ok"))

        for idx, rule_type in enumerate(rule_types):
            accum[model][f"{rule_type}_total_all"] += 1
            if parse_ok and pred is not None and idx < len(gold) and idx < len(pred):
                accum[model][f"{rule_type}_matches_all"] += int(gold[idx] == pred[idx])
                accum[model][f"{rule_type}_total_parsed"] += 1
                accum[model][f"{rule_type}_matches_parsed"] += int(gold[idx] == pred[idx])
            else:
                # Parse failures count as zero matches in all-samples metrics.
                pass

    metrics: dict[str, dict[str, float]] = {}
    for model, counts in accum.items():
        metrics[model] = {
            "hard_rule_accuracy_all_samples": safe_div(counts["hard_matches_all"], counts["hard_total_all"]),
            "soft_rule_accuracy_all_samples": safe_div(counts["soft_matches_all"], counts["soft_total_all"]),
            "hard_rule_accuracy_parsed_only": safe_div(counts["hard_matches_parsed"], counts["hard_total_parsed"]),
            "soft_rule_accuracy_parsed_only": safe_div(counts["soft_matches_parsed"], counts["soft_total_parsed"]),
        }
    return metrics


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_types_by_question, dataset_summary = load_rule_metadata(args.bench_json)
    report_df = pd.read_csv(args.report_csv)
    predictions_df = pd.read_csv(args.predictions_csv)
    type_metrics = compute_rule_type_metrics(predictions_df, rule_types_by_question)

    report_rows = []
    for row in report_df.to_dict(orient="records"):
        model = str(row["model"])
        merged = {key: row[key] for key in CORE_COLUMNS if key in row}
        merged.update(type_metrics.get(model, {}))
        report_rows.append(merged)

    summary = {
        "dataset": dataset_summary,
        "models": report_rows,
        "source_report_csv": args.report_csv,
        "source_predictions_csv": args.predictions_csv,
    }

    summary_json = output_dir / "paper_benchmark_summary.json"
    summary_md = output_dir / "paper_benchmark_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Bench Paper-Ready Summary",
        "",
        "## Dataset",
        "",
        f"- Questions: `{dataset_summary['questions_total']}`",
        f"- Answer instances: `{dataset_summary['answers_total']}`",
        f"- Refined rubric lengths: `{dataset_summary['rubric_length_distribution']}`",
        f"- Hard-rule counts/question: `{dataset_summary['hard_rule_count_distribution']}`",
        f"- Soft-rule counts/question: `{dataset_summary['soft_rule_count_distribution']}`",
        f"- Rubric-length bucket note: {dataset_summary['rubric_length_bucketing_note']}",
        "",
        "## Model Summary",
        "",
        "| model | parse | rule_weighted | hard_acc | soft_acc | score_mae | pairwise | top1 | full_parse_q |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report_rows:
        lines.append(
            "| {model} | {parse:.4f} | {weighted:.4f} | {hard:.4f} | {soft:.4f} | {mae:.4f} | {pair:.4f} | {top1:.4f} | {qparse:.4f} |".format(
                model=row["model"],
                parse=float(row.get("parse_success_rate", 0.0)),
                weighted=float(row.get("rule_weighted_accuracy_all_samples", 0.0)),
                hard=float(row.get("hard_rule_accuracy_all_samples", 0.0)),
                soft=float(row.get("soft_rule_accuracy_all_samples", 0.0)),
                mae=float(row.get("score_mae", 0.0)),
                pair=float(row.get("pairwise_order_accuracy_all_pairs", 0.0)),
                top1=float(row.get("top1_set_accuracy_all_questions", 0.0)),
                qparse=float(row.get("question_full_parse_rate", 0.0)),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Dataset questions:     {dataset_summary['questions_total']}")
    print(f"Dataset answers:       {dataset_summary['answers_total']}")
    print(f"Models summarized:     {len(report_rows)}")
    print(f"Summary JSON:          {summary_json}")
    print(f"Summary MD:            {summary_md}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
