#!/usr/bin/env python3
"""Compare RaR-Science judge scores against a pseudo-gold reference judge."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-scored-bank", required=True)
    parser.add_argument("--candidate-scored-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args()


def load_scored_bank(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def almost_equal(x: float, y: float, tol: float = 1e-12) -> bool:
    return abs(x - y) <= tol


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


def flatten_bank(records: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for question in records:
        qid = question["question_id"]
        for candidate in question["candidates"]:
            cid = candidate["candidate_id"]
            by_pair[(qid, cid)] = candidate
            by_question[qid][cid] = candidate
    return by_pair, by_question


def criterion_scores(candidate: dict[str, Any]) -> list[int]:
    reward_result = candidate.get("reward_result", {})
    scores = reward_result.get("criterion_scores") if isinstance(reward_result, dict) else None
    if isinstance(scores, list):
        return [int(x) for x in scores]
    return []


def criterion_weights(candidate: dict[str, Any]) -> list[float]:
    reward_result = candidate.get("reward_result", {})
    weights = reward_result.get("criterion_weights") if isinstance(reward_result, dict) else None
    if isinstance(weights, list):
        return [float(x) for x in weights]
    return []


def num_parse_failures(candidate: dict[str, Any]) -> int:
    reward_result = candidate.get("reward_result", {})
    if isinstance(reward_result, dict):
        return int(reward_result.get("num_parse_failures", 0))
    return 0


def total_judged_items(candidate: dict[str, Any]) -> int:
    scores = criterion_scores(candidate)
    return len(scores) if scores else 1


def sample_parse_ok(candidate: dict[str, Any]) -> bool:
    scores = criterion_scores(candidate)
    return bool(scores) and num_parse_failures(candidate) == 0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def matthews_corrcoef(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_div(num, den)


def pairwise_order_accuracy(
    ref_by_question: dict[str, dict[str, dict[str, Any]]],
    cand_by_question: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, float]:
    correct = 0
    total = 0
    for qid, ref_candidates in ref_by_question.items():
        cand_candidates = cand_by_question.get(qid, {})
        shared_ids = sorted(set(ref_candidates) & set(cand_candidates))
        if len(shared_ids) < 2:
            continue
        for left, right in itertools.combinations(shared_ids, 2):
            ref_diff = float(ref_candidates[left]["score"]) - float(ref_candidates[right]["score"])
            cand_diff = float(cand_candidates[left]["score"]) - float(cand_candidates[right]["score"])
            total += 1
            if (
                (ref_diff == 0 and cand_diff == 0)
                or (ref_diff > 0 and cand_diff > 0)
                or (ref_diff < 0 and cand_diff < 0)
            ):
                correct += 1
    return {"pairwise_order_accuracy": safe_div(correct, total), "pairwise_pairs_total": total}


def init_candidate_accumulator() -> dict[str, Any]:
    return {
        "samples_total": 0,
        "samples_parse_ok": 0,
        "candidate_parse_failures": 0,
        "candidate_judged_items": 0,
        "reference_parse_failures": 0,
        "reference_judged_items": 0,
        "rule_exact_matches": 0,
        "rule_items_total": 0,
        "rule_items_matched": 0,
        "rule_positive_total": 0,
        "rule_positive_matched": 0,
        "rule_penalty_total": 0,
        "rule_penalty_matched": 0,
        "rule_weight_total_abs": 0.0,
        "rule_weight_matched_abs": 0.0,
        "rule_weight_total_positive": 0.0,
        "rule_weight_matched_positive": 0.0,
        "rule_weight_total_penalty_abs": 0.0,
        "rule_weight_matched_penalty_abs": 0.0,
        "yes_tp": 0,
        "yes_fp": 0,
        "yes_fn": 0,
        "yes_tn": 0,
        "score_exact_matches": 0,
        "abs_errors": [],
        "sq_errors": [],
        "signed_errors": [],
        "reference_scores": [],
        "candidate_scores": [],
    }


def finalize_candidate_summary(acc: dict[str, Any]) -> dict[str, Any]:
    samples_total = acc["samples_total"]
    samples_parse_ok = acc["samples_parse_ok"]
    samples_parse_failed = samples_total - samples_parse_ok

    yes_precision, yes_recall, yes_f1 = precision_recall_f1(acc["yes_tp"], acc["yes_fp"], acc["yes_fn"])
    no_precision, no_recall, no_f1 = precision_recall_f1(acc["yes_tn"], acc["yes_fn"], acc["yes_fp"])

    abs_errors = acc["abs_errors"]
    sq_errors = acc["sq_errors"]
    signed_errors = acc["signed_errors"]
    ref_scores = acc["reference_scores"]
    cand_scores = acc["candidate_scores"]

    return {
        "samples_total": samples_total,
        "samples_parse_ok": samples_parse_ok,
        "samples_parse_failed": samples_parse_failed,
        "parse_success_rate": safe_div(samples_parse_ok, samples_total),
        "reference_parse_success_rate": (
            1.0 - safe_div(acc["reference_parse_failures"], acc["reference_judged_items"])
            if acc["reference_judged_items"]
            else 1.0
        ),
        "candidate_parse_success_rate": (
            1.0 - safe_div(acc["candidate_parse_failures"], acc["candidate_judged_items"])
            if acc["candidate_judged_items"]
            else 1.0
        ),
        "rule_exact_match_rate": safe_div(acc["rule_exact_matches"], samples_parse_ok),
        "rule_hamming_accuracy": safe_div(acc["rule_items_matched"], acc["rule_items_total"]),
        "rule_positive_accuracy": safe_div(acc["rule_positive_matched"], acc["rule_positive_total"]),
        "rule_penalty_accuracy": safe_div(acc["rule_penalty_matched"], acc["rule_penalty_total"]),
        "rule_weighted_accuracy_abs": safe_div(acc["rule_weight_matched_abs"], acc["rule_weight_total_abs"]),
        "rule_weighted_accuracy_positive": safe_div(
            acc["rule_weight_matched_positive"], acc["rule_weight_total_positive"]
        ),
        "rule_weighted_accuracy_penalty_abs": safe_div(
            acc["rule_weight_matched_penalty_abs"], acc["rule_weight_total_penalty_abs"]
        ),
        "yes_precision": yes_precision,
        "yes_recall": yes_recall,
        "yes_f1": yes_f1,
        "no_precision": no_precision,
        "no_recall": no_recall,
        "no_f1": no_f1,
        "macro_f1": (yes_f1 + no_f1) / 2.0,
        "balanced_accuracy": (yes_recall + no_recall) / 2.0,
        "jaccard_yes": safe_div(acc["yes_tp"], acc["yes_tp"] + acc["yes_fp"] + acc["yes_fn"]),
        "mcc": matthews_corrcoef(acc["yes_tp"], acc["yes_fp"], acc["yes_fn"], acc["yes_tn"]),
        "score_exact_match_rate": safe_div(acc["score_exact_matches"], samples_parse_ok),
        "score_mae": mean(abs_errors) if abs_errors else 0.0,
        "score_rmse": math.sqrt(mean(sq_errors)) if sq_errors else 0.0,
        "score_mean_error_bias": mean(signed_errors) if signed_errors else 0.0,
        "score_within_0p05_rate": safe_div(sum(1 for x in abs_errors if x <= 0.05), len(abs_errors)),
        "score_within_0p10_rate": safe_div(sum(1 for x in abs_errors if x <= 0.10), len(abs_errors)),
        "score_pearson": pearson_corr(ref_scores, cand_scores),
        "score_spearman": spearman_corr(ref_scores, cand_scores),
        "reference_mean_score": mean(ref_scores) if ref_scores else 0.0,
        "candidate_mean_score": mean(cand_scores) if cand_scores else 0.0,
    }


def main() -> int:
    args = parse_args()
    reference_records = load_scored_bank(args.reference_scored_bank)
    candidate_records = load_scored_bank(args.candidate_scored_bank)

    ref_flat, ref_by_question = flatten_bank(reference_records)
    cand_flat, cand_by_question = flatten_bank(candidate_records)

    shared_keys = sorted(set(ref_flat) & set(cand_flat))
    if not shared_keys:
        raise ValueError("No overlapping (question_id, candidate_id) pairs between the scored banks.")

    overall = init_candidate_accumulator()
    per_candidate_accumulators: dict[str, dict[str, Any]] = defaultdict(init_candidate_accumulator)
    candidate_parse_failures = 0
    candidate_judged_items = 0
    reference_parse_failures = 0
    reference_judged_items = 0
    parsed_ref_by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    parsed_cand_by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for key in shared_keys:
        ref_candidate = ref_flat[key]
        cand_candidate = cand_flat[key]

        ref_items = criterion_scores(ref_candidate)
        cand_items = criterion_scores(cand_candidate)
        ref_weights = criterion_weights(ref_candidate)
        candidate_id = key[1]

        for acc in (overall, per_candidate_accumulators[candidate_id]):
            acc["samples_total"] += 1
            acc["candidate_parse_failures"] += num_parse_failures(cand_candidate)
            acc["candidate_judged_items"] += total_judged_items(cand_candidate)
            acc["reference_parse_failures"] += num_parse_failures(ref_candidate)
            acc["reference_judged_items"] += total_judged_items(ref_candidate)

        candidate_parse_failures += num_parse_failures(cand_candidate)
        candidate_judged_items += total_judged_items(cand_candidate)
        reference_parse_failures += num_parse_failures(ref_candidate)
        reference_judged_items += total_judged_items(ref_candidate)

        parsed_pair = (
            sample_parse_ok(ref_candidate)
            and sample_parse_ok(cand_candidate)
            and len(ref_items) == len(cand_items)
            and len(ref_items) == len(ref_weights)
            and bool(ref_items)
        )
        if not parsed_pair:
            continue

        parsed_ref_by_question[key[0]][candidate_id] = ref_candidate
        parsed_cand_by_question[key[0]][candidate_id] = cand_candidate

        ref_score = float(ref_candidate["score"])
        cand_score = float(cand_candidate["score"])
        abs_error = abs(ref_score - cand_score)
        signed_error = cand_score - ref_score
        sq_error = signed_error ** 2

        for acc in (overall, per_candidate_accumulators[candidate_id]):
            acc["samples_parse_ok"] += 1
            acc["reference_scores"].append(ref_score)
            acc["candidate_scores"].append(cand_score)
            acc["abs_errors"].append(abs_error)
            acc["sq_errors"].append(sq_error)
            acc["signed_errors"].append(signed_error)
            acc["score_exact_matches"] += int(almost_equal(ref_score, cand_score))
            acc["rule_exact_matches"] += int(ref_items == cand_items)

            for ref_item, cand_item, weight in zip(ref_items, cand_items, ref_weights, strict=True):
                matched = int(ref_item == cand_item)
                abs_weight = abs(weight)

                acc["rule_items_total"] += 1
                acc["rule_items_matched"] += matched
                acc["rule_weight_total_abs"] += abs_weight
                acc["rule_weight_matched_abs"] += abs_weight * matched

                if ref_item == 1 and cand_item == 1:
                    acc["yes_tp"] += 1
                elif ref_item == 0 and cand_item == 1:
                    acc["yes_fp"] += 1
                elif ref_item == 1 and cand_item == 0:
                    acc["yes_fn"] += 1
                else:
                    acc["yes_tn"] += 1

                if weight > 0:
                    acc["rule_positive_total"] += 1
                    acc["rule_positive_matched"] += matched
                    acc["rule_weight_total_positive"] += weight
                    acc["rule_weight_matched_positive"] += weight * matched
                elif weight < 0:
                    acc["rule_penalty_total"] += 1
                    acc["rule_penalty_matched"] += matched
                    acc["rule_weight_total_penalty_abs"] += abs_weight
                    acc["rule_weight_matched_penalty_abs"] += abs_weight * matched

    pairwise = pairwise_order_accuracy(parsed_ref_by_question, parsed_cand_by_question)

    overall_summary = finalize_candidate_summary(overall)
    per_candidate_summary = {
        candidate_id: finalize_candidate_summary(acc)
        for candidate_id, acc in sorted(per_candidate_accumulators.items())
    }

    summary = {
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "reference_scored_bank": args.reference_scored_bank,
        "candidate_scored_bank": args.candidate_scored_bank,
        "shared_question_candidate_pairs": len(shared_keys),
        **overall_summary,
        # Backward-compatible aliases for older analysis code.
        "criterion_item_agreement": overall_summary["rule_hamming_accuracy"],
        "criterion_items_total": overall["rule_items_total"],
        "score_mean_absolute_error": overall_summary["score_mae"],
        "candidate_parse_success_rate": (
            1.0 - (candidate_parse_failures / candidate_judged_items)
            if candidate_judged_items
            else 1.0
        ),
        "reference_parse_success_rate": (
            1.0 - (reference_parse_failures / reference_judged_items)
            if reference_judged_items
            else 1.0
        ),
        **pairwise,
        "per_candidate_summary": per_candidate_summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "alignment_summary.json"
    markdown_path = output_dir / "alignment_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown_lines = [
        "# RaR-Science Judge Alignment Summary",
        "",
        f"- Reference: `{args.reference_label}`",
        f"- Candidate: `{args.candidate_label}`",
        f"- Shared pairs: `{summary['shared_question_candidate_pairs']}`",
        f"- Parse success: `{summary['parse_success_rate']:.4f}`",
        f"- Rule exact match: `{summary['rule_exact_match_rate']:.4f}`",
        f"- Rule yes/no accuracy: `{summary['rule_hamming_accuracy']:.4f}`",
        f"- Rule weighted accuracy (abs weight): `{summary['rule_weighted_accuracy_abs']:.4f}`",
        f"- Rule weighted accuracy (positive): `{summary['rule_weighted_accuracy_positive']:.4f}`",
        f"- Rule weighted accuracy (penalty abs): `{summary['rule_weighted_accuracy_penalty_abs']:.4f}`",
        f"- Total-score exact match: `{summary['score_exact_match_rate']:.4f}`",
        f"- Total-score MAE: `{summary['score_mae']:.4f}`",
        f"- Total-score RMSE: `{summary['score_rmse']:.4f}`",
        f"- Total-score Pearson: `{summary['score_pearson']:.4f}`",
        f"- Total-score Spearman: `{summary['score_spearman']:.4f}`",
        f"- Pairwise order accuracy: `{summary['pairwise_order_accuracy']:.4f}` over `{summary['pairwise_pairs_total']}` pairs",
        "",
        "## Per-Candidate Type",
        "",
        "| candidate_id | n_total | n_parse_ok | rule_exact | yes/no_acc | weighted_acc_abs | score_exact | mae | pearson | spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate_id, metrics in summary["per_candidate_summary"].items():
        markdown_lines.append(
            f"| {candidate_id} | {metrics['samples_total']} | {metrics['samples_parse_ok']} | "
            f"{metrics['rule_exact_match_rate']:.4f} | {metrics['rule_hamming_accuracy']:.4f} | "
            f"{metrics['rule_weighted_accuracy_abs']:.4f} | {metrics['score_exact_match_rate']:.4f} | "
            f"{metrics['score_mae']:.4f} | {metrics['score_pearson']:.4f} | {metrics['score_spearman']:.4f} |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Reference label:       {args.reference_label}")
    print(f"Candidate label:       {args.candidate_label}")
    print(f"Shared pairs:          {summary['shared_question_candidate_pairs']}")
    print(f"Parse success:         {summary['parse_success_rate']:.4f}")
    print(f"Rule yes/no acc.:      {summary['rule_hamming_accuracy']:.4f}")
    print(f"Rule weighted acc.:    {summary['rule_weighted_accuracy_abs']:.4f}")
    print(f"Score MAE:             {summary['score_mae']:.4f}")
    print(f"Pairwise order acc.:   {summary['pairwise_order_accuracy']:.4f}")
    print(f"Summary JSON:          {summary_path}")
    print(f"Summary MD:            {markdown_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
