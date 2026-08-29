#!/usr/bin/env python3
"""Analyze OpenRubricBench human audit annotations against hidden GPT labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--hidden-key-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "1", "true"}:
        return "Yes"
    if text in {"no", "n", "0", "false"}:
        return "No"
    if text in {"unsure", "not sure", "not_sure", "tie", "unknown", "?"}:
        return "Unsure"
    return ""


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_rows = {row["item_id"]: row for row in load_csv(Path(args.annotations_csv))}
    key_rows = {row["item_id"]: row for row in load_csv(Path(args.hidden_key_csv))}

    criterion_total = 0
    criterion_agree = 0
    weighted_total = 0.0
    weighted_agree = 0.0
    unsure_total = 0
    missing_total = 0
    confusion = Counter()
    per_rule = defaultdict(lambda: {"total": 0, "agree": 0, "unsure": 0, "missing": 0})
    per_item_scores: dict[str, dict[str, Any]] = {}

    for item_id, key in key_rows.items():
        ann = ann_rows.get(item_id, {})
        human_score = 0.0
        gpt_score = 0.0
        max_score = 0.0
        complete = True
        for i in range(1, 7):
            human = norm_label(ann.get(f"human_rule_{i}", ""))
            gpt = norm_label(key.get(f"gpt_rule_{i}", ""))
            weight = float(key.get(f"rule_{i}_weight", 1) or 1)
            max_score += weight
            if gpt == "Yes":
                gpt_score += weight
            if human == "Yes":
                human_score += weight
            if human == "":
                missing_total += 1
                per_rule[i]["missing"] += 1
                complete = False
                continue
            if human == "Unsure":
                unsure_total += 1
                per_rule[i]["unsure"] += 1
                complete = False
                continue
            criterion_total += 1
            per_rule[i]["total"] += 1
            weighted_total += weight
            if human == gpt:
                criterion_agree += 1
                per_rule[i]["agree"] += 1
                weighted_agree += weight
            confusion[(gpt, human)] += 1

        per_item_scores[item_id] = {
            "item_id": item_id,
            "pair_id": key["pair_id"],
            "complete": complete,
            "human_score": human_score,
            "gpt_score": gpt_score,
            "max_score": max_score,
            "human_score_norm": human_score / max_score if max_score else None,
            "gpt_score_norm": gpt_score / max_score if max_score else None,
        }

    complete_items = [row for row in per_item_scores.values() if row["complete"]]
    human_scores = [row["human_score_norm"] for row in complete_items if row["human_score_norm"] is not None]
    gpt_scores = [row["gpt_score_norm"] for row in complete_items if row["gpt_score_norm"] is not None]
    score_mae = (
        sum(abs(h - g) for h, g in zip(human_scores, gpt_scores)) / len(human_scores)
        if human_scores
        else None
    )

    pair_groups = defaultdict(list)
    for row in complete_items:
        pair_groups[row["pair_id"]].append(row)
    pair_total = 0
    pair_agree = 0
    human_tie = 0
    pair_rows = []
    for pair_id, rows in sorted(pair_groups.items()):
        if len(rows) != 2:
            continue
        a, b = rows
        gpt_pref = 1 if a["gpt_score"] > b["gpt_score"] else -1 if a["gpt_score"] < b["gpt_score"] else 0
        human_pref = 1 if a["human_score"] > b["human_score"] else -1 if a["human_score"] < b["human_score"] else 0
        if human_pref == 0:
            human_tie += 1
        if gpt_pref != 0 and human_pref != 0:
            pair_total += 1
            if gpt_pref == human_pref:
                pair_agree += 1
        pair_rows.append(
            {
                "pair_id": pair_id,
                "item_a": a["item_id"],
                "item_b": b["item_id"],
                "gpt_score_a": a["gpt_score"],
                "gpt_score_b": b["gpt_score"],
                "human_score_a": a["human_score"],
                "human_score_b": b["human_score"],
                "gpt_pref": gpt_pref,
                "human_pref": human_pref,
                "preference_agree": gpt_pref == human_pref if gpt_pref != 0 and human_pref != 0 else "",
            }
        )

    summary = {
        "items_total": len(key_rows),
        "items_complete": len(complete_items),
        "criteria_compared_excluding_unsure_missing": criterion_total,
        "criteria_missing": missing_total,
        "criteria_unsure": unsure_total,
        "criterion_agreement": criterion_agree / criterion_total if criterion_total else None,
        "weighted_criterion_agreement": weighted_agree / weighted_total if weighted_total else None,
        "confusion_gpt_human": {f"{gpt}->{human}": count for (gpt, human), count in sorted(confusion.items())},
        "per_rule": {
            str(i): {
                **stats,
                "agreement": stats["agree"] / stats["total"] if stats["total"] else None,
            }
            for i, stats in sorted(per_rule.items())
        },
        "complete_item_score_mae_norm": score_mae,
        "complete_item_score_pearson_norm": pearson(human_scores, gpt_scores),
        "pairwise_pairs_complete_without_human_tie": pair_total,
        "pairwise_human_ties": human_tie,
        "pairwise_preference_agreement_excluding_human_ties": pair_agree / pair_total if pair_total else None,
    }

    (output_dir / "human_gpt_agreement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "pairwise_agreement.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "pair_id",
            "item_a",
            "item_b",
            "gpt_score_a",
            "gpt_score_b",
            "human_score_a",
            "human_score_b",
            "gpt_pref",
            "human_pref",
            "preference_agree",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
