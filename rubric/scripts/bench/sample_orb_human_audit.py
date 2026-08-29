#!/usr/bin/env python3
"""Sample OpenRubricBench heldout responses for criterion-level human audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-test-json",
        default="Bench/data/fixed_split_seed42_train40_dev10_test50/test.json",
        help="OpenRubricBench heldout/test JSON used by the paper-ready small-model runs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--items-per-md", type=int, default=10)
    return parser.parse_args()


def normalize_verdict(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true"}:
        return "Yes"
    if text in {"no", "n", "0", "false"}:
        return "No"
    return text


def rule_weight(rule_type: str) -> int:
    return 3 if str(rule_type).strip().lower() == "hard" else 1


def clean_cell(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def response_grade(answer: dict[str, Any], num_rules: int) -> list[str]:
    by_rule: dict[int, str] = {}
    for item in answer.get("grade", []):
        try:
            rule_id = int(item.get("rule_id"))
        except (TypeError, ValueError):
            continue
        by_rule[rule_id] = normalize_verdict(item.get("verdict"))
    return [by_rule.get(i, "") for i in range(1, num_rules + 1)]


def sample_pairs(records: list[dict[str, Any]], rng: random.Random, num_pairs: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        answers = record.get("answers", [])
        valid_pairs = [
            (a, b)
            for a, b in combinations(answers, 2)
            if a.get("total_score") != b.get("total_score")
        ]
        if valid_pairs:
            candidates.append({"record": record, "pairs": valid_pairs})
    if len(candidates) < num_pairs:
        raise ValueError(f"Need {num_pairs} eligible questions, found {len(candidates)}")
    selected = rng.sample(candidates, num_pairs)
    sampled: list[dict[str, Any]] = []
    for pair_index, item in enumerate(selected, start=1):
        a, b = rng.choice(item["pairs"])
        sampled.append({"pair_index": pair_index, "record": item["record"], "answers": [a, b]})
    return sampled


def build_items(sampled_pairs: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pair in sampled_pairs:
        record = pair["record"]
        pair_id = f"P{pair['pair_index']:03d}"
        for member_index, answer in enumerate(pair["answers"], start=1):
            items.append(
                {
                    "pair_id": pair_id,
                    "pair_member": member_index,
                    "question_id": str(record.get("question_id", "")),
                    "question": clean_cell(record.get("question", "")),
                    "rubric": record.get("refined_rubric", []),
                    "answer_id": answer.get("answer_id"),
                    "model": answer.get("model", ""),
                    "response": clean_cell(answer.get("answer_text", "")),
                    "gpt_total_score": answer.get("total_score"),
                    "gpt_labels": response_grade(answer, len(record.get("refined_rubric", []))),
                }
            )
    rng.shuffle(items)
    for item_index, item in enumerate(items, start=1):
        item["item_id"] = f"ORB-AUDIT-{item_index:03d}"
    return items


def md_for_item(item: dict[str, Any]) -> str:
    lines = [
        f"## {item['item_id']}",
        "",
        f"Pair ID: `{item['pair_id']}`",
        f"Question ID: `{item['question_id']}`",
        "",
        "### Question",
        item["question"],
        "",
        "### Rubric Criteria",
    ]
    for rule in item["rubric"]:
        rule_id = rule.get("rule_id", "")
        rule_type = str(rule.get("type", "")).lower()
        weight = rule_weight(rule_type)
        rubric = clean_cell(rule.get("rubric", ""))
        lines.append(f"{rule_id}. [{rule_type}; weight {weight}] {rubric}")
    lines.extend(["", "### Candidate Response", item["response"], ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    bench_path = Path(args.bench_test_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_dir = output_dir / "items_md"
    md_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(bench_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    sampled_pairs = sample_pairs(records, rng, args.num_pairs)
    items = build_items(sampled_pairs, rng)

    visible_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    key_jsonl = output_dir / "audit_key_hidden.jsonl"
    visible_jsonl = output_dir / "audit_items_visible.jsonl"
    with key_jsonl.open("w", encoding="utf-8") as key_handle, visible_jsonl.open("w", encoding="utf-8") as visible_handle:
        for item in items:
            rules = item["rubric"]
            visible = {
                "item_id": item["item_id"],
                "pair_id": item["pair_id"],
                "question_id": item["question_id"],
                "question": item["question"],
                "rubric": rules,
                "response": item["response"],
            }
            visible_handle.write(json.dumps(visible, ensure_ascii=False) + "\n")

            row = {
                "item_id": item["item_id"],
                "pair_id": item["pair_id"],
                "question_id": item["question_id"],
                "confidence_1_to_5": "",
                "notes": "",
                "annotator": "",
            }
            key_row = {
                "item_id": item["item_id"],
                "pair_id": item["pair_id"],
                "pair_member": item["pair_member"],
                "question_id": item["question_id"],
                "answer_id": item["answer_id"],
                "model": item["model"],
                "gpt_total_score": item["gpt_total_score"],
            }
            for i, rule in enumerate(rules, start=1):
                row[f"human_rule_{i}"] = ""
                key_row[f"rule_{i}_type"] = rule.get("type", "")
                key_row[f"rule_{i}_weight"] = rule_weight(str(rule.get("type", "")))
                key_row[f"gpt_rule_{i}"] = item["gpt_labels"][i - 1] if i - 1 < len(item["gpt_labels"]) else ""
            visible_rows.append(row)
            hidden_rows.append(key_row)
            key_handle.write(json.dumps(key_row, ensure_ascii=False) + "\n")

    rule_cols = [f"human_rule_{i}" for i in range(1, 7)]
    write_csv(
        output_dir / "annotations_empty.csv",
        visible_rows,
        ["item_id", "pair_id", "question_id", *rule_cols, "confidence_1_to_5", "notes", "annotator"],
    )
    key_cols = [
        "item_id",
        "pair_id",
        "pair_member",
        "question_id",
        "answer_id",
        "model",
        "gpt_total_score",
    ]
    for i in range(1, 7):
        key_cols.extend([f"rule_{i}_type", f"rule_{i}_weight", f"gpt_rule_{i}"])
    write_csv(output_dir / "audit_key_hidden.csv", hidden_rows, key_cols)

    for chunk_start in range(0, len(items), args.items_per_md):
        chunk = items[chunk_start : chunk_start + args.items_per_md]
        first = chunk_start + 1
        last = chunk_start + len(chunk)
        path = md_dir / f"items_{first:03d}_{last:03d}.md"
        path.write_text("\n---\n\n".join(md_for_item(item) for item in chunk), encoding="utf-8")

    sampled_manifest = {
        "source": str(bench_path),
        "source_questions": len(records),
        "split": "heldout521_train417_dev104_seed42",
        "seed": args.seed,
        "num_pairs": args.num_pairs,
        "num_responses": len(items),
        "sampling": "Sample 50 heldout questions without replacement; for each, choose 2 of 4 responses uniformly among pairs with different GPT total_score; shuffle the resulting 100 responses.",
        "visible_items": str(visible_jsonl),
        "annotation_csv": str(output_dir / "annotations_empty.csv"),
        "hidden_key_csv": str(output_dir / "audit_key_hidden.csv"),
        "markdown_dir": str(md_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(sampled_manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# OpenRubricBench Human Audit Sample",
                "",
                "Fill `annotations_empty.csv` using `Yes`, `No`, or `Unsure` for each `human_rule_*` column.",
                "Use the files in `items_md/` for reading. They are split into 10 responses per file.",
                "Do not inspect `audit_key_hidden.csv` or `audit_key_hidden.jsonl` while annotating; those contain GPT labels and scores for later agreement analysis.",
                "",
                "Sampling protocol: 50 heldout questions, 2 responses per question, selected from response pairs with different GPT total scores, then shuffled into 100 response items.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sampled_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
