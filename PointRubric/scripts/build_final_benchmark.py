#!/usr/bin/env python3
"""Merge construction passes and apply the released PointRubric filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more graded benchmark JSON files.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def answer_score(item: dict[str, Any], answer_id: int) -> float | None:
    for answer in item.get("answers", []):
        if int(answer.get("answer_id", -1)) == answer_id:
            try:
                return float(answer["total_score"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def keep_item(item: dict[str, Any]) -> bool:
    first = answer_score(item, 1)
    second = answer_score(item, 2)
    scores = [
        float(answer["total_score"])
        for answer in item.get("answers", [])
        if isinstance(answer.get("total_score"), (int, float))
    ]
    return first == 10 and second is not None and second <= 3 and any(2 <= score <= 8 for score in scores)


def main() -> None:
    args = parse_args()
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_count = 0

    for input_name in args.inputs:
        with Path(input_name).open(encoding="utf-8") as handle:
            rows = json.load(handle)
        input_count += len(rows)
        for item in rows:
            question_id = str(item.get("question_id", ""))
            if not question_id or question_id in seen or not keep_item(item):
                continue
            seen.add(question_id)
            merged.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"input_rows": input_count, "output_rows": len(merged), "output": str(output_path)}))


if __name__ == "__main__":
    main()
