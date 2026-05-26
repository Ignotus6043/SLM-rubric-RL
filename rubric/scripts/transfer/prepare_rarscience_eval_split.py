#!/usr/bin/env python3
"""Create fixed RaR-Science dev/test splits for transfer and external eval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rubric_rl.rar_science_utils import (
    load_parquet_rows,
    stable_question_id,
    write_parquet_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input RaR-Science val parquet.")
    parser.add_argument("--output-dir", required=True, help="Output directory for fixed splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-questions", type=int, default=200)
    parser.add_argument("--test-questions", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_parquet_rows(args.input)
    annotated_rows = []
    seen_question_ids: set[str] = set()

    for index, row in enumerate(rows):
        qid = stable_question_id(row, index)
        if qid in seen_question_ids:
            continue
        seen_question_ids.add(qid)
        row_copy = dict(row)
        row_copy["question_id"] = qid
        annotated_rows.append(row_copy)

    total_questions = len(annotated_rows)
    required = args.dev_questions + args.test_questions
    if total_questions < required:
        raise ValueError(
            f"Need at least {required} unique questions for fixed splits, found {total_questions}."
        )

    rng = random.Random(args.seed)
    indices = list(range(total_questions))
    rng.shuffle(indices)

    dev_indices = indices[: args.dev_questions]
    test_indices = indices[args.dev_questions : args.dev_questions + args.test_questions]

    dev_rows = []
    test_rows = []
    for idx in dev_indices:
        row = dict(annotated_rows[idx])
        row["split_role"] = "dev"
        dev_rows.append(row)
    for idx in test_indices:
        row = dict(annotated_rows[idx])
        row["split_role"] = "test"
        test_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = output_dir / "dev.parquet"
    test_path = output_dir / "test.parquet"
    manifest_path = output_dir / "manifest.json"

    write_parquet_rows(dev_rows, str(dev_path))
    write_parquet_rows(test_rows, str(test_path))

    manifest = {
        "source_input": args.input,
        "seed": args.seed,
        "unique_questions_available": total_questions,
        "dev_questions": len(dev_rows),
        "test_questions": len(test_rows),
        "dev_path": str(dev_path),
        "test_path": str(test_path),
        "dev_question_ids": [row["question_id"] for row in dev_rows],
        "test_question_ids": [row["question_id"] for row in test_rows],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Source val rows:      {len(rows)}")
    print(f"Unique questions:     {total_questions}")
    print(f"Dev questions:        {len(dev_rows)}")
    print(f"Test questions:       {len(test_rows)}")
    print(f"Seed:                 {args.seed}")
    print(f"Dev split:            {dev_path}")
    print(f"Test split:           {test_path}")
    print(f"Manifest:             {manifest_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
