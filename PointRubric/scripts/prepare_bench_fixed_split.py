#!/usr/bin/env python3
"""Create a fixed question-level Bench split for tuned-judge evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_frac = args.train_frac + args.dev_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-8:
        raise SystemExit("train/dev/test fractions must sum to 1.0")

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(round(n_total * args.train_frac))
    n_dev = int(round(n_total * args.dev_frac))
    if n_train + n_dev > n_total:
        n_dev = max(0, n_total - n_train)
    n_test = n_total - n_train - n_dev

    train_idx = sorted(indices[:n_train])
    dev_idx = sorted(indices[n_train:n_train + n_dev])
    test_idx = sorted(indices[n_train + n_dev:])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [data[idx] for idx in train_idx],
        "dev": [data[idx] for idx in dev_idx],
        "test": [data[idx] for idx in test_idx],
    }
    for split_name, rows in splits.items():
        (output_dir / f"{split_name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "input": args.input,
        "seed": args.seed,
        "fractions": {
            "train": args.train_frac,
            "dev": args.dev_frac,
            "test": args.test_frac,
        },
        "counts": {
            "total_questions": n_total,
            "train_questions": len(splits["train"]),
            "dev_questions": len(splits["dev"]),
            "test_questions": len(splits["test"]),
            "train_answer_instances": sum(len(row.get("answers", [])) for row in splits["train"]),
            "dev_answer_instances": sum(len(row.get("answers", [])) for row in splits["dev"]),
            "test_answer_instances": sum(len(row.get("answers", [])) for row in splits["test"]),
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=======================================")
    print(f"Input:                 {args.input}")
    print(f"Output dir:            {output_dir}")
    print(f"Seed:                  {args.seed}")
    print(f"Questions train/dev/test: {len(splits['train'])} / {len(splits['dev'])} / {len(splits['test'])}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
