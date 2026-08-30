#!/usr/bin/env python3
"""Build deterministic GPT-label verdict splits from a RaR-Science scored bank."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """
You are an impartial rubric grader.

Instruction:
{question}

Candidate Response:
{response}

Rubric Rules ({num_rules} total):
{rubric_rules}

Task:
For each rubric rule, decide whether the response satisfies it.

Output format requirement (strict):
- Return exactly {num_rules} lines.
- Each line must follow this format exactly:
  Rule i: Yes/No
- Use i = 1..{num_rules} in order.
- Use only Yes or No after the colon.

Do not output reasoning, JSON, markdown, headings, or any text outside those rule lines.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-scored-bank", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[50, 100, 200, 350])
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--heldout-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-id", default="reference_answer")
    return parser.parse_args()


def format_rubric_rules(rubric: list[dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(rubric, start=1):
        title = str(item.get("title", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        text = f"{title}: {description}" if title and description else title or description
        lines.append(f"{index}. {text}")
    return "\n".join(lines)


def build_prompt(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    rubric = record.get("rubric") or []
    return PROMPT_TEMPLATE.format(
        question=record.get("question", ""),
        response=candidate.get("response", ""),
        rubric_rules=format_rubric_rules(rubric),
        num_rules=len(rubric),
    )


def build_target(scores: list[Any]) -> str:
    return "\n".join(
        f"Rule {index}: {'Yes' if float(score) >= 0.5 else 'No'}"
        for index, score in enumerate(scores, start=1)
    )


def load_examples(path: Path, candidate_id: str) -> list[dict[str, Any]]:
    examples = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = [
                candidate
                for candidate in record.get("candidates", [])
                if candidate.get("candidate_id") == candidate_id
            ]
            if not candidates:
                skipped += 1
                continue
            candidate = candidates[0]
            reward = candidate.get("reward_result") or {}
            scores = reward.get("criterion_scores") or []
            rubric = record.get("rubric") or []
            parse_failures = int(candidate.get("num_parse_failures", reward.get("num_parse_failures", 0)))
            if parse_failures != 0 or len(scores) != len(rubric) or not scores:
                skipped += 1
                continue
            examples.append(
                {
                    "id": f"{record.get('question_id')}_{candidate_id}",
                    "question_id": str(record.get("question_id", "")),
                    "candidate_id": candidate_id,
                    "instruction": build_prompt(record, candidate),
                    "input": "",
                    "output": build_target(scores),
                    "gpt_score": candidate.get("score"),
                    "num_rules": len(rubric),
                }
            )
    if skipped:
        print(f"[WARN] skipped {skipped} examples due to missing candidate or parse failure.")
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    examples = load_examples(Path(args.gpt_scored_bank), args.candidate_id)
    largest_train_size = max(args.train_sizes)
    required = largest_train_size + args.dev_count + args.heldout_count
    if len(examples) < required:
        raise SystemExit(f"ERROR: need {required} parsed examples, found {len(examples)}.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    train_pool = examples[:largest_train_size]
    dev = examples[largest_train_size : largest_train_size + args.dev_count]
    heldout = examples[
        largest_train_size + args.dev_count : largest_train_size + args.dev_count + args.heldout_count
    ]
    dev_ids = [row["question_id"] for row in dev]
    heldout_ids = [row["question_id"] for row in heldout]
    (output_root / "dev_question_ids.txt").write_text("\n".join(dev_ids) + "\n", encoding="utf-8")
    (output_root / "heldout_question_ids.txt").write_text("\n".join(heldout_ids) + "\n", encoding="utf-8")

    split_metadata = {}
    for size in args.train_sizes:
        size_dir = output_root / f"train{size}"
        train = train_pool[:size]
        write_jsonl(size_dir / "sft_train.jsonl", train)
        write_jsonl(size_dir / "sft_dev.jsonl", dev)
        write_jsonl(size_dir / "sft_test.jsonl", heldout)
        split_metadata[str(size)] = {
            "train_examples": len(train),
            "dev_examples": len(dev),
            "heldout_examples": len(heldout),
            "train_question_ids": [row["question_id"] for row in train],
            "dev_question_ids": dev_ids,
            "heldout_question_ids": heldout_ids,
            "data_dir": str(size_dir),
        }

    manifest = {
        "task": "rarscience_gpt_verdict_sft_nested_splits",
        "gpt_scored_bank": str(Path(args.gpt_scored_bank)),
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "available_examples": len(examples),
        "train_sizes": args.train_sizes,
        "dev_count": args.dev_count,
        "heldout_count": args.heldout_count,
        "heldout_question_ids_file": str(output_root / "heldout_question_ids.txt"),
        "dev_question_ids_file": str(output_root / "dev_question_ids.txt"),
        "splits": split_metadata,
    }
    manifest_path = output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=======================================")
    print("RaR-Science GPT verdict splits ready")
    print(f"Output root:      {output_root}")
    print(f"Train sizes:      {' '.join(map(str, args.train_sizes))}")
    print(f"Dev examples:     {len(dev)}")
    print(f"Heldout examples: {len(heldout)}")
    print(f"Manifest:         {manifest_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
