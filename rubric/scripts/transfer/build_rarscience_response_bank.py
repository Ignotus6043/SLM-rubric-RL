#!/usr/bin/env python3
"""Build a fixed RaR-Science response bank for transfer and external evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rubric_rl.evaluate_model import generate_responses
from rubric_rl.rar_science_utils import load_parquet_rows, slugify_model_name, stable_question_id


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_path: str
    model_name: str
    temperature: float
    top_p: float
    max_response_length: int
    generation_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows in the split.")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Only include the original dataset reference answer as a candidate.",
    )
    parser.add_argument(
        "--only-base-model",
        action="store_true",
        help="Only generate the base model candidate, plus the reference answer.",
    )
    parser.add_argument("--base-model-path", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--base-model-name", default="Qwen3-4B-Base")
    parser.add_argument("--tiny-model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tiny-model-name", default="Qwen3-0.6B")
    parser.add_argument("--small-model-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--small-model-name", default="Qwen3-1.7B")
    parser.add_argument("--fourb-model-path", default="Qwen/Qwen3-4B")
    parser.add_argument("--fourb-model-name", default="Qwen3-4B")
    parser.add_argument("--eightb-model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--eightb-model-name", default="Qwen3-8B")
    parser.add_argument("--rl-model-path", default="")
    parser.add_argument("--rl-model-name", default="rl_checkpoint")
    parser.add_argument("--include-weak-sampled-base", action="store_true")
    parser.add_argument("--sampled-temperature", type=float, default=0.8)
    parser.add_argument("--sampled-top-p", type=float, default=0.95)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--disable-generator-thinking",
        action="store_true",
        help="Disable Qwen3 thinking for candidate answer generation only. This does not affect judge reasoning.",
    )
    parser.add_argument(
        "--fail-on-unclosed-thinking",
        action="store_true",
        help="Exit nonzero if generated candidate responses contain unclosed <think> blocks.",
    )
    return parser.parse_args()


def build_candidate_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    if args.reference_only:
        return []

    base_spec = CandidateSpec(
        candidate_id=f"{slugify_model_name(args.base_model_name)}_greedy",
        model_path=args.base_model_path,
        model_name=args.base_model_name,
        temperature=0.0,
        top_p=1.0,
        max_response_length=args.max_response_length,
        generation_mode="greedy",
    )
    if args.only_base_model:
        specs = [base_spec]
    else:
        specs = [
            base_spec,
            CandidateSpec(
                candidate_id=f"{slugify_model_name(args.tiny_model_name)}_greedy",
                model_path=args.tiny_model_path,
                model_name=args.tiny_model_name,
                temperature=0.0,
                top_p=1.0,
                max_response_length=args.max_response_length,
                generation_mode="greedy",
            ),
            CandidateSpec(
                candidate_id=f"{slugify_model_name(args.small_model_name)}_greedy",
                model_path=args.small_model_path,
                model_name=args.small_model_name,
                temperature=0.0,
                top_p=1.0,
                max_response_length=args.max_response_length,
                generation_mode="greedy",
            ),
            CandidateSpec(
                candidate_id=f"{slugify_model_name(args.fourb_model_name)}_greedy",
                model_path=args.fourb_model_path,
                model_name=args.fourb_model_name,
                temperature=0.0,
                top_p=1.0,
                max_response_length=args.max_response_length,
                generation_mode="greedy",
            ),
            CandidateSpec(
                candidate_id=f"{slugify_model_name(args.eightb_model_name)}_greedy",
                model_path=args.eightb_model_path,
                model_name=args.eightb_model_name,
                temperature=0.0,
                top_p=1.0,
                max_response_length=args.max_response_length,
                generation_mode="greedy",
            ),
        ]
    if args.rl_model_path:
        specs.append(
            CandidateSpec(
                candidate_id=f"{slugify_model_name(args.rl_model_name)}_greedy",
                model_path=args.rl_model_path,
                model_name=args.rl_model_name,
                temperature=0.0,
                top_p=1.0,
                max_response_length=args.max_response_length,
                generation_mode="greedy",
            )
        )
    if args.include_weak_sampled_base:
        specs.append(
            CandidateSpec(
                candidate_id=f"{slugify_model_name(args.base_model_name)}_sampled",
                model_path=args.base_model_path,
                model_name=args.base_model_name,
                temperature=args.sampled_temperature,
                top_p=args.sampled_top_p,
                max_response_length=args.max_response_length,
                generation_mode="sampled",
            )
        )
    return specs


def summarize_candidates(questions: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for question in questions:
        for candidate in question["candidates"]:
            candidate_id = candidate["candidate_id"]
            response = candidate.get("response", "") or ""
            item = stats.setdefault(
                candidate_id,
                {
                    "num_examples": 0,
                    "mean_response_chars": 0.0,
                    "max_response_chars": 0,
                    "think_open_count": 0,
                    "think_close_count": 0,
                    "unclosed_thinking_count": 0,
                },
            )
            response_chars = len(response)
            item["num_examples"] += 1
            item["mean_response_chars"] += response_chars
            item["max_response_chars"] = max(item["max_response_chars"], response_chars)
            has_open = "<think>" in response
            has_close = "</think>" in response
            item["think_open_count"] += int(has_open)
            item["think_close_count"] += int(has_close)
            item["unclosed_thinking_count"] += int(has_open and not has_close)

    for item in stats.values():
        if item["num_examples"]:
            item["mean_response_chars"] /= item["num_examples"]
    return stats


def main() -> int:
    args = parse_args()
    rows = load_parquet_rows(args.dataset_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    candidate_specs = build_candidate_specs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "response_bank.jsonl"
    summary_path = output_dir / "response_bank_summary.json"

    questions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        questions.append(
            {
                "question_id": row.get("question_id") or stable_question_id(row, index),
                "split_role": row.get("split_role") or extra_info.get("split", "unknown"),
                "data_source": row.get("data_source", "rubric_rl"),
                "question": extra_info.get("question", ""),
                "reference_answer": extra_info.get("reference_answer", ""),
                "rubric": extra_info.get("rubric", []),
                "prompt": row.get("prompt"),
                "candidates": [
                    {
                        "candidate_id": "reference_answer",
                        "model_name": "reference_answer",
                        "model_path": "",
                        "generation_mode": "reference",
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "response": extra_info.get("reference_answer", ""),
                    }
                ],
            }
        )

    for spec in candidate_specs:
        responses = generate_responses(
            model_path=spec.model_path,
            rows=rows,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_response_length=spec.max_response_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_thinking=not args.disable_generator_thinking,
        )
        for question, response in zip(questions, responses, strict=True):
            question["candidates"].append(
                {
                    "candidate_id": spec.candidate_id,
                    "model_name": spec.model_name,
                    "model_path": spec.model_path,
                    "generation_mode": spec.generation_mode,
                    "temperature": spec.temperature,
                    "top_p": spec.top_p,
                    "response": response,
                }
            )

    with bank_path.open("w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")

    per_candidate_summary = summarize_candidates(questions)
    summary = {
        "dataset_path": args.dataset_path,
        "num_questions": len(questions),
        "split_roles": sorted({question["split_role"] for question in questions}),
        "reference_only": bool(args.reference_only),
        "generator_thinking_disabled": bool(args.disable_generator_thinking),
        "candidate_ids": [candidate["candidate_id"] for candidate in questions[0]["candidates"]] if questions else [],
        "num_candidates_per_question": len(questions[0]["candidates"]) if questions else 0,
        "mean_reference_answer_chars": mean(
            len(question["reference_answer"]) for question in questions
        ) if questions else 0.0,
        "per_candidate_summary": per_candidate_summary,
        "output_path": str(bank_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    unclosed_total = sum(item["unclosed_thinking_count"] for item in per_candidate_summary.values())

    print("=======================================")
    print(f"Dataset:                {args.dataset_path}")
    print(f"Questions:              {len(questions)}")
    print(f"Candidates/question:    {summary['num_candidates_per_question']}")
    print(f"Candidate IDs:          {', '.join(summary['candidate_ids'])}")
    print(f"Generator thinking off: {summary['generator_thinking_disabled']}")
    print(f"Unclosed think blocks:  {unclosed_total}")
    print(f"Response bank:          {bank_path}")
    print(f"Summary:                {summary_path}")
    print("=======================================")
    if args.fail_on_unclosed_thinking and unclosed_total:
        print("ERROR: generated bank contains unclosed <think> blocks.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
