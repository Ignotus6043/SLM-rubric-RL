#!/usr/bin/env python3
"""
Offline evaluation for a model on the rubric dataset.

Generates one response per example with a local vLLM model, then scores the
responses with the same rubric reward function used during RL training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rubric_rl.env_utils import load_dotenv_if_present
from rubric_rl.reward_function import compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dotenv-path", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-usage-log", default="")
    parser.add_argument("--judge-max-concurrent", type=int, default=0)
    parser.add_argument("--judge-timeout", type=int, default=0)
    parser.add_argument("--judge-max-tokens", type=int, default=0)
    parser.add_argument("--aggregation-strategy", choices=("explicit", "implicit"), default="")
    parser.add_argument("--explicit-protocol", choices=("joint", "per_criterion"), default="")
    parser.add_argument("--strict-final-parse", action="store_true")
    parser.add_argument("--score-test-split", action="store_true")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen3 thinking in candidate answer generation when the tokenizer supports it.",
    )
    parser.add_argument("--score-batch-size", type=int, default=8)
    return parser.parse_args()


def configure_judge_environment(args: argparse.Namespace) -> None:
    loaded = load_dotenv_if_present(args.dotenv_path or None)
    if loaded is not None:
        print(f"Loaded dotenv: {loaded}")

    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI-KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI-KEY"]
    if not os.environ.get("OPENAI_ORG_ID") and os.environ.get("OPENAI-ORG-ID"):
        os.environ["OPENAI_ORG_ID"] = os.environ["OPENAI-ORG-ID"]

    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model
        os.environ.pop("JUDGE_API_BASE", None)
        if args.judge_model.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit(
                    "OPENAI_API_KEY is not set. Put it in your shell or pass --dotenv-path to a .env containing it."
                )
            os.environ["JUDGE_API_KEY"] = os.environ["OPENAI_API_KEY"]

    if args.judge_usage_log:
        os.environ["JUDGE_USAGE_LOG"] = args.judge_usage_log
    if args.judge_max_concurrent > 0:
        os.environ["JUDGE_MAX_CONCURRENT"] = str(args.judge_max_concurrent)
    if args.judge_timeout > 0:
        os.environ["JUDGE_TIMEOUT"] = str(args.judge_timeout)
    if args.judge_max_tokens > 0:
        os.environ["JUDGE_MAX_TOKENS"] = str(args.judge_max_tokens)
    if args.aggregation_strategy:
        os.environ["AGGREGATION_STRATEGY"] = args.aggregation_strategy
    if args.explicit_protocol:
        os.environ["RUBRIC_EXPLICIT_PROTOCOL"] = args.explicit_protocol
    os.environ["RUBRIC_STRICT_FINAL_PARSE"] = "true" if args.strict_final_parse else "false"
    os.environ["RUBRIC_SCORE_TEST_SPLIT"] = "true" if args.score_test_split else "false"


def load_rows(dataset_path: str, max_samples: int, seed: int) -> list[dict[str, Any]]:
    table = pq.read_table(dataset_path)
    rows = table.to_pylist()
    if max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        rows = [rows[i] for i in indices[:max_samples]]
    return rows


def build_prompt_text(tokenizer, prompt_field: Any, *, enable_thinking: bool = True) -> str:
    if isinstance(prompt_field, str):
        return prompt_field
    if isinstance(prompt_field, list):
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if not enable_thinking:
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(prompt_field, **kwargs)
        except TypeError:
            if "enable_thinking" not in kwargs:
                raise
            kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(prompt_field, **kwargs)
    raise TypeError(f"Unsupported prompt type: {type(prompt_field).__name__}")


def generate_responses(
    model_path: str,
    rows: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_response_length: int,
    gpu_memory_utilization: float,
    enable_thinking: bool = True,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [build_prompt_text(tokenizer, row["prompt"], enable_thinking=enable_thinking) for row in rows]

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_response_length,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    return [out.outputs[0].text for out in outputs]


async def score_rows(
    rows: list[dict[str, Any]],
    responses: list[str],
    score_batch_size: int,
) -> list[dict[str, Any] | float]:
    results: list[dict[str, Any] | float] = []
    for start in range(0, len(rows), score_batch_size):
        batch_rows = rows[start : start + score_batch_size]
        batch_responses = responses[start : start + score_batch_size]
        tasks = [
            compute_score(
                data_source=row.get("data_source", "rubric_rl"),
                solution_str=response,
                ground_truth=row.get("reward_model", {}).get("ground_truth", ""),
                extra_info=row.get("extra_info", {}),
            )
            for row, response in zip(batch_rows, batch_responses, strict=True)
        ]
        results.extend(await asyncio.gather(*tasks))
    return results


def normalize_score(result: dict[str, Any] | float) -> float:
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


def parse_failure_count(result: dict[str, Any] | float) -> int:
    if isinstance(result, dict):
        return int(result.get("num_parse_failures", 0))
    return 0


def total_judged_items(result: dict[str, Any] | float) -> int:
    if isinstance(result, dict):
        criterion_scores = result.get("criterion_scores")
        if isinstance(criterion_scores, list):
            return len(criterion_scores)
    return 1


def main() -> int:
    args = parse_args()
    configure_judge_environment(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset_path, args.max_samples, args.seed)
    responses = generate_responses(
        model_path=args.model_path,
        rows=rows,
        temperature=args.temperature,
        top_p=args.top_p,
        max_response_length=args.max_response_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=not args.disable_thinking,
    )
    reward_results = asyncio.run(score_rows(rows, responses, args.score_batch_size))

    per_example_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    scores = [normalize_score(result) for result in reward_results]
    parse_failures = [parse_failure_count(result) for result in reward_results]
    judged_items = [total_judged_items(result) for result in reward_results]
    total_parse_failures = sum(parse_failures)
    total_judged_criteria = sum(judged_items)
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(args.model_path.rstrip("/")),
        "dataset_path": args.dataset_path,
        "judge_model": os.environ.get("JUDGE_MODEL", ""),
        "num_examples": len(rows),
        "aggregation_strategy": os.environ.get("AGGREGATION_STRATEGY", "explicit"),
        "explicit_protocol": os.environ.get("RUBRIC_EXPLICIT_PROTOCOL", "joint"),
        "strict_final_parse": os.environ.get("RUBRIC_STRICT_FINAL_PARSE", "false"),
        "score_test_split": os.environ.get("RUBRIC_SCORE_TEST_SPLIT", "false"),
        "judge_usage_log": os.environ.get("JUDGE_USAGE_LOG", ""),
        "mean_score": mean(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "total_parse_failures": total_parse_failures,
        "parse_success_rate": (
            1.0 - (total_parse_failures / total_judged_criteria)
            if total_judged_criteria
            else 1.0
        ),
        "mean_response_chars": mean(len(resp) for resp in responses) if responses else 0.0,
        "mean_response_tokens_approx": mean(len(resp.split()) * 1.3 for resp in responses) if responses else 0.0,
    }

    with per_example_path.open("w", encoding="utf-8") as f:
        for idx, (row, response, reward_result, score) in enumerate(
            zip(rows, responses, reward_results, scores, strict=True)
        ):
            extra_info = row.get("extra_info", {})
            rubric = extra_info.get("rubric", [])
            record = {
                "index": idx,
                "data_source": row.get("data_source", "rubric_rl"),
                "question": extra_info.get("question", ""),
                "response": response,
                "score": score,
                "num_parse_failures": parse_failure_count(reward_result),
                "rubric_titles": [item.get("title", "") for item in rubric],
                "reward_result": reward_result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=======================================")
    print(f"Model:        {summary['model_name']}")
    print(f"Examples:     {summary['num_examples']}")
    print(f"Mean score:   {summary['mean_score']:.4f}")
    print(f"Min / Max:    {summary['min_score']:.4f} / {summary['max_score']:.4f}")
    print(f"Parse succ.:  {summary['parse_success_rate']:.4f}")
    print(f"Predictions:  {per_example_path}")
    print(f"Summary:      {summary_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
