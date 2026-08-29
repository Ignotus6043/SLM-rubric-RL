#!/usr/bin/env python3
"""Fail-fast reward-function preflight for rubric RL jobs."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--reward-func-path", required=True)
    parser.add_argument("--reward-func-name", default="compute_score")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--response-mode", choices=("ground_truth", "short"), default="ground_truth")
    parser.add_argument("--allow-parse-failures", action="store_true")
    return parser.parse_args()


def load_reward_function(repo_dir: Path, reward_func_path: str, reward_func_name: str):
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "verl"))

    path = Path(reward_func_path)
    if not path.is_absolute():
        path = repo_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"reward function file not found: {path}")

    spec = importlib.util.spec_from_file_location("rubric_reward_preflight_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import reward function module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    fn = getattr(module, reward_func_name, None)
    if fn is None:
        raise AttributeError(f"{path} has no function named {reward_func_name!r}")
    return fn


def load_rows(data_file: Path, num_samples: int) -> list[dict[str, Any]]:
    rows = pq.read_table(data_file).to_pylist()
    selected: list[dict[str, Any]] = []
    for row in rows:
        extra_info = row.get("extra_info") or {}
        rubric = extra_info.get("rubric") or []
        if rubric:
            selected.append(row)
        if len(selected) >= num_samples:
            break
    if not selected:
        raise ValueError(f"no rows with non-empty extra_info.rubric in {data_file}")
    return selected


def response_for(row: dict[str, Any], response_mode: str) -> tuple[str, str]:
    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}
    ground_truth = str(reward_model.get("ground_truth") or extra_info.get("reference_answer") or "")
    if response_mode == "short" or not ground_truth:
        return "This is a concise candidate answer for reward preflight.", ground_truth
    return ground_truth, ground_truth


def validate_reward_result(
    result: Any,
    expected_criteria: int,
    sample_idx: int,
    *,
    allow_parse_failures: bool,
) -> dict[str, Any]:
    if isinstance(result, (int, float)):
        score = float(result)
        if not math.isfinite(score):
            raise ValueError(f"sample {sample_idx}: scalar reward is not finite: {result}")
        return {"score": score}

    if not isinstance(result, dict):
        raise TypeError(f"sample {sample_idx}: reward must be float or dict, got {type(result).__name__}")

    if "score" not in result:
        raise KeyError(f"sample {sample_idx}: dict reward missing 'score'")
    score = float(result["score"])
    if not math.isfinite(score):
        raise ValueError(f"sample {sample_idx}: dict reward score is not finite: {result['score']!r}")

    criterion_scores = result.get("criterion_scores")
    criterion_weights = result.get("criterion_weights")
    criterion_parse_ok = result.get("criterion_parse_ok")
    for key, value in (
        ("criterion_scores", criterion_scores),
        ("criterion_weights", criterion_weights),
        ("criterion_parse_ok", criterion_parse_ok),
    ):
        if value is None:
            continue
        if not isinstance(value, list):
            raise TypeError(f"sample {sample_idx}: {key} must be a list, got {type(value).__name__}")
        if len(value) != expected_criteria:
            raise ValueError(
                f"sample {sample_idx}: {key} length {len(value)} != rubric length {expected_criteria}"
            )

    for key in ("criterion_scores", "criterion_weights"):
        value = result.get(key)
        if value is None:
            continue
        for item_idx, item in enumerate(value):
            item_float = float(item)
            if not math.isfinite(item_float):
                raise ValueError(f"sample {sample_idx}: {key}[{item_idx}] is not finite: {item!r}")

    if not allow_parse_failures:
        num_parse_failures = int(result.get("num_parse_failures") or 0)
        raw_num_parse_failures = int(result.get("raw_num_parse_failures") or 0)
        if num_parse_failures or raw_num_parse_failures:
            raise ValueError(
                f"sample {sample_idx}: reward parse failed "
                f"(num_parse_failures={num_parse_failures}, raw_num_parse_failures={raw_num_parse_failures})"
            )
        if criterion_parse_ok is not None and not all(bool(value) for value in criterion_parse_ok):
            raise ValueError(f"sample {sample_idx}: not all criterion_parse_ok values are true")

    return result


def run_metric_aggregation_check(repo_dir: Path, results: list[dict[str, Any]]) -> None:
    sys.path.insert(0, str(repo_dir / "verl"))
    from verl.trainer.ppo.metric_utils import process_validation_metrics

    infos: dict[str, list[Any]] = {"reward": [float(result.get("score", 0.0)) for result in results]}
    all_keys = set().union(*(result.keys() for result in results))
    for key in all_keys:
        if key == "score":
            continue
        infos[key] = [result.get(key) for result in results]

    metrics = process_validation_metrics(
        data_sources=["rubric_rl"] * len(results),
        sample_uids=[f"preflight_{idx}" for idx in range(len(results))],
        infos_dict=infos,
    )
    if "rubric_rl" not in metrics or "reward" not in metrics["rubric_rl"]:
        raise RuntimeError(f"validation metric preflight did not produce reward metrics: {metrics}")


async def main_async() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    reward_fn = load_reward_function(repo_dir, args.reward_func_path, args.reward_func_name)
    rows = load_rows(Path(args.data_file), args.num_samples)

    validated_results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        rubric = extra_info.get("rubric") or []
        extra_info = dict(extra_info)
        if extra_info.get("split") == "test":
            extra_info["split"] = "val"
        response, ground_truth = response_for(row, args.response_mode)
        result = await reward_fn(
            data_source=str(row.get("data_source") or "rubric_rl"),
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        validated = validate_reward_result(
            result,
            len(rubric),
            idx,
            allow_parse_failures=args.allow_parse_failures,
        )
        if not isinstance(validated, dict):
            validated = {"score": float(validated)}
        validated_results.append(validated)

    run_metric_aggregation_check(repo_dir, validated_results)
    print(
        json.dumps(
            {
                "ok": True,
                "samples_checked": len(validated_results),
                "scores": [float(result.get("score", 0.0)) for result in validated_results],
                "reward_func_path": args.reward_func_path,
                "reward_func_name": args.reward_func_name,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
