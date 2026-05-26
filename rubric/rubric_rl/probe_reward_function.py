"""verl reward function backed by a frozen representation-probe server."""

from __future__ import annotations

import os
import time
from typing import Any

import aiohttp


def _probe_reward_api_base() -> str:
    return os.environ.get("PROBE_REWARD_API_BASE", "http://127.0.0.1:8621").rstrip("/")


def _timeout_seconds() -> float:
    try:
        return float(os.environ.get("PROBE_REWARD_TIMEOUT", "600"))
    except ValueError:
        return 600.0


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, Any] | float:
    start_time = time.perf_counter()
    question = extra_info.get("question", "")
    rubric = extra_info.get("rubric", [])
    split = extra_info.get("split", "train")

    if split == "test":
        return 0.0
    if not rubric:
        return {
            "score": 0.0,
            "criterion_scores": [],
            "criterion_weights": [],
            "criterion_parse_ok": [],
            "num_parse_failures": 0,
            "parse_source": "probe_reward_empty_rubric",
            "raw_parse_success": True,
            "fallback_used": False,
            "judge_time_seconds": 0.0,
        }

    payload = {
        "data_source": data_source,
        "question": question,
        "rubric": rubric,
        "response": solution_str,
        "ground_truth": ground_truth,
    }
    timeout = aiohttp.ClientTimeout(total=_timeout_seconds())
    url = f"{_probe_reward_api_base()}/score"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        print(f"Error calling probe reward server: {exc}")
        return {
            "score": 0.0,
            "criterion_scores": [0.0] * len(rubric),
            "criterion_weights": [
                abs(float(item.get("weight", 1.0) if isinstance(item, dict) and item.get("weight", 1.0) is not None else 1.0))
                for item in rubric
            ],
            "criterion_parse_ok": [False] * len(rubric),
            "num_parse_failures": len(rubric),
            "raw_num_parse_failures": len(rubric),
            "parse_source": "probe_reward_server_error",
            "raw_parse_success": False,
            "fallback_used": False,
            "probe_reward_error": repr(exc),
            "judge_time_seconds": float(elapsed),
        }

    if not data.get("ok"):
        elapsed = time.perf_counter() - start_time
        error = data.get("error", "unknown probe reward server error")
        print(f"Probe reward server returned error: {error}")
        return {
            "score": 0.0,
            "criterion_scores": [0.0] * len(rubric),
            "criterion_weights": [
                abs(float(item.get("weight", 1.0) if isinstance(item, dict) and item.get("weight", 1.0) is not None else 1.0))
                for item in rubric
            ],
            "criterion_parse_ok": [False] * len(rubric),
            "num_parse_failures": len(rubric),
            "raw_num_parse_failures": len(rubric),
            "parse_source": "probe_reward_server_error",
            "raw_parse_success": False,
            "fallback_used": False,
            "probe_reward_error": str(error),
            "judge_time_seconds": float(elapsed),
        }

    result = data.get("result")
    if not isinstance(result, dict):
        return 0.0
    result["judge_time_seconds"] = float(time.perf_counter() - start_time)
    return result
