"""
Rubric-based reward function for verl.

Supports two scoring strategies:
- Explicit: Score one response against all rubric criteria in a single call,
  then aggregate by weight
- Implicit: Pass all rubric items to judge in one call, get single Likert score (1-10)

Environment variables:
    AGGREGATION_STRATEGY: "explicit" or "implicit" (default: "explicit")
    RUBRIC_EXPLICIT_PROTOCOL: "joint" or "per_criterion" (default: "joint")
    RUBRIC_STRICT_FINAL_PARSE: when true, only accept the required boxed final
        answer format instead of falling back to loose number extraction.
"""

import os
import re
import asyncio
import time
from typing import List, Dict, Any, Optional

from rubric_rl.judge_client import call_judge, call_judge_batch
from rubric_rl.prompts import build_explicit_joint_prompt, build_explicit_prompt, build_implicit_prompt


# Default strategy
AGGREGATION_STRATEGY = os.environ.get("AGGREGATION_STRATEGY", "explicit")


def _explicit_protocol() -> str:
    value = os.environ.get("RUBRIC_EXPLICIT_PROTOCOL", "joint").strip().lower()
    if value in {"per_criterion", "per-criterion", "legacy", "independent"}:
        return "per_criterion"
    return "joint"


def _strict_final_parse_enabled() -> bool:
    value = os.environ.get("RUBRIC_STRICT_FINAL_PARSE", "false")
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("RUBRIC_PARSE_RETRY_ATTEMPTS", "1")))
    except ValueError:
        return 1


def _hard_fallback_enabled() -> bool:
    value = os.environ.get("RUBRIC_PARSE_HARD_FALLBACK", "false")
    return value.lower() in {"1", "true", "yes", "on"}


def _score_test_split_enabled() -> bool:
    value = os.environ.get("RUBRIC_SCORE_TEST_SPLIT", "false")
    return value.lower() in {"1", "true", "yes", "on"}


def _fallback_positive_threshold() -> float:
    try:
        return float(os.environ.get("RUBRIC_PARSE_FALLBACK_POSITIVE_THRESHOLD", "5"))
    except ValueError:
        return 5.0


def _criterion_importance_score(criterion: Dict[str, Any]) -> float:
    value = criterion.get("score", criterion.get("weight", 0.0))
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _criterion_effective_weight(criterion: Dict[str, Any]) -> float:
    weight = criterion.get("weight", 1.0)
    if weight is None:
        weight = 1.0
    try:
        value = float(weight)
    except (TypeError, ValueError):
        value = 1.0
    return abs(value)


def _fallback_verdicts_from_rubric(rubric: List[Dict[str, Any]]) -> List[int]:
    threshold = _fallback_positive_threshold()
    return [1 if _criterion_importance_score(item) >= threshold else 0 for item in rubric]


def extract_binary_score(response: str, *, strict: Optional[bool] = None) -> Optional[int]:
    """
    Extract binary score (0 or 1) from explicit strategy judge response.

    Expects format: \boxed{0} or \boxed{1}

    Args:
        response: Judge response text

    Returns:
        0 or 1, or None if extraction fails
    """
    if not response:
        return None

    if strict is None:
        strict = _strict_final_parse_enabled()

    # Look for \boxed{...} pattern
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        score_str = match.group(1).strip()
        try:
            score = int(score_str)
            if score in (0, 1):
                return score
        except ValueError:
            pass

    if strict:
        return None

    # Try to find 0 or 1 in the response
    if re.search(r"\b0\b", response):
        return 0
    if re.search(r"\b1\b", response):
        return 1

    return None


def _normalize_yes_no_token(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token in {"yes", "y", "true", "1"}:
        return 1
    if token in {"no", "n", "false", "0"}:
        return 0
    return None


def parse_explicit_verdict_vector(
    text: str,
    num_rules: int,
    *,
    strict: Optional[bool] = None,
) -> Optional[List[int]]:
    """
    Strict parser for the explicit joint protocol:
      Rule i: <short justification>. Verdict: Yes/No

    This mirrors the Bench parser so that static and RL evaluation share the
    same standardized output format.
    """
    if not text:
        return None

    if strict is None:
        strict = _strict_final_parse_enabled()

    verdict_by_rule: Dict[int, int] = {}
    strict_pattern = re.compile(
        r"^\s*Rule\s*(\d+)\s*:\s*.*?\bVerdict\s*:\s*(Yes|No|Y|N)\s*$",
        flags=re.IGNORECASE,
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = strict_pattern.match(line)
        if not match:
            continue
        rule_id = int(match.group(1))
        verdict = _normalize_yes_no_token(match.group(2))
        if verdict is None:
            continue
        verdict_by_rule[rule_id] = verdict

    if len(verdict_by_rule) != num_rules:
        verdict_by_rule = {}
    elif set(verdict_by_rule.keys()) == set(range(1, num_rules + 1)):
        return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]

    if strict:
        return None

    relaxed_patterns = [
        re.compile(
            r"^\s*(?:[-*]\s*)?Rule\s*(\d+)\s*[:.)-]?\s*.*?\bVerdict\s*[:=-]?\s*(Yes|No|Y|N|True|False|0|1)\b.*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*]\s*)?Rule\s*(\d+)\s*[:.)-]?\s*.*?\b(Yes|No|Y|N|True|False)\b\s*$",
            flags=re.IGNORECASE,
        ),
    ]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for pattern in relaxed_patterns:
            match = pattern.match(line)
            if not match:
                continue
            rule_id = int(match.group(1))
            verdict = _normalize_yes_no_token(match.group(2))
            if verdict is not None:
                verdict_by_rule[rule_id] = verdict
                break

    if set(verdict_by_rule.keys()) == set(range(1, num_rules + 1)):
        return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]

    # Qwen-style local judges often drift into concise enumerations such as
    # "1. Yes", "Rule 1 - No", or JSON-ish `"Rule 1": "Yes"` despite the
    # requested format. Accept those in relaxed mode while still requiring a
    # complete one-verdict-per-rule vector.
    broad_patterns = [
        re.compile(
            r"\bRule\s*(\d+)\b.*?\b(?:Verdict|Answer|Satisfied)\s*[:=-]?\s*(Yes|No|Y|N|True|False|0|1)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*]\s*)?(?:Rule\s*)?(\d+)\s*[:.)\]-]\s*(Yes|No|Y|N|True|False|0|1)\b\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:[-*]\s*)?Rule\s*(\d+)\s*[:.)\]-]\s*.*?\b(Yes|No|Y|N|True|False|0|1)\b\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"[\"']?(?:Rule\s*)?(\d+)[\"']?\s*[:=]\s*[\"']?(Yes|No|Y|N|True|False|0|1)[\"']?",
            flags=re.IGNORECASE,
        ),
    ]

    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        for pattern in broad_patterns:
            match = pattern.search(line)
            if not match:
                continue
            rule_id = int(match.group(1))
            verdict = _normalize_yes_no_token(match.group(2))
            if verdict is not None:
                verdict_by_rule[rule_id] = verdict
                break

    if set(verdict_by_rule.keys()) == set(range(1, num_rules + 1)):
        return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]

    return None


def extract_likert_score(response: str, *, strict: Optional[bool] = None) -> Optional[float]:
    """
    Extract Likert score (1-10) from implicit strategy judge response.

    Expects format: \boxed{score} where score is between 1 and 10.

    Args:
        response: Judge response text

    Returns:
        Score between 0 and 1 (normalized from 1-10), or None if extraction fails
    """
    if not response:
        return None

    if strict is None:
        strict = _strict_final_parse_enabled()

    # Look for \boxed{...} pattern
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        score_str = match.group(1).strip()
        try:
            score = float(score_str)
            # Clamp to [1, 10] range
            score = max(1.0, min(10.0, score))
            # Normalize to [0, 1]
            return (score - 1.0) / 9.0
        except ValueError:
            pass

    if strict:
        return None

    # Try to find any number 1-10 in the response
    numbers = re.findall(r"\d+(?:\.\d+)?", response)
    for num_str in numbers:
        try:
            score = float(num_str)
            if 1.0 <= score <= 10.0:
                return (score - 1.0) / 9.0
        except ValueError:
            pass

    return None


def _aggregate_explicit_verdicts(
    rubric: List[Dict[str, Any]],
    verdicts: List[int],
    *,
    raw_judge_response: str,
    raw_judge_responses: List[str],
    judge_attempts: int,
    judge_time_seconds: float,
    parse_source: str,
    fallback_used: bool,
    raw_parse_success: bool,
) -> Dict[str, Any]:
    weighted_sum = 0.0
    total_positive_weight = 0.0
    criterion_scores = []
    criterion_weights = []
    criterion_parse_ok = []

    for criterion, verdict in zip(rubric, verdicts, strict=True):
        weight = _criterion_effective_weight(criterion)
        criterion_parse_ok.append(True)
        criterion_scores.append(float(verdict))
        criterion_weights.append(float(weight))
        weighted_sum += verdict * weight
        if weight > 0:
            total_positive_weight += weight

    # Compute weighted score using only positive weights as denominator.
    # Negative-weight criteria are penalty terms that can subtract from the score.
    # Scores can be negative when penalties dominate — intentional for RL signal.
    if total_positive_weight > 0:
        final_score = weighted_sum / total_positive_weight
    else:
        final_score = 0.0

    return {
        "score": final_score,
        "criterion_scores": criterion_scores,
        "criterion_weights": criterion_weights,
        "criterion_parse_ok": criterion_parse_ok,
        "num_parse_failures": 0,
        "raw_judge_response": raw_judge_response,
        "raw_judge_responses": raw_judge_responses,
        "judge_attempts": judge_attempts,
        "judge_time_seconds": float(judge_time_seconds),
        "parse_source": parse_source,
        "fallback_used": fallback_used,
        "raw_parse_success": raw_parse_success,
        "raw_num_parse_failures": 0 if raw_parse_success else len(rubric),
    }


async def compute_explicit_score(
    question: str,
    rubric: List[Dict[str, Any]],
    response: str,
) -> Dict[str, Any]:
    """
    Compute score using explicit strategy.

    Call the judge once for the full rubric, parse a strict verdict vector, and
    aggregate with the rubric weights.

    Args:
        question: Original question
        rubric: List of rubric criteria (each should have: title, description, weight)
        response: AI response to evaluate

    Returns:
        Dict with:
            - score: Aggregated score between 0 and 1 (weighted average)
            - criterion_scores: List of per-criterion binary scores (0 or 1)
            - criterion_weights: List of per-criterion weights
    """
    prompt = build_explicit_joint_prompt(
        question=question,
        rubric=rubric,
        response=response,
    )
    raw_judge_responses: List[str] = []
    max_attempts = _parse_retry_attempts()
    judge_time_seconds = 0.0

    for attempt_idx in range(1, max_attempts + 1):
        start_time = time.perf_counter()
        judge_response = await call_judge(prompt)
        judge_time_seconds += time.perf_counter() - start_time
        raw_judge_responses.append(judge_response)
        verdicts = parse_explicit_verdict_vector(
            judge_response,
            len(rubric),
            strict=_strict_final_parse_enabled(),
        )
        if verdicts is not None:
            return _aggregate_explicit_verdicts(
                rubric,
                verdicts,
                raw_judge_response=judge_response,
                raw_judge_responses=raw_judge_responses,
                judge_attempts=attempt_idx,
                judge_time_seconds=judge_time_seconds,
                parse_source="judge",
                fallback_used=False,
                raw_parse_success=True,
            )

    last_response = raw_judge_responses[-1] if raw_judge_responses else ""
    if _hard_fallback_enabled():
        fallback_verdicts = _fallback_verdicts_from_rubric(rubric)
        return _aggregate_explicit_verdicts(
            rubric,
            fallback_verdicts,
            raw_judge_response=last_response,
            raw_judge_responses=raw_judge_responses,
            judge_attempts=max_attempts,
            judge_time_seconds=judge_time_seconds,
            parse_source="rubric_weight_fallback",
            fallback_used=True,
            raw_parse_success=False,
        )

    criterion_scores = [0.0] * len(rubric)
    criterion_weights = [
        _criterion_effective_weight(item)
        for item in rubric
    ]
    criterion_parse_ok = [False] * len(rubric)
    num_parse_failures = len(rubric)
    return {
        "score": 0.0,
        "criterion_scores": criterion_scores,
        "criterion_weights": criterion_weights,
        "criterion_parse_ok": criterion_parse_ok,
        "num_parse_failures": num_parse_failures,
        "raw_judge_response": last_response,
        "raw_judge_responses": raw_judge_responses,
        "judge_attempts": max_attempts,
        "judge_time_seconds": float(judge_time_seconds),
        "parse_source": "parse_failed",
        "fallback_used": False,
        "raw_parse_success": False,
        "raw_num_parse_failures": len(rubric),
    }


async def compute_explicit_per_criterion_score(
    question: str,
    rubric: List[Dict[str, Any]],
    response: str,
) -> Dict[str, Any]:
    """
    Legacy explicit protocol: one boxed 0/1 judge call per rubric criterion.

    This matches the older single-GPU 8B judge RL runs and is intentionally
    kept as an option for baseline comparability.
    """
    prompts = [
        build_explicit_prompt(
            criterion=criterion,
            question=question,
            response=response,
        )
        for criterion in rubric
    ]

    start_time = time.perf_counter()
    judge_responses = await call_judge_batch(prompts)
    judge_time_seconds = time.perf_counter() - start_time

    weighted_sum = 0.0
    total_positive_weight = 0.0
    criterion_scores = []
    criterion_weights = []
    criterion_parse_ok = []
    num_parse_failures = 0

    for criterion, judge_response in zip(rubric, judge_responses, strict=True):
        weight = _criterion_effective_weight(criterion)
        binary_score = extract_binary_score(
            judge_response,
            strict=_strict_final_parse_enabled(),
        )
        parse_ok = binary_score is not None
        if binary_score is None:
            binary_score = 0
            num_parse_failures += 1

        criterion_parse_ok.append(parse_ok)
        criterion_scores.append(float(binary_score))
        criterion_weights.append(float(weight))
        weighted_sum += binary_score * weight
        if weight > 0:
            total_positive_weight += weight

    if total_positive_weight > 0:
        final_score = weighted_sum / total_positive_weight
    else:
        final_score = 0.0

    raw_parse_success = num_parse_failures == 0
    return {
        "score": final_score,
        "criterion_scores": criterion_scores,
        "criterion_weights": criterion_weights,
        "criterion_parse_ok": criterion_parse_ok,
        "num_parse_failures": num_parse_failures,
        "raw_judge_response": "\n".join(judge_responses),
        "raw_judge_responses": list(judge_responses),
        "judge_attempts": 1,
        "judge_time_seconds": float(judge_time_seconds),
        "parse_source": "per_criterion_boxed",
        "fallback_used": False,
        "raw_parse_success": raw_parse_success,
        "raw_num_parse_failures": num_parse_failures,
    }


async def compute_implicit_score(
    question: str,
    rubric: List[Dict[str, Any]],
    response: str,
) -> float:
    """
    Compute score using implicit strategy.

    Pass all rubric items to judge in one call, get single Likert score (1-10).
    No weights - judge decides holistically.

    Args:
        question: Original question
        rubric: List of rubric criteria
        response: AI response to evaluate

    Returns:
        Score between 0 and 1 (normalized from 1-10)
    """
    # Build implicit prompt
    prompt = build_implicit_prompt(
        question=question,
        rubric=rubric,
        response=response,
    )

    # Call judge
    judge_response = await call_judge(prompt)

    # Extract Likert score (1-10) and normalize to 0-1
    score = extract_likert_score(judge_response)
    if score is None:
        return 0.0

    return score


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> Dict[str, Any] | float:
    """
    Main entry point for verl reward function.

    Args:
        data_source: Data source identifier (not used directly)
        solution_str: The AI response to evaluate
        ground_truth: Reference answer (not used directly, rubric is in extra_info)
        extra_info: Dict containing:
            - question: Original question
            - rubric: List of rubric criteria
            - split: "train", "val", or "test"

    Returns:
        For explicit strategy: Dict with "score", "criterion_scores", "criterion_weights"
        For implicit strategy: Float score between 0 and 1
    """
    try:
        question = extra_info.get("question", "")
        rubric = extra_info.get("rubric", [])
        split = extra_info.get("split", "train")

        # For RL, the test split should not provide a reward signal. Offline
        # eval launchers can opt in explicitly when scoring held-out test data.
        if split == "test" and not _score_test_split_enabled():
            return 0.0

        if not rubric:
            print("Warning: No rubric found in extra_info")
            return 0.0

        # Choose strategy
        strategy = os.environ.get("AGGREGATION_STRATEGY", AGGREGATION_STRATEGY)

        if strategy == "implicit":
            return await compute_implicit_score(
                question=question,
                rubric=rubric,
                response=solution_str,
            )
        else:
            if _explicit_protocol() == "per_criterion":
                return await compute_explicit_per_criterion_score(
                    question=question,
                    rubric=rubric,
                    response=solution_str,
                )

            # Default to explicit joint strategy - returns dict with per-criterion scores
            return await compute_explicit_score(
                question=question,
                rubric=rubric,
                response=solution_str,
            )

    except Exception as e:
        print(f"Error computing score: {e}")
        return 0.0


# For testing purposes
if __name__ == "__main__":
    async def test_explicit():
        """Test explicit scoring strategy."""
        question = "What is the capital of France?"
        rubric = [
            {
                "title": "Correct Answer",
                "description": "Essential Criteria: The response must correctly identify Paris as the capital of France.",
                "weight": 1.0,
            },
            {
                "title": "Explanation",
                "description": "Important Criteria: The response should provide a brief explanation.",
                "weight": 0.5,
            },
        ]
        response = "Paris is the capital of France."

        result = await compute_explicit_score(
            question=question,
            rubric=rubric,
            response=response,
        )
        print(f"Explicit result: {result}")
        print(f"  score: {result['score']}")
        print(f"  criterion_scores: {result['criterion_scores']}")
        print(f"  criterion_weights: {result['criterion_weights']}")

    async def test_implicit():
        """Test implicit scoring strategy."""
        question = "What is the capital of France?"
        rubric = [
            {
                "title": "Correct Answer",
                "description": "Essential Criteria: The response must correctly identify Paris as the capital of France.",
                "weight": 1.0,
            },
            {
                "title": "Explanation",
                "description": "Important Criteria: The response should provide a brief explanation.",
                "weight": 0.5,
            },
        ]
        response = "Paris is the capital of France."

        score = await compute_implicit_score(
            question=question,
            rubric=rubric,
            response=response,
        )
        print(f"Implicit score: {score}")

    async def test_reward_function():
        """Test the main compute_score function."""
        extra_info = {
            "question": "What is the capital of France?",
            "rubric": [
                {
                    "title": "Correct Answer",
                    "description": "Essential Criteria: The response must correctly identify Paris as the capital of France.",
                    "weight": 1.0,
                },
            ],
            "split": "train",
        }

        score = await compute_score(
            data_source="rubric_rl",
            solution_str="Paris is the capital of France.",
            ground_truth="Paris",
            extra_info=extra_info,
        )
        print(f"Reward function score: {score}")

    # Run tests
    asyncio.run(test_explicit())
    asyncio.run(test_implicit())
    asyncio.run(test_reward_function())
