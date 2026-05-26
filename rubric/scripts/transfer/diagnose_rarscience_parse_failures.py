#!/usr/bin/env python3
"""Diagnose RaR-Science judge parse failures from a scored response bank."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RULE_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?Rule\s*(\d+)\b", flags=re.IGNORECASE)
VERDICT_RE = re.compile(r"\bVerdict\s*[:=-]?\s*(Yes|No|Y|N|True|False|0|1)\b", flags=re.IGNORECASE)


def _normalize_yes_no_token(token: str) -> int | None:
    token = token.strip().lower()
    if token in {"yes", "y", "true", "1"}:
        return 1
    if token in {"no", "n", "false", "0"}:
        return 0
    return None


def parse_explicit_verdict_vector(text: str, num_rules: int, *, strict: bool) -> list[int] | None:
    """Self-contained copy of the explicit judge parser; avoids importing litellm."""
    if not text:
        return None

    verdict_by_rule: dict[int, int] = {}
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

    if len(verdict_by_rule) == num_rules and set(verdict_by_rule) == set(range(1, num_rules + 1)):
        return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]

    if strict:
        return None

    verdict_by_rule = {}
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

    if set(verdict_by_rule) == set(range(1, num_rules + 1)):
        return [verdict_by_rule[idx] for idx in range(1, num_rules + 1)]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def raw_attempts(reward_result: dict[str, Any]) -> list[str]:
    attempts = reward_result.get("raw_judge_responses")
    if isinstance(attempts, list):
        return [str(item or "") for item in attempts]
    return [str(reward_result.get("raw_judge_response", "") or "")]


def raw_group_attempts(reward_result: dict[str, Any]) -> list[dict[str, Any]]:
    groups = reward_result.get("raw_judge_responses_by_group")
    if not isinstance(groups, list):
        return []
    normalized = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        start = int(group.get("group_start", index))
        end = int(group.get("group_end", start + 1))
        normalized.append(
            {
                "group_index": int(group.get("group_index", index)),
                "group_start": start,
                "group_end": end,
                "expected_rules": max(0, end - start),
                "raw_judge_response": str(group.get("raw_judge_response", "") or ""),
                "parse_source": str(group.get("parse_source", "") or ""),
            }
        )
    return normalized


def classify_raw(text: str, expected_rules: int) -> tuple[str, dict[str, Any]]:
    rule_ids = []
    verdict_lines = 0
    for line in text.splitlines():
        rule_match = RULE_LINE_RE.search(line)
        if rule_match:
            rule_ids.append(int(rule_match.group(1)))
        if VERDICT_RE.search(line):
            verdict_lines += 1

    strict_vector = parse_explicit_verdict_vector(text, expected_rules, strict=True)
    loose_vector = parse_explicit_verdict_vector(text, expected_rules, strict=False)

    if not text.strip():
        reason = "empty_response"
    elif strict_vector is not None:
        reason = "parser_mismatch_strict_should_have_parsed"
    elif loose_vector is not None:
        reason = "strict_only_failure"
    elif "<think>" in text and "</think>" not in text:
        reason = "unclosed_thinking"
    elif "</think>" in text and not text.split("</think>", 1)[1].strip():
        reason = "reasoning_only_no_final_content"
    elif rule_ids and len(set(rule_ids)) < expected_rules:
        reason = "partial_rule_lines"
    elif verdict_lines and verdict_lines < expected_rules:
        reason = "partial_verdict_lines"
    elif not rule_ids:
        reason = "no_rule_lines"
    else:
        reason = "malformed_rule_lines"

    details = {
        "chars": len(text),
        "contains_think": "<think>" in text or "</think>" in text,
        "rule_line_count": len(rule_ids),
        "unique_rule_line_count": len(set(rule_ids)),
        "verdict_line_count": verdict_lines,
        "strict_parse_ok": strict_vector is not None,
        "loose_parse_ok": loose_vector is not None,
        "starts_with": text[:200],
        "ends_with": text[-400:],
    }
    return reason, details


def candidate_id_of(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id", "unknown"))


def main() -> int:
    args = parse_args()
    scored_bank = Path(args.scored_bank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(scored_bank)
    overall = Counter()
    per_candidate: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    group_examples: list[dict[str, Any]] = []
    attempt_hist = Counter()
    group_overall = Counter()
    group_by_size = defaultdict(Counter)

    for question in records:
        expected_rules = len(question.get("rubric", []) or [])
        for candidate in question.get("candidates", []):
            cid = candidate_id_of(candidate)
            reward_result = candidate.get("reward_result", {})
            if not isinstance(reward_result, dict):
                continue
            attempts = raw_attempts(reward_result)
            used_fallback = bool(reward_result.get("fallback_used", False))
            raw_parse_success = bool(reward_result.get("raw_parse_success", False))
            attempt_hist[len(attempts)] += 1

            overall["samples"] += 1
            per_candidate[cid]["samples"] += 1
            overall["raw_parse_success"] += int(raw_parse_success)
            per_candidate[cid]["raw_parse_success"] += int(raw_parse_success)
            overall["fallback_used"] += int(used_fallback)
            per_candidate[cid]["fallback_used"] += int(used_fallback)

            if raw_parse_success:
                continue

            reason, details = classify_raw(attempts[-1] if attempts else "", expected_rules)
            overall[f"reason:{reason}"] += 1
            per_candidate[cid][f"reason:{reason}"] += 1
            if len(examples) < args.max_examples:
                examples.append(
                    {
                        "question_id": question.get("question_id"),
                        "candidate_id": cid,
                        "expected_rules": expected_rules,
                        "attempts": len(attempts),
                        "fallback_used": used_fallback,
                        "reason": reason,
                        "details": details,
                    }
                )

            groups = raw_group_attempts(reward_result)
            parse_ok = reward_result.get("criterion_parse_ok")
            parse_ok = parse_ok if isinstance(parse_ok, list) else []
            for group in groups:
                group_size = int(group["expected_rules"])
                if group_size <= 0:
                    continue
                start = int(group["group_start"])
                end = int(group["group_end"])
                local_parse_ok = all(bool(item) for item in parse_ok[start:end]) if parse_ok else False
                group_overall["groups"] += 1
                group_overall["group_parse_success"] += int(local_parse_ok)
                group_by_size[group_size]["groups"] += 1
                group_by_size[group_size]["group_parse_success"] += int(local_parse_ok)
                if local_parse_ok:
                    continue
                group_reason, group_details = classify_raw(group["raw_judge_response"], group_size)
                group_overall[f"reason:{group_reason}"] += 1
                group_by_size[group_size][f"reason:{group_reason}"] += 1
                if len(group_examples) < args.max_examples:
                    group_examples.append(
                        {
                            "question_id": question.get("question_id"),
                            "candidate_id": cid,
                            "group_index": group["group_index"],
                            "group_start": start,
                            "group_end": end,
                            "expected_rules": group_size,
                            "reason": group_reason,
                            "details": group_details,
                        }
                    )

    summary = {
        "scored_bank": str(scored_bank),
        "overall": dict(overall),
        "group_overall": dict(group_overall),
        "group_by_size": {str(key): dict(value) for key, value in sorted(group_by_size.items())},
        "attempt_histogram": {str(k): v for k, v in sorted(attempt_hist.items())},
        "per_candidate": {key: dict(value) for key, value in sorted(per_candidate.items())},
        "examples": examples,
        "group_examples": group_examples,
    }
    json_path = output_dir / "parse_failure_diagnosis.json"
    md_path = output_dir / "parse_failure_diagnosis.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RaR-Science Parse Failure Diagnosis",
        "",
        f"- Scored bank: `{scored_bank}`",
        f"- Samples: `{overall['samples']}`",
        f"- Raw parse success: `{overall['raw_parse_success']}`",
        f"- Fallback used: `{overall['fallback_used']}`",
        "",
        "## Failure Reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(overall.items()):
        if key.startswith("reason:"):
            lines.append(f"| {key.split(':', 1)[1]} | {value} |")

    if group_overall:
        lines.extend(
            [
                "",
                "## Group-Level Failure Reasons",
                "",
                f"- Groups: `{group_overall['groups']}`",
                f"- Group parse success: `{group_overall['group_parse_success']}`",
                "",
                "| reason | count |",
                "| --- | ---: |",
            ]
        )
        for key, value in sorted(group_overall.items()):
            if key.startswith("reason:"):
                lines.append(f"| {key.split(':', 1)[1]} | {value} |")

        lines.extend(
            [
                "",
                "## Group Parse By Size",
                "",
                "| group_size | groups | parse_ok | top_failure_reasons |",
                "| ---: | ---: | ---: | --- |",
            ]
        )
        for group_size, counter in sorted(group_by_size.items()):
            failures = [
                (key.split(":", 1)[1], value)
                for key, value in counter.items()
                if key.startswith("reason:")
            ]
            failures.sort(key=lambda item: (-item[1], item[0]))
            top = ", ".join(f"{name}={count}" for name, count in failures[:3])
            lines.append(
                f"| {group_size} | {counter['groups']} | {counter['group_parse_success']} | {top or 'NA'} |"
            )

    lines.extend(
        [
            "",
            "## Per Candidate",
            "",
            "| candidate_id | samples | raw_parse | fallback | top_failure_reasons |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for cid, counter in sorted(per_candidate.items()):
        failures = [
            (key.split(":", 1)[1], value)
            for key, value in counter.items()
            if key.startswith("reason:")
        ]
        failures.sort(key=lambda item: (-item[1], item[0]))
        top = ", ".join(f"{name}={count}" for name, count in failures[:3])
        lines.append(
            f"| {cid} | {counter['samples']} | {counter['raw_parse_success']} | "
            f"{counter['fallback_used']} | {top or 'NA'} |"
        )

    lines.extend(["", "## Examples", ""])
    for idx, example in enumerate(examples, start=1):
        details = example["details"]
        lines.extend(
            [
                f"### Example {idx}",
                "",
                f"- question_id: `{example['question_id']}`",
                f"- candidate_id: `{example['candidate_id']}`",
                f"- reason: `{example['reason']}`",
                f"- attempts: `{example['attempts']}`",
                f"- rule/verdict lines: `{details['rule_line_count']}` / `{details['verdict_line_count']}`",
                "",
                "Last 400 chars:",
                "",
                "```text",
                details["ends_with"],
                "```",
                "",
            ]
        )

    if group_examples:
        lines.extend(["", "## Group Examples", ""])
        for idx, example in enumerate(group_examples, start=1):
            details = example["details"]
            lines.extend(
                [
                    f"### Group Example {idx}",
                    "",
                    f"- question_id: `{example['question_id']}`",
                    f"- candidate_id: `{example['candidate_id']}`",
                    f"- group: `{example['group_start']}:{example['group_end']}`",
                    f"- expected local rules: `{example['expected_rules']}`",
                    f"- reason: `{example['reason']}`",
                    f"- rule/verdict lines: `{details['rule_line_count']}` / `{details['verdict_line_count']}`",
                    "",
                    "Last 400 chars:",
                    "",
                    "```text",
                    details["ends_with"],
                    "```",
                    "",
                ]
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("=======================================")
    print(f"Scored bank:     {scored_bank}")
    print(f"Samples:         {overall['samples']}")
    print(f"Raw parse ok:    {overall['raw_parse_success']}")
    print(f"Fallback used:   {overall['fallback_used']}")
    print(f"Diagnosis JSON:  {json_path}")
    print(f"Diagnosis MD:    {md_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
