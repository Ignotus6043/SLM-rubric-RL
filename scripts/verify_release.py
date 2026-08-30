#!/usr/bin/env python3
"""Verify committed dataset checksums, split integrity, and release hygiene."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_hashes(manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    for relative_path, expected in manifest["sha256"].items():
        path = ROOT / relative_path
        if not path.is_file():
            raise AssertionError(f"Missing checksummed file: {relative_path}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(
                f"SHA-256 mismatch for {relative_path}: expected {expected}, got {actual}"
            )


def question_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids = {str(row.get("question_id", "")) for row in rows}
    if "" in ids:
        raise AssertionError("A dataset split contains an empty question_id")
    if len(ids) != len(rows):
        raise AssertionError("A dataset split contains duplicate question IDs")
    return ids


def verify_pointrubric() -> None:
    manifest_path = ROOT / "datasets/PointRubric/dataset_manifest.json"
    verify_manifest_hashes(manifest_path)
    manifest = read_json(manifest_path)
    full_rows = read_json(ROOT / manifest["released_files"]["full_benchmark"])
    full_ids = question_ids(full_rows)

    split_dir = ROOT / manifest["released_files"]["fixed_split_dir"]
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        rows = read_json(split_dir / f"{split}.json")
        split_ids[split] = question_ids(rows)
        expected = manifest["counts"][f"{split}_questions"]
        if len(rows) != expected:
            raise AssertionError(f"PointRubric {split} count is {len(rows)}, expected {expected}")

    if any(
        split_ids[left] & split_ids[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise AssertionError("PointRubric train/dev/test splits overlap")
    if set().union(*split_ids.values()) != full_ids:
        raise AssertionError("PointRubric splits do not exactly cover the full benchmark")


def read_id_file(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"Duplicate question IDs in {path.relative_to(ROOT)}")
    return ids


def verify_rarscience() -> None:
    manifest_path = ROOT / "datasets/RaR-Science-Static/dataset_manifest.json"
    verify_manifest_hashes(manifest_path)
    manifest = read_json(manifest_path)
    combined_ids = read_json(ROOT / manifest["released_files"]["question_ids"])

    split_ids: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        path = ROOT / manifest["released_files"][f"{split}_ids"]
        file_ids = read_id_file(path)
        json_ids = [str(value) for value in combined_ids[split]]
        if file_ids != json_ids:
            raise AssertionError(f"RaR-Science {split} ID text and JSON records differ")
        split_ids[split] = set(file_ids)
        expected = manifest["paper_split"][f"{split}_questions"]
        if len(file_ids) != expected:
            raise AssertionError(f"RaR-Science {split} count is {len(file_ids)}, expected {expected}")

    if any(
        split_ids[left] & split_ids[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise AssertionError("RaR-Science train/dev/test splits overlap")
    all_ids = set().union(*split_ids.values())
    if all_ids != set(map(str, combined_ids["all"])):
        raise AssertionError("RaR-Science splits do not exactly cover question_ids.json['all']")

    id_to_split = {question_id: split for split, ids in split_ids.items() for question_id in ids}
    bank_path = ROOT / manifest["released_files"]["reference_scored_bank"]
    bank_ids: set[str] = set()
    candidates = 0
    decisions = 0
    forbidden = ("/scratch/", "/Users/", "fx2137")
    with bank_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if any(token in line for token in forbidden):
                raise AssertionError(f"Private machine identifier found on bank line {line_number}")
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in bank_ids:
                raise AssertionError(f"Duplicate bank question_id: {question_id}")
            bank_ids.add(question_id)
            if row.get("split_role") != id_to_split.get(question_id):
                raise AssertionError(f"Incorrect split_role for {question_id}")

            row_candidates = row.get("candidates", [])
            expected_candidate_ids = manifest["bank_construction"]["candidate_ids"]
            if [candidate.get("candidate_id") for candidate in row_candidates] != expected_candidate_ids:
                raise AssertionError(f"Unexpected candidate IDs for {question_id}")
            candidates += len(row_candidates)

            for candidate in row_candidates:
                result = candidate["reward_result"]
                labels = result["criterion_scores"]
                weights = result["criterion_weights"]
                parse_ok = result["criterion_parse_ok"]
                if not (len(labels) == len(weights) == len(parse_ok) == len(row["rubric"])):
                    raise AssertionError(f"Criterion vector length mismatch for {question_id}")
                if any(label not in (0, 0.0, 1, 1.0) for label in labels):
                    raise AssertionError(f"Non-binary criterion label for {question_id}")
                if not all(parse_ok) or candidate.get("num_parse_failures") != 0:
                    raise AssertionError(f"Reference-label parse failure for {question_id}")
                expected_score = sum(label * weight for label, weight in zip(labels, weights)) / sum(weights)
                if not math.isclose(float(result["score"]), expected_score, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(f"Aggregate score mismatch for {question_id}")
                decisions += len(labels)

    expected_labels = manifest["reference_labels"]
    if bank_ids != all_ids:
        raise AssertionError("RaR-Science bank and frozen split question IDs differ")
    if len(bank_ids) != expected_labels["questions"]:
        raise AssertionError("RaR-Science question count differs from the manifest")
    if candidates != expected_labels["candidates"]:
        raise AssertionError("RaR-Science candidate count differs from the manifest")
    if decisions != expected_labels["criterion_decisions"]:
        raise AssertionError("RaR-Science criterion count differs from the manifest")


def main() -> None:
    verify_pointrubric()
    verify_rarscience()
    print("Release verification passed: checksums, splits, labels, and aggregate scores are consistent.")


if __name__ == "__main__":
    main()
