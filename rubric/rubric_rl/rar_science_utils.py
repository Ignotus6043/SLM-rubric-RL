#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def load_parquet_rows(dataset_path: str) -> list[dict[str, Any]]:
    return pq.read_table(dataset_path).to_pylist()


def write_parquet_rows(rows: list[dict[str, Any]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out)


def question_text(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    question = extra_info.get("question") or ""
    if not question:
        question = json.dumps(row.get("prompt", ""), ensure_ascii=False)
    return question


def stable_question_id(row: dict[str, Any], index: int | None = None) -> str:
    question = question_text(row)
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
    return f"q_{digest}"


def unique_rows_by_question(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        qid = stable_question_id(row, index)
        if qid in seen:
            continue
        seen.add(qid)
        unique_rows.append(row)
    return unique_rows


def slugify_model_name(model_name: str) -> str:
    slug = model_name.strip().lower()
    slug = slug.replace("/", "_")
    slug = re.sub(r"([0-9])\.([0-9])", r"\1p\2", slug)
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "model"
