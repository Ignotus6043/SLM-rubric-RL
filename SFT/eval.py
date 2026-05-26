import argparse
import gc
import json
import math
import os
import random
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]

SAFETY_BUFFER_GB = 12.0
MAX_BATCH_SIZE = 32
DEFAULT_MAX_PROMPT_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_WORKERS = 2
DEFAULT_SEED = 42


@dataclass
class ModelSpec:
    label: str
    model_ref: str
    adapter_ref: Optional[str] = None


def setup_hf_cache() -> str:
    cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir

    # Fix for occasional SSL_CERT_FILE misconfig (same pattern as OpenRubrics).
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if ssl_cert_file and not os.path.exists(ssl_cert_file):
        common_paths = [
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/ssl/ca-bundle.pem",
        ]
        for path in common_paths:
            if os.path.exists(path):
                os.environ["SSL_CERT_FILE"] = path
                break
        else:
            del os.environ["SSL_CERT_FILE"]

    return cache_dir


def resolve_reference(ref: str, base_dir: Path) -> str:
    ref = ref.strip()
    if not ref:
        return ref
    if os.path.isabs(ref):
        return ref
    candidate = base_dir / ref
    if candidate.exists():
        return str(candidate.resolve())
    return ref


def parse_model_spec(spec: str, base_dir: Path) -> ModelSpec:
    parts = [part.strip() for part in spec.split("::")]
    if len(parts) == 1:
        model_ref = resolve_reference(parts[0], base_dir)
        return ModelSpec(label=parts[0], model_ref=model_ref)
    if len(parts) == 2:
        label, model_ref = parts
        return ModelSpec(label=label, model_ref=resolve_reference(model_ref, base_dir))
    if len(parts) == 3:
        label, model_ref, adapter_ref = parts
        return ModelSpec(
            label=label,
            model_ref=resolve_reference(model_ref, base_dir),
            adapter_ref=resolve_reference(adapter_ref, base_dir),
        )
    raise ValueError(
        "Invalid model spec. Use `model`, `label::model`, or `label::base_model::adapter_path`."
    )


def load_model_and_tokenizer(
    spec: ModelSpec,
    cache_dir: str,
    hf_token: Optional[str],
    attn_implementation: Optional[str] = None,
    force_single_gpu: bool = False,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_ref,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": "auto",
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "token": hf_token,
    }
    if force_single_gpu and torch.cuda.is_available():
        model_kwargs["low_cpu_mem_usage"] = True
    else:
        model_kwargs["device_map"] = "auto"

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    elif "phi-3" in spec.model_ref.lower():
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(spec.model_ref, **model_kwargs)
    if spec.adapter_ref:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "peft is required to evaluate LoRA adapters. Install it in the HPC environment."
            ) from exc
        model = PeftModel.from_pretrained(model, spec.adapter_ref, is_trainable=False)

    if force_single_gpu and torch.cuda.is_available():
        model = model.to(f"cuda:{torch.cuda.current_device()}")

    return tokenizer, model


def calculate_auto_batch_size(
    model_config: Any,
    max_prompt_length: int,
    max_new_tokens: int,
    safety_buffer_gb: float = SAFETY_BUFFER_GB,
) -> int:
    if not torch.cuda.is_available():
        return 1

    free_mem, _ = torch.cuda.mem_get_info()
    usable_mem = free_mem - (safety_buffer_gb * 1024**3)
    if usable_mem <= 0:
        return 1

    n_layers = getattr(model_config, "num_hidden_layers", 24)
    n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 8))
    n_heads = max(1, getattr(model_config, "num_attention_heads", 32))
    head_dim = getattr(model_config, "hidden_size", 4096) // n_heads
    bytes_per_token = 4 * n_layers * n_kv_heads * head_dim

    denom = (max_prompt_length + max_new_tokens) * max(1, bytes_per_token)
    auto_bs = usable_mem // denom
    if auto_bs <= 0:
        return 1

    return min(MAX_BATCH_SIZE, max(1, 2 ** int(math.log2(auto_bs))))


def format_rubric_rules(refined_rubric: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, rule in enumerate(refined_rubric, start=1):
        rule_type = "Hard Rule" if rule.get("type") == "hard" else "Soft Rule"
        text = rule.get("rubric", "").strip()
        lines.append(f"{i}. {text} [{rule_type}]")
    return "\n".join(lines)


def parse_verdict_vector(text: str, num_rules: int) -> Optional[List[int]]:
    """
    Strict parser for the required format:
      Rule i: <justification>. Verdict: Yes/No

    Requirements:
    - We must parse rule indices 1..num_rules exactly once.
    - Verdict token must be Yes/No (or Y/N variant).
    - Output vector is ordered by rule id.
    """
    if not text:
        return None

    verdict_by_rule: Dict[int, int] = {}
    pattern = re.compile(
        r"^\s*Rule\s*(\d+)\s*:\s*.*?\bVerdict\s*:\s*(Yes|No|Y|N)\s*$",
        flags=re.IGNORECASE,
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        rid = int(m.group(1))
        tok = m.group(2).strip().lower()
        verdict_by_rule[rid] = 1 if tok in {"yes", "y"} else 0

    if len(verdict_by_rule) != num_rules:
        return None
    if set(verdict_by_rule.keys()) != set(range(1, num_rules + 1)):
        return None

    return [verdict_by_rule[i] for i in range(1, num_rules + 1)]


def verdicts_to_score(verdicts: List[int], rule_weights: List[int]) -> int:
    return int(sum(v * w for v, w in zip(verdicts, rule_weights)))


@dataclass
class BenchSample:
    sample_id: str
    question_id: str
    answer_id: int
    question: str
    response: str
    rubric_rules: str
    num_rules: int
    rule_weights: List[int]
    gold_verdicts: List[int]
    gold_total_score: int


class BenchRubricDataset(Dataset):
    def __init__(self, samples: List[BenchSample], prompt_template: str):
        self.samples = samples
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        prompt = self.prompt_template.format(
            instruction=s.question,
            response=s.response,
            rubric_rules=s.rubric_rules,
            num_rules=s.num_rules,
        )
        return {
            "sample_id": s.sample_id,
            "question_id": s.question_id,
            "answer_id": s.answer_id,
            "num_rules": s.num_rules,
            # Store variable-length vectors as JSON strings to avoid
            # default DataLoader collation transposing list dimensions.
            "rule_weights_json": json.dumps(s.rule_weights),
            "gold_verdicts_json": json.dumps(s.gold_verdicts),
            "gold_total_score": s.gold_total_score,
            "prompt": prompt,
        }


def build_prompt_for_sample(sample: BenchSample, prompt_template: str) -> str:
    return prompt_template.format(
        instruction=sample.question,
        response=sample.response,
        rubric_rules=sample.rubric_rules,
        num_rules=sample.num_rules,
    )


def truncate_prompt_text(tokenizer: AutoTokenizer, prompt: str, max_prompt_length: int) -> str:
    token_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=True,
    )["input_ids"]
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def materialize_eval_inputs(
    dataset: BenchRubricDataset,
    prompt_tokenizer: Optional[AutoTokenizer] = None,
    max_prompt_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    eval_inputs: List[Dict[str, Any]] = []
    for sample in dataset.samples:
        prompt = build_prompt_for_sample(sample, dataset.prompt_template)
        if prompt_tokenizer is not None and max_prompt_length is not None:
            prompt = truncate_prompt_text(prompt_tokenizer, prompt, max_prompt_length)
        eval_inputs.append(
            {
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "answer_id": int(sample.answer_id),
                "num_rules": int(sample.num_rules),
                "rule_weights": [int(x) for x in sample.rule_weights],
                "gold_verdicts": [int(x) for x in sample.gold_verdicts],
                "gold_total_score": int(sample.gold_total_score),
                "prompt": prompt,
            }
        )
    return eval_inputs


def build_prediction_row(model_label: str, eval_input: Dict[str, Any], completion: str) -> Dict[str, Any]:
    first_nonempty = next((ln.strip() for ln in completion.splitlines() if ln.strip()), "")
    num_rules = int(eval_input["num_rules"])
    rule_weights = [int(x) for x in eval_input["rule_weights"]]
    gold_verdicts = [int(x) for x in eval_input["gold_verdicts"]]
    gold_total = int(eval_input["gold_total_score"])

    pred_verdicts = parse_verdict_vector(completion, num_rules)
    parse_ok = pred_verdicts is not None and len(pred_verdicts) == num_rules

    if parse_ok:
        pred_total = verdicts_to_score(pred_verdicts, rule_weights)
        rule_matches = sum(int(a == b) for a, b in zip(gold_verdicts, pred_verdicts))
        weighted_rule_matches = sum(
            w for a, b, w in zip(gold_verdicts, pred_verdicts, rule_weights) if a == b
        )
        vector_exact = pred_verdicts == gold_verdicts
        score_exact = pred_total == gold_total
    else:
        pred_total = None
        rule_matches = 0
        weighted_rule_matches = 0
        vector_exact = False
        score_exact = False

    return {
        "model": model_label,
        "sample_id": eval_input["sample_id"],
        "question_id": eval_input["question_id"],
        "answer_id": int(eval_input["answer_id"]),
        "num_rules": num_rules,
        "weight_sum": int(sum(rule_weights)),
        "gold_verdicts": gold_verdicts,
        "pred_verdicts": pred_verdicts,
        "gold_vector_yn": vector_to_yn(gold_verdicts),
        "pred_vector_yn": vector_to_yn(pred_verdicts),
        "gold_total_score": gold_total,
        "pred_total_score": pred_total,
        "parse_ok": parse_ok,
        "vector_exact": vector_exact,
        "rule_matches": int(rule_matches),
        "weighted_rule_matches": int(weighted_rule_matches),
        "score_exact": score_exact,
        "raw_output": completion,
        "first_line_output": first_nonempty,
    }


def flatten_samples(data: List[Dict[str, Any]]) -> List[BenchSample]:
    samples: List[BenchSample] = []

    for q in data:
        question_id = str(q.get("question_id", ""))
        question = q.get("question", "")
        refined = q.get("refined_rubric", [])
        answers = q.get("answers", [])

        # Preserve refined rubric order as canonical rule order.
        rule_ids = [str(rule.get("rule_id", i + 1)) for i, rule in enumerate(refined)]
        rule_weights = [3 if rule.get("type") == "hard" else 1 for rule in refined]
        rubric_rules = format_rubric_rules(refined)
        num_rules = len(refined)

        if num_rules == 0:
            continue

        for ans in answers:
            answer_id = int(ans.get("answer_id", -1))
            answer_text = ans.get("answer_text", "")
            grade = ans.get("grade", [])

            verdict_map: Dict[str, int] = {}
            for item in grade:
                rid = str(item.get("rule_id", ""))
                verdict = str(item.get("verdict", "")).strip().lower()
                verdict_map[rid] = 1 if verdict == "yes" else 0

            gold_verdicts = [verdict_map.get(rid, 0) for rid in rule_ids]
            computed_total = verdicts_to_score(gold_verdicts, rule_weights)
            gold_total_score = int(ans.get("total_score", computed_total))

            samples.append(
                BenchSample(
                    sample_id=f"{question_id}_{answer_id}",
                    question_id=question_id,
                    answer_id=answer_id,
                    question=question,
                    response=answer_text,
                    rubric_rules=rubric_rules,
                    num_rules=num_rules,
                    rule_weights=rule_weights,
                    gold_verdicts=gold_verdicts,
                    gold_total_score=gold_total_score,
                )
            )

    return samples


def load_question_ids_from_jsonl(path: str) -> Set[str]:
    question_ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question_id" not in row:
                raise ValueError(f"Missing `question_id` in {path} at line {line_no}.")
            question_ids.add(str(row["question_id"]))
    if not question_ids:
        raise RuntimeError(f"No question IDs found in filter file: {path}")
    return question_ids


def filter_bench_by_question_ids(data: List[Dict[str, Any]], allowed_question_ids: Set[str]) -> List[Dict[str, Any]]:
    return [row for row in data if str(row.get("question_id", "")) in allowed_question_ids]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def vector_to_yn(vec: Optional[List[int]]) -> str:
    if vec is None:
        return ""
    return "".join("Y" if int(v) == 1 else "N" for v in vec)


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return safe_div(num, den_x * den_y)


def average_ranks(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson_corr(average_ranks(xs), average_ranks(ys))


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_samples = len(rows)
    parsed_rows = [r for r in rows if r.get("parse_ok")]
    n_parsed = len(parsed_rows)

    total_rules_all = sum(r["num_rules"] for r in rows)
    total_rules = sum(r["num_rules"] for r in parsed_rows)
    matched_rules = sum(r["rule_matches"] for r in parsed_rows)
    weighted_totals_all = sum(r["weight_sum"] for r in rows)
    weighted_matches = sum(r["weighted_rule_matches"] for r in parsed_rows)
    weighted_totals = sum(r["weight_sum"] for r in parsed_rows)

    vector_exact_all = sum(1 for r in rows if r["vector_exact"])
    vector_exact = sum(1 for r in parsed_rows if r["vector_exact"])
    score_exact_all = sum(1 for r in rows if r["score_exact"])
    score_exact = sum(1 for r in parsed_rows if r["score_exact"])

    sample_hamming_error_all = sum((r["num_rules"] - r["rule_matches"]) / max(1, r["num_rules"]) for r in rows)
    sample_hamming_error = sum((r["num_rules"] - r["rule_matches"]) / max(1, r["num_rules"]) for r in parsed_rows)
    sample_weighted_error_all = sum(
        (r["weight_sum"] - r["weighted_rule_matches"]) / max(1, r["weight_sum"]) for r in rows
    )
    sample_weighted_error = sum(
        (r["weight_sum"] - r["weighted_rule_matches"]) / max(1, r["weight_sum"]) for r in parsed_rows
    )

    score_abs_err = sum(abs(r["pred_total_score"] - r["gold_total_score"]) for r in parsed_rows)
    score_sq_err = sum((r["pred_total_score"] - r["gold_total_score"]) ** 2 for r in parsed_rows)
    score_bias = sum((r["pred_total_score"] - r["gold_total_score"]) for r in parsed_rows)
    score_within_1_all = sum(
        1
        for r in rows
        if r["pred_total_score"] is not None and abs(r["pred_total_score"] - r["gold_total_score"]) <= 1
    )
    score_within_2_all = sum(
        1
        for r in rows
        if r["pred_total_score"] is not None and abs(r["pred_total_score"] - r["gold_total_score"]) <= 2
    )
    score_within_1 = sum(1 for r in parsed_rows if abs(r["pred_total_score"] - r["gold_total_score"]) <= 1)
    score_within_2 = sum(1 for r in parsed_rows if abs(r["pred_total_score"] - r["gold_total_score"]) <= 2)

    # Rule-level binary stats for YES class.
    tp = fp = fn = tn = 0
    for r in parsed_rows:
        y_true = r["gold_verdicts"]
        y_pred = r["pred_verdicts"]
        for gt, pr in zip(y_true, y_pred):
            if gt == 1 and pr == 1:
                tp += 1
            elif gt == 0 and pr == 1:
                fp += 1
            elif gt == 1 and pr == 0:
                fn += 1
            else:
                tn += 1

    precision_yes = safe_div(tp, tp + fp)
    recall_yes = safe_div(tp, tp + fn)
    f1_yes = safe_div(2 * precision_yes * recall_yes, precision_yes + recall_yes)
    precision_no = safe_div(tn, tn + fn)
    recall_no = safe_div(tn, tn + fp)
    f1_no = safe_div(2 * precision_no * recall_no, precision_no + recall_no)
    macro_f1 = (f1_yes + f1_no) / 2.0
    balanced_accuracy = (recall_yes + recall_no) / 2.0
    jaccard_yes = safe_div(tp, tp + fp + fn)
    mcc = safe_div((tp * tn) - (fp * fn), math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # Question-level ordering metrics over answer-level total scores.
    by_question_all: Dict[str, List[Dict[str, Any]]] = {}
    by_question: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_question_all.setdefault(r["question_id"], []).append(r)
    for r in parsed_rows:
        by_question.setdefault(r["question_id"], []).append(r)

    pair_correct = 0
    pair_total = 0
    top1_correct = 0
    bottom1_correct = 0
    q_count = 0

    for q_rows in by_question.values():
        if len(q_rows) < 2:
            continue

        q_count += 1
        for a, b in combinations(q_rows, 2):
            gold_diff = a["gold_total_score"] - b["gold_total_score"]
            pred_diff = a["pred_total_score"] - b["pred_total_score"]
            pair_total += 1
            if (gold_diff == 0 and pred_diff == 0) or (gold_diff > 0 and pred_diff > 0) or (gold_diff < 0 and pred_diff < 0):
                pair_correct += 1

        gold_max = max(x["gold_total_score"] for x in q_rows)
        pred_max = max(x["pred_total_score"] for x in q_rows)
        gold_best = {x["answer_id"] for x in q_rows if x["gold_total_score"] == gold_max}
        pred_best = {x["answer_id"] for x in q_rows if x["pred_total_score"] == pred_max}
        if gold_best & pred_best:
            top1_correct += 1

        gold_min = min(x["gold_total_score"] for x in q_rows)
        pred_min = min(x["pred_total_score"] for x in q_rows)
        gold_worst = {x["answer_id"] for x in q_rows if x["gold_total_score"] == gold_min}
        pred_worst = {x["answer_id"] for x in q_rows if x["pred_total_score"] == pred_min}
        if gold_worst & pred_worst:
            bottom1_correct += 1

    pair_total_all = 0
    questions_total = 0
    questions_fully_parseable = 0
    top1_all_questions_correct = 0
    bottom1_all_questions_correct = 0
    for q_rows in by_question_all.values():
        if len(q_rows) < 2:
            continue

        questions_total += 1
        pair_total_all += math.comb(len(q_rows), 2)
        if not all(r["parse_ok"] for r in q_rows):
            continue

        questions_fully_parseable += 1
        gold_max = max(x["gold_total_score"] for x in q_rows)
        pred_max = max(x["pred_total_score"] for x in q_rows)
        gold_best = {x["answer_id"] for x in q_rows if x["gold_total_score"] == gold_max}
        pred_best = {x["answer_id"] for x in q_rows if x["pred_total_score"] == pred_max}
        if gold_best & pred_best:
            top1_all_questions_correct += 1

        gold_min = min(x["gold_total_score"] for x in q_rows)
        pred_min = min(x["pred_total_score"] for x in q_rows)
        gold_worst = {x["answer_id"] for x in q_rows if x["gold_total_score"] == gold_min}
        pred_worst = {x["answer_id"] for x in q_rows if x["pred_total_score"] == pred_min}
        if gold_worst & pred_worst:
            bottom1_all_questions_correct += 1

    gold_scores = [float(r["gold_total_score"]) for r in parsed_rows]
    pred_scores = [float(r["pred_total_score"]) for r in parsed_rows]

    return {
        "samples_total": n_samples,
        "samples_parse_ok": n_parsed,
        "parse_success_rate": safe_div(n_parsed, n_samples),
        "samples_parse_failed": n_samples - n_parsed,
        "rule_exact_match_rate": safe_div(vector_exact, n_parsed),
        "rule_exact_match_rate_all_samples": safe_div(vector_exact_all, n_samples),
        "rule_hamming_accuracy": safe_div(matched_rules, total_rules),
        "rule_hamming_accuracy_all_samples": safe_div(matched_rules, total_rules_all),
        "rule_hamming_error": 1.0 - safe_div(matched_rules, total_rules),
        "rule_hamming_error_per_sample": safe_div(sample_hamming_error, n_parsed),
        "rule_hamming_error_per_sample_all_samples": safe_div(sample_hamming_error_all, n_samples),
        "rule_weighted_accuracy": safe_div(weighted_matches, weighted_totals),
        "rule_weighted_accuracy_all_samples": safe_div(weighted_matches, weighted_totals_all),
        "rule_weighted_error": 1.0 - safe_div(weighted_matches, weighted_totals),
        "rule_weighted_error_per_sample": safe_div(sample_weighted_error, n_parsed),
        "rule_weighted_error_per_sample_all_samples": safe_div(sample_weighted_error_all, n_samples),
        "yes_precision": precision_yes,
        "yes_recall": recall_yes,
        "yes_f1": f1_yes,
        "no_precision": precision_no,
        "no_recall": recall_no,
        "no_f1": f1_no,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "jaccard_yes": jaccard_yes,
        "mcc": mcc,
        "score_exact_match_rate": safe_div(score_exact, n_parsed),
        "score_exact_match_rate_all_samples": safe_div(score_exact_all, n_samples),
        "score_mae": safe_div(score_abs_err, n_parsed),
        "score_rmse": math.sqrt(safe_div(score_sq_err, n_parsed)) if n_parsed else 0.0,
        "score_mean_error_bias": safe_div(score_bias, n_parsed),
        "score_within_1_rate": safe_div(score_within_1, n_parsed),
        "score_within_1_rate_all_samples": safe_div(score_within_1_all, n_samples),
        "score_within_2_rate": safe_div(score_within_2, n_parsed),
        "score_within_2_rate_all_samples": safe_div(score_within_2_all, n_samples),
        "score_pearson": pearson_corr(gold_scores, pred_scores),
        "score_spearman": spearman_corr(gold_scores, pred_scores),
        "pairwise_order_accuracy": safe_div(pair_correct, pair_total),
        "pairwise_order_accuracy_all_pairs": safe_div(pair_correct, pair_total_all),
        "pairwise_parseable_coverage": safe_div(pair_total, pair_total_all),
        "top1_set_accuracy": safe_div(top1_correct, q_count),
        "top1_set_accuracy_all_questions": safe_div(top1_all_questions_correct, questions_total),
        "bottom1_set_accuracy": safe_div(bottom1_correct, q_count),
        "bottom1_set_accuracy_all_questions": safe_div(bottom1_all_questions_correct, questions_total),
        "questions_with_parseable_answers": q_count,
        "questions_total_with_multiple_answers": questions_total,
        "question_parseable_coverage": safe_div(q_count, questions_total),
        "questions_fully_parseable": questions_fully_parseable,
        "question_full_parse_rate": safe_div(questions_fully_parseable, questions_total),
    }


def evaluate_model(
    model_label: str,
    dataset: BenchRubricDataset,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch_size: int,
    num_workers: int,
    max_prompt_length: int,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        shuffle=False,
    )

    rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in tqdm(dl, desc=model_label):
            inputs = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

            for i in range(len(batch["sample_id"])):
                prompt_len = inputs.input_ids[i].shape[0]
                completion = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
                rows.append(
                    build_prediction_row(
                        model_label,
                        {
                            "sample_id": batch["sample_id"][i],
                            "question_id": batch["question_id"][i],
                            "answer_id": int(batch["answer_id"][i]),
                            "num_rules": int(batch["num_rules"][i]),
                            "rule_weights": [int(x) for x in json.loads(batch["rule_weights_json"][i])],
                            "gold_verdicts": [int(x) for x in json.loads(batch["gold_verdicts_json"][i])],
                            "gold_total_score": int(batch["gold_total_score"][i]),
                        },
                        completion,
                    )
                )

    metrics = compute_metrics(rows)
    metrics["model"] = model_label
    metrics["batch_size"] = batch_size
    return metrics, rows


def evaluate_model_vllm(
    spec: ModelSpec,
    dataset: BenchRubricDataset,
    cache_dir: str,
    hf_token: Optional[str],
    max_prompt_length: int,
    max_new_tokens: int,
    max_num_seqs: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    enforce_eager: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if spec.adapter_ref:
        raise RuntimeError("The vLLM backend currently supports pretrained or merged checkpoints, not base+adapter evaluation.")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        exc_text = str(exc)
        if "undefined symbol" in exc_text or "vllm/_C" in exc_text or "vllm._C" in exc_text:
            raise RuntimeError(
                "vLLM is installed but its compiled extension is incompatible with the current torch/CUDA runtime. "
                "Reinstall vLLM against the torch version inside this environment, or reinstall torch and vLLM together."
            ) from exc
        raise RuntimeError("vLLM is not installed in the current environment.") from exc

    prompt_tokenizer = AutoTokenizer.from_pretrained(
        spec.model_ref,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    eval_inputs = materialize_eval_inputs(
        dataset,
        prompt_tokenizer=prompt_tokenizer,
        max_prompt_length=max_prompt_length,
    )
    del prompt_tokenizer

    llm_kwargs: Dict[str, Any] = {
        "model": spec.model_ref,
        "trust_remote_code": True,
        "download_dir": cache_dir,
        "dtype": "auto",
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": enforce_eager,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate([item["prompt"] for item in eval_inputs], sampling_params)
    rows = [
        build_prediction_row(
            spec.label,
            eval_input,
            output.outputs[0].text if output.outputs else "",
        )
        for eval_input, output in zip(eval_inputs, outputs)
    ]

    metrics = compute_metrics(rows)
    metrics["model"] = spec.label
    metrics["batch_size"] = max_num_seqs

    del llm, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, rows


def write_outputs(
    output_dir: str,
    metric_rows: List[Dict[str, Any]],
    all_pred_rows: List[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> None:
    report_csv = os.path.join(output_dir, "report.csv")
    metrics_csv = os.path.join(output_dir, "overall_metrics.csv")
    preds_csv = os.path.join(output_dir, "per_sample_predictions.csv")
    config_json = os.path.join(output_dir, "run_config.json")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(report_csv, index=False)
    # Backward-compatible alias for older scripts.
    metrics_df.to_csv(metrics_csv, index=False)

    pred_df = pd.DataFrame(all_pred_rows)
    if "raw_output" in pred_df.columns:
        pred_df["raw_output"] = pred_df["raw_output"].astype(str).str.slice(0, 1000)
    pred_df.to_csv(preds_csv, index=False)

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base, merged, or LoRA-adapted models on PointRubric.")
    parser.add_argument("--data-file", type=str, default="bench.json")
    parser.add_argument(
        "--question-id-filter-jsonl",
        type=str,
        default=None,
        help="Optional JSONL file whose `question_id` values define the evaluated question subset.",
    )
    parser.add_argument("--prompt-file", type=str, default=os.path.join("..", "PointRubric", "rubric_prompts", "benchmark_judge.txt"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of answer-samples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--inference-backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend to use for generation.",
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Optional fixed batch size for all evaluated models.",
    )
    parser.add_argument(
        "--auto-batch-size-multiplier",
        type=int,
        default=1,
        help="Multiply the auto-estimated batch size by this factor before capping.",
    )
    parser.add_argument(
        "--auto-batch-safety-buffer-gb",
        type=float,
        default=SAFETY_BUFFER_GB,
        help="Reserved GPU memory buffer used during automatic batch-size estimation.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention backend override for model loading.",
    )
    parser.add_argument(
        "--force-single-gpu",
        action="store_true",
        help="Force the full model onto the current GPU instead of using device_map=auto.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the vLLM backend.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory that vLLM may use.",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences batched by vLLM.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max model length override for vLLM. Defaults to max_prompt_length + max_new_tokens.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Force vLLM to run in eager mode.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model specs: `model`, `label::model`, or `label::base_model::adapter_path`.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.batch_size_override is not None and args.batch_size_override < 1:
        raise ValueError("--batch-size-override must be >= 1.")
    if args.auto_batch_size_multiplier < 1:
        raise ValueError("--auto-batch-size-multiplier must be >= 1.")
    if args.auto_batch_safety_buffer_gb < 0:
        raise ValueError("--auto-batch-safety-buffer-gb must be >= 0.")
    if args.vllm_tensor_parallel_size < 1:
        raise ValueError("--vllm-tensor-parallel-size must be >= 1.")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("--vllm-gpu-memory-utilization must be in (0, 1].")
    if args.vllm_max_num_seqs is not None and args.vllm_max_num_seqs < 1:
        raise ValueError("--vllm-max-num-seqs must be >= 1.")
    if args.vllm_max_model_len is not None and args.vllm_max_model_len < 1:
        raise ValueError("--vllm-max-model-len must be >= 1.")
    if args.inference_backend == "vllm" and args.attn_implementation != "auto":
        print("[WARN] --attn-implementation is ignored when using the vLLM backend.")
    if args.inference_backend == "vllm" and args.force_single_gpu:
        print("[WARN] --force-single-gpu is ignored when using the vLLM backend.")

    cache_dir = setup_hf_cache()
    hf_token = os.environ.get("HF_TOKEN")

    base_dir = Path(__file__).resolve().parent
    data_file = resolve_reference(args.data_file, base_dir)
    question_id_filter_jsonl = (
        resolve_reference(args.question_id_filter_jsonl, base_dir) if args.question_id_filter_jsonl else None
    )
    prompt_file = resolve_reference(args.prompt_file, base_dir)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    default_output_dir = str(base_dir / "Results" / f"bench_eval_{timestamp}")
    output_dir = args.output_dir if args.output_dir else default_output_dir
    if not os.path.isabs(output_dir):
        output_dir = str((base_dir / output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Data file: {data_file}")
    if question_id_filter_jsonl:
        print(f"[INFO] Question filter JSONL: {question_id_filter_jsonl}")
    print(f"[INFO] Prompt file: {prompt_file}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Inference backend: {args.inference_backend}")

    with open(data_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    if question_id_filter_jsonl:
        allowed_question_ids = load_question_ids_from_jsonl(question_id_filter_jsonl)
        bench_data = filter_bench_by_question_ids(bench_data, allowed_question_ids)
        matched_question_ids = {str(row.get("question_id", "")) for row in bench_data}
        missing_question_ids = sorted(allowed_question_ids - matched_question_ids)
        print(
            f"[INFO] Filtered benchmark to {len(matched_question_ids)} / {len(allowed_question_ids)} "
            f"questions from the provided JSONL split"
        )
        if missing_question_ids:
            preview = ", ".join(missing_question_ids[:5])
            suffix = " ..." if len(missing_question_ids) > 5 else ""
            print(f"[WARN] Question IDs in filter but not bench data: {preview}{suffix}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    all_samples = flatten_samples(bench_data)
    if args.max_samples is not None:
        all_samples = all_samples[: args.max_samples]

    print(f"[INFO] Loaded {len(all_samples)} answer-samples for evaluation")
    if not all_samples:
        raise RuntimeError("No valid samples found in bench dataset.")

    dataset = BenchRubricDataset(all_samples, prompt_template)

    model_specs = [parse_model_spec(spec, base_dir) for spec in (args.models if args.models else MODEL_NAMES)]

    metric_rows: List[Dict[str, Any]] = []
    all_pred_rows: List[Dict[str, Any]] = []
    failed_models: List[str] = []

    for spec in model_specs:
        print(f"\n>>> Loading {spec.label} ...")
        try:
            if args.inference_backend == "vllm":
                max_num_seqs = args.vllm_max_num_seqs or args.batch_size_override or MAX_BATCH_SIZE
                max_model_len = args.vllm_max_model_len or (args.max_prompt_length + args.max_new_tokens)
                print(f"[INFO] vLLM tensor parallel size for {spec.label}: {args.vllm_tensor_parallel_size}")
                print(f"[INFO] vLLM gpu memory utilization for {spec.label}: {args.vllm_gpu_memory_utilization}")
                print(f"[INFO] vLLM max num seqs for {spec.label}: {max_num_seqs}")
                print(f"[INFO] vLLM max model len for {spec.label}: {max_model_len}")
                model_metrics, rows = evaluate_model_vllm(
                    spec=spec,
                    dataset=dataset,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                    max_num_seqs=max_num_seqs,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_model_len=max_model_len,
                    enforce_eager=args.vllm_enforce_eager,
                )
            else:
                tokenizer, model = load_model_and_tokenizer(
                    spec,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                    attn_implementation=None if args.attn_implementation == "auto" else args.attn_implementation,
                    force_single_gpu=args.force_single_gpu,
                )
                auto_batch_size = calculate_auto_batch_size(
                    model.config,
                    args.max_prompt_length,
                    args.max_new_tokens,
                    safety_buffer_gb=args.auto_batch_safety_buffer_gb,
                )
                auto_batch_size = min(MAX_BATCH_SIZE, max(1, auto_batch_size * args.auto_batch_size_multiplier))
                batch_size = args.batch_size_override if args.batch_size_override is not None else auto_batch_size
                print(f"[INFO] Attention backend for {spec.label}: {args.attn_implementation}")
                print(f"[INFO] Force single GPU for {spec.label}: {args.force_single_gpu}")
                print(f"[INFO] Auto batch size for {spec.label}: {auto_batch_size}")
                if args.batch_size_override is not None:
                    print(f"[INFO] Overriding batch size for {spec.label}: {batch_size}")

                model_metrics, rows = evaluate_model(
                    model_label=spec.label,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    model=model,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                )

            metric_rows.append(model_metrics)
            all_pred_rows.extend(rows)

            write_outputs(
                output_dir=output_dir,
                metric_rows=metric_rows,
                all_pred_rows=all_pred_rows,
                run_config={
                    "data_file": data_file,
                    "question_id_filter_jsonl": question_id_filter_jsonl,
                    "prompt_file": prompt_file,
                    "output_dir": output_dir,
                    "models": [spec.label for spec in model_specs],
                    "model_specs": [
                        {
                            "label": item.label,
                            "model_ref": item.model_ref,
                            "adapter_ref": item.adapter_ref,
                        }
                        for item in model_specs
                    ],
                    "failed_models": failed_models,
                    "max_samples": args.max_samples,
                    "seed": args.seed,
                    "num_workers": args.num_workers,
                    "max_prompt_length": args.max_prompt_length,
                    "max_new_tokens": args.max_new_tokens,
                    "inference_backend": args.inference_backend,
                    "batch_size_override": args.batch_size_override,
                    "auto_batch_size_multiplier": args.auto_batch_size_multiplier,
                    "auto_batch_safety_buffer_gb": args.auto_batch_safety_buffer_gb,
                    "attn_implementation": args.attn_implementation,
                    "force_single_gpu": args.force_single_gpu,
                    "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                    "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "vllm_max_num_seqs": args.vllm_max_num_seqs,
                    "vllm_max_model_len": args.vllm_max_model_len,
                    "vllm_enforce_eager": args.vllm_enforce_eager,
                },
            )

            if args.inference_backend == "transformers":
                del model, tokenizer
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as exc:
            print(f"[ERROR] Failed model {spec.label}: {exc}")
            traceback.print_exc()
            failed_models.append(spec.label)
            write_outputs(
                output_dir=output_dir,
                metric_rows=metric_rows,
                all_pred_rows=all_pred_rows,
                run_config={
                    "data_file": data_file,
                    "question_id_filter_jsonl": question_id_filter_jsonl,
                    "prompt_file": prompt_file,
                    "output_dir": output_dir,
                    "models": [spec.label for spec in model_specs],
                    "model_specs": [
                        {
                            "label": item.label,
                            "model_ref": item.model_ref,
                            "adapter_ref": item.adapter_ref,
                        }
                        for item in model_specs
                    ],
                    "failed_models": failed_models,
                    "max_samples": args.max_samples,
                    "seed": args.seed,
                    "num_workers": args.num_workers,
                    "max_prompt_length": args.max_prompt_length,
                    "max_new_tokens": args.max_new_tokens,
                    "inference_backend": args.inference_backend,
                    "batch_size_override": args.batch_size_override,
                    "auto_batch_size_multiplier": args.auto_batch_size_multiplier,
                    "auto_batch_safety_buffer_gb": args.auto_batch_safety_buffer_gb,
                    "attn_implementation": args.attn_implementation,
                    "force_single_gpu": args.force_single_gpu,
                    "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                    "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "vllm_max_num_seqs": args.vllm_max_num_seqs,
                    "vllm_max_model_len": args.vllm_max_model_len,
                    "vllm_enforce_eager": args.vllm_enforce_eager,
                },
            )

    report_csv = os.path.join(output_dir, "report.csv")
    metrics_csv = os.path.join(output_dir, "overall_metrics.csv")
    preds_csv = os.path.join(output_dir, "per_sample_predictions.csv")

    print("\n[INFO] Benchmark complete")
    print(f"[INFO] Report: {report_csv}")
    print(f"[INFO] Metrics (alias): {metrics_csv}")
    print(f"[INFO] Per-sample predictions: {preds_csv}")
    if failed_models:
        print(f"[WARN] Failed models: {failed_models}")


if __name__ == "__main__":
    main()
