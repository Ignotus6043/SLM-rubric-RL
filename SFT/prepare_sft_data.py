import argparse
import json
import os
import random
from typing import Dict, List, Optional, Set, Tuple


PROMPT_TEMPLATE = """You are an impartial rubric grader.

Instruction:
{instruction}

Candidate Response:
{response}

Rubric Rules ({num_rules} total):
{rubric_rules}

Task:
For each rubric rule, decide whether the response satisfies it.

Output format requirement (strict):
- Return exactly {num_rules} lines.
- Each line must follow this format exactly:
  Rule i: <short justification>. Verdict: Yes/No
- Use i = 1..{num_rules} in order.
- Keep each justification brief (1 sentence).
- Verdict must be exactly Yes or No.
"""


def format_rubric_rules(refined_rubric: List[Dict]) -> str:
    lines: List[str] = []
    for i, rule in enumerate(refined_rubric, start=1):
        rtype = "Hard Rule" if rule.get("type") == "hard" else "Soft Rule"
        lines.append(f"{i}. {rule.get('rubric', '').strip()} [{rtype}]")
    return "\n".join(lines)


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return "No justification provided."
    if text[-1] not in ".!?":
        text += "."
    return text


def build_target_lines(answer: Dict, refined_rubric: List[Dict]) -> str:
    verdict_map = {str(g.get("rule_id")): str(g.get("verdict", "")).strip().lower() for g in answer.get("grade", [])}
    out: List[str] = []
    for i, rule in enumerate(refined_rubric, start=1):
        rid = str(rule.get("rule_id", i))
        verdict = "Yes" if verdict_map.get(rid) == "yes" else "No"
        # Keep justification short and deterministic from rubric text.
        rub = rule.get("rubric", "").strip()
        just = rub if len(rub) <= 140 else rub[:137] + "..."
        out.append(f"Rule {i}: {ensure_sentence(just)} Verdict: {verdict}")
    return "\n".join(out)


def convert_examples(bench: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    all_items: List[Dict] = []
    per_question_count: Dict[str, int] = {}

    for q in bench:
        qid = str(q.get("question_id"))
        instruction = q.get("question", "")
        refined = q.get("refined_rubric", [])
        num_rules = len(refined)
        if num_rules == 0:
            continue

        rubric_rules = format_rubric_rules(refined)
        for ans in q.get("answers", []):
            response = ans.get("answer_text", "")
            prompt = PROMPT_TEMPLATE.format(
                instruction=instruction,
                response=response,
                num_rules=num_rules,
                rubric_rules=rubric_rules,
            ).strip()
            target = build_target_lines(ans, refined)

            ex = {
                "id": f"{qid}_{ans.get('answer_id')}",
                "question_id": qid,
                "model_answer_id": ans.get("answer_id"),
                "instruction": prompt,
                "input": "",
                "output": target,
            }
            all_items.append(ex)
            per_question_count[qid] = per_question_count.get(qid, 0) + 1

    return all_items, per_question_count


def split_by_question(examples: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    qids = sorted({ex["question_id"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(qids)

    n_train_q = max(1, int(len(qids) * train_ratio))
    train_q = set(qids[:n_train_q])

    train = [ex for ex in examples if ex["question_id"] in train_q]
    test = [ex for ex in examples if ex["question_id"] not in train_q]
    return train, test


def split_by_manifest(
    examples: List[Dict],
    train_question_ids: Set[str],
    test_question_ids: Set[str],
) -> Tuple[List[Dict], List[Dict]]:
    train = [ex for ex in examples if ex["question_id"] in train_question_ids]
    test = [ex for ex in examples if ex["question_id"] in test_question_ids]
    return train, test


def convert_bench_rows_to_examples(rows: List[Dict]) -> List[Dict]:
    examples, _ = convert_examples(rows)
    return examples


def load_fixed_split(split_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    base = os.path.abspath(split_dir)
    paths = {
        "train": os.path.join(base, "train.json"),
        "dev": os.path.join(base, "dev.json"),
        "test": os.path.join(base, "test.json"),
    }
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing fixed split files in {base}: {', '.join(missing)}")

    split_rows = {}
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            split_rows[name] = json.load(f)

    manifest_path = os.path.join(base, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return (
        convert_bench_rows_to_examples(split_rows["train"]),
        convert_bench_rows_to_examples(split_rows["dev"]),
        convert_bench_rows_to_examples(split_rows["test"]),
        manifest,
    )


def load_split_manifest(path: str) -> Tuple[Set[str], Set[str], Dict]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_question_ids = {str(x) for x in manifest.get("train_question_ids", [])}
    test_question_ids = {str(x) for x in manifest.get("test_question_ids", [])}
    if not train_question_ids or not test_question_ids:
        raise ValueError(f"Split manifest must contain non-empty train/test question IDs: {path}")
    if train_question_ids & test_question_ids:
        raise ValueError(f"Split manifest has overlapping train/test question IDs: {path}")
    return train_question_ids, test_question_ids, manifest


def build_split_manifest(
    bench_path: str,
    train_ratio: float,
    seed: int,
    train: List[Dict],
    test: List[Dict],
    source_manifest: Optional[str],
) -> Dict:
    return {
        "bench_path": os.path.abspath(bench_path),
        "split_method": "question_level",
        "train_ratio": train_ratio,
        "seed": seed,
        "split_manifest_source": source_manifest,
        "total_examples": len(train) + len(test),
        "train_examples": len(train),
        "test_examples": len(test),
        "train_questions": len({x["question_id"] for x in train}),
        "test_questions": len({x["question_id"] for x in test}),
        "train_question_ids": sorted({str(x["question_id"]) for x in train}, key=int),
        "test_question_ids": sorted({str(x["question_id"]) for x in test}, key=int),
    }


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="bench.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.4,
        help="Legacy train/test fallback ratio. Canonical paper runs use --fixed-split-dir.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional JSON manifest with explicit frozen train/test question IDs.",
    )
    parser.add_argument(
        "--fixed-split-dir",
        default=None,
        help="Bench fixed split directory containing train.json, dev.json, test.json.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.fixed_split_dir:
        train, dev, test, source_manifest = load_fixed_split(args.fixed_split_dir)
        write_jsonl(os.path.join(args.out_dir, "sft_train.jsonl"), train)
        write_jsonl(os.path.join(args.out_dir, "sft_dev.jsonl"), dev)
        write_jsonl(os.path.join(args.out_dir, "sft_test.jsonl"), test)

        meta = {
            "split_method": "bench_fixed_train_dev_test",
            "fixed_split_dir": os.path.abspath(args.fixed_split_dir),
            "source_manifest": source_manifest,
            "train_examples": len(train),
            "dev_examples": len(dev),
            "test_examples": len(test),
            "train_questions": len({x["question_id"] for x in train}),
            "dev_questions": len({x["question_id"] for x in dev}),
            "test_questions": len({x["question_id"] for x in test}),
            "train_question_ids": sorted({str(x["question_id"]) for x in train}, key=int),
            "dev_question_ids": sorted({str(x["question_id"]) for x in dev}, key=int),
            "test_question_ids": sorted({str(x["question_id"]) for x in test}, key=int),
        }
        with open(os.path.join(args.out_dir, "split_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.out_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(json.dumps(meta, indent=2))
        return

    with open(args.bench, "r", encoding="utf-8") as f:
        bench = json.load(f)

    examples, _ = convert_examples(bench)
    all_question_ids = {str(ex["question_id"]) for ex in examples}
    source_manifest = None
    if args.split_manifest:
        train_question_ids, test_question_ids, manifest = load_split_manifest(args.split_manifest)
        source_manifest = os.path.abspath(args.split_manifest)
        if train_question_ids | test_question_ids != all_question_ids:
            raise ValueError(
                "Split manifest question IDs do not match the current benchmark question set."
            )
        train, test = split_by_manifest(examples, train_question_ids, test_question_ids)
        args.train_ratio = float(manifest.get("train_ratio", args.train_ratio))
        args.seed = int(manifest.get("seed", args.seed))
    else:
        train, test = split_by_question(examples, args.train_ratio, args.seed)

    train_path = os.path.join(args.out_dir, "sft_train.jsonl")
    test_path = os.path.join(args.out_dir, "sft_test.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)

    meta = build_split_manifest(
        bench_path=args.bench,
        train_ratio=args.train_ratio,
        seed=args.seed,
        train=train,
        test=test,
        source_manifest=source_manifest,
    )
    with open(os.path.join(args.out_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
