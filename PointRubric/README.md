# PointRubric

`PointRubric` is the main static evaluator-fidelity benchmark for this project.

## Current benchmark construction

The current `data/bench.json` is built with the following filters:

1. Merge `data/4o-v1/bench.json` and `data/4o-v2/bench.json`.
2. Keep only problems where `answer1.total_score == 10`.
3. Keep only problems where `answer2.total_score <= 3`.
4. Keep only problems with at least one answer score in `[2, 8]`.
5. Deduplicate by `question_id` (one record per problem).

Final benchmark size: **1042 problems**.

This corresponds to:

- `1042` questions
- `4168` answer instances

## Protocol policy

### Frozen / zero-shot / off-the-shelf judges

Use the full benchmark:

- `data/bench.json`

This is the main static evaluator-fidelity result.

### PointRubric-tuned judges

Do **not** train on the full benchmark and then report on the same data.

For any judge that is tuned on PointRubric itself, first create the fixed held-out split:

```bash
cd PointRubric
python3 prepare_bench_fixed_split.py \
  --input data/bench.json \
  --output-dir data/fixed_split_seed42_train40_dev10_test50 \
  --seed 42 \
  --train-frac 0.4 \
  --dev-frac 0.1 \
  --test-frac 0.5
```

This creates a question-level split with:

- train: `40%`
- dev: `10%`
- test: `50%`
- seed: `42`

For tuned judges:

- train on `train.json`
- select the best checkpoint on `dev.json`
- report only `test.json`

Canonical PointRubric-SFT tuned judges:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`

The SFT training configs must select the best checkpoint on the dev split:

- `eval_dataset: sft_dev_bench`
- `load_best_model_at_end: true`
- `metric_for_best_model: eval_loss`
- `greater_is_better: false`

Final tuned-judge metrics must be reported on the held-out `test.json` split
only. Do not report PointRubric-SFT models on the full benchmark.

## Main static judge set

The frozen paper-ready static runs should include:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`

## Canonical evaluation entrypoint

Run local model judging with:

```bash
python benchmark_eval.py \
  --data-file data/bench.json \
  --prompt-file rubric_prompts/benchmark_judge.txt \
  --output-dir Results/static_eval
```
