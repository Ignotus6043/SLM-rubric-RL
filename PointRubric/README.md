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

### Paper evaluation protocol

Table 3 evaluates every method on the same held-out `test.json` questions.
Any trained or calibrated component uses only `train.json` and `dev.json`.
Do **not** train on the full benchmark and then report on the same data.

Create the fixed paper split with:

```bash
cd PointRubric
python3 scripts/prepare_bench_fixed_split.py \
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

Final paper metrics must be reported on the held-out `test.json` split only.
The complete `data/bench.json` remains available for explicitly labeled
full-benchmark analyses of frozen, untuned judges.

## Main static judge set

The frozen paper-ready static runs should include:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`

## Canonical evaluation entrypoint

From the repository root, evaluate a Generative judge on the paper's held-out
test split with:

```bash
python PointRubric/scripts/benchmark_eval.py \
  --data-file PointRubric/data/bench.json \
  --prompt-file PointRubric/rubric_prompts/benchmark_judge.txt \
  --split-dir PointRubric/data/fixed_split_seed42_train40_dev10_test50 \
  --eval-split test \
  --models Qwen/Qwen3-1.7B \
  --output-dir outputs/pointrubric/generative/qwen3_1p7b
```

Omit `--split-dir` and use `--eval-split all` only for frozen, untuned judges
that are meant to be evaluated on the full benchmark.
