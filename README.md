# PointRubric Supplementary Bundle

This folder is the recommended upload unit for the paper supplement. It contains the canonical benchmark data and non-revealing scripts needed to inspect and reproduce the benchmark/evaluator workflows. It intentionally excludes local credentials, API smoke tests, Slurm launchers, raw construction dumps, caches, model checkpoints, logs, result tables, paper drafting notes, and machine-specific manuals.

## Contents

- `PointRubric/data/bench.json`: full PointRubric/OpenRubricBench benchmark.
- `PointRubric/data/fixed_split_seed42_train40_dev10_test50/`: canonical train/dev/test split used for tuned judges.
- `PointRubric/answer_prompts/` and `PointRubric/rubric_prompts/`: prompt templates.
- `datasets/PointRubric/`: source, split, checksum, and question-ID record for PointRubric.
- `datasets/RaR-Science-Static/`: source, split, checksum, and question-ID record for the RaR-Science static-transfer experiments.
- `PointRubric/scripts/`: benchmark evaluation, fixed-split generation, and summary utilities.
- `rubric/scripts/bench/`: local generated/logprob/probe evaluator scripts for PointRubric.
- `rubric/scripts/transfer/`: selected RaR-Science transfer/evaluator scripts that do not contain local launch configuration.
- `rubric/rubric_rl/`: probe reward server/function code used by the RL reward path.
- `SFT/`: SFT data preparation and evaluation utilities only.

## Deliberately Omitted

- `.env`, API keys, tokens, and local credential tests.
- OpenAI API benchmark-construction scripts and scratch OpenRubrics processing code.
- Slurm/HPC launchers, account names, email addresses, absolute paths, and cluster manuals.
- `hf_cache/`, `LLaMA-Factory/`, `rubric/verl/`, checkpoints, model weights, logs, and generated result folders.
- Raw construction dumps such as `PointRubric/data/4o-v*/` and old split variants.
- Paper drafting notes and internal audit/protocol documents.

## Basic Use

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python PointRubric/scripts/prepare_bench_fixed_split.py \
  --input PointRubric/data/bench.json \
  --output-dir /tmp/pointrubric_split \
  --seed 42 --train-frac 0.4 --dev-frac 0.1 --test-frac 0.5
```

For local-model scoring, set `HF_HOME` and `HF_TOKEN` in the shell only if your selected model requires them. Do not add secrets to files in this bundle.
