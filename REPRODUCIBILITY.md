# Reproducibility Guide

This release separates three reproducibility targets:

1. **Exact static data:** use the committed PointRubric benchmark and RaR-Science-Static response bank, labels, and question splits.
2. **Judge and reward code:** rerun the Generative, Yes/No Logprob, and Probe implementations on compatible model checkpoints.
3. **RL training:** connect the released reward function to VERL using the configuration recorded in `configs/paper_rl.json`.

Downloaded model weights, Probe artifacts, and RL checkpoints are not committed. The exact sanitized RaR-Science-Static reference labels used in the paper are committed; new API-based construction runs remain non-deterministic and write to ignored output directories.

## 1. Environment

The final RL environment used Python 3.12.13 and the package versions in `requirements.txt`. The most consequential versions were:

| Component | Version |
| --- | --- |
| PyTorch | 2.8.0 |
| Transformers | 4.56.1 |
| Datasets | 4.8.4 |
| vLLM | 0.11.0 |
| Ray | 2.55.0 |
| VERL | 0.8.0.dev0 |

Create a clean environment and install a CUDA-compatible PyTorch build for your system. Then install the remaining dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

VERL is an external dependency and is not vendored. Install a revision compatible with the captured `0.8.0.dev0` package environment from the [official VERL repository](https://github.com/volcengine/verl). The SFT configs target [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which is also installed separately.

For local imports:

```bash
export PYTHONPATH="$PWD/rubric${PYTHONPATH:+:$PYTHONPATH}"
```

Copy `.env.example` to an untracked `.env` only if needed. Prefer exporting credentials in the job environment. Never commit populated values.

## 2. Verify the frozen PointRubric split

The paper split is already committed at `PointRubric/data/fixed_split_seed42_train40_dev10_test50/`. Recreate it into an ignored directory:

```bash
python PointRubric/scripts/prepare_bench_fixed_split.py \
  --input PointRubric/data/bench.json \
  --output-dir outputs/pointrubric_split \
  --seed 42 --train-frac 0.4 --dev-frac 0.1 --test-frac 0.5
```

Compare its three split JSON files and counts with the committed split. The path fields in `manifest.json` will reflect the output directory you choose. Dataset provenance, counts, question IDs, and SHA-256 digests are under `datasets/PointRubric/`.

## 3. PointRubric construction

The released `bench.json` is the canonical artifact. A procedural run of its historical API-based construction stages is:

```bash
NUM_SAMPLES=10
python PointRubric/scripts/retrieve_data.py "$NUM_SAMPLES" 42
python PointRubric/scripts/refine_rubric.py
python PointRubric/scripts/generate_answers.py
python PointRubric/scripts/grader.py
```

These stages read/write `PointRubric/data/` and require `OPENAI_API_KEY`. API outputs can change as hosted model snapshots change, so a fresh construction run is procedural rather than byte-identical. To merge completed grading passes and apply the released quality filters:

```bash
python PointRubric/scripts/build_final_benchmark.py \
  --inputs path/to/pass1/bench.json path/to/pass2/bench.json \
  --output outputs/bench.json
```

## 4. Static judge evaluation

Run a Generative judge on the paper's held-out PointRubric test split:

```bash
python PointRubric/scripts/benchmark_eval.py \
  --data-file PointRubric/data/bench.json \
  --prompt-file PointRubric/rubric_prompts/benchmark_judge.txt \
  --split-dir PointRubric/data/fixed_split_seed42_train40_dev10_test50 \
  --eval-split test \
  --models Qwen/Qwen3-1.7B \
  --output-dir outputs/pointrubric/generative/qwen3_1p7b
```

Run the Yes/No Logprob readout:

```bash
python rubric/scripts/bench/score_bench_hf_logprob.py \
  --bench-json PointRubric/data/bench.json \
  --judge-model Qwen/Qwen3-1.7B \
  --output-dir outputs/pointrubric/logprob \
  --split-dir PointRubric/data/fixed_split_seed42_train40_dev10_test50 \
  --eval-split test --score-mode binary --seed 42
```

Fit and evaluate a linear last-token Probe:

```bash
python rubric/scripts/bench/score_bench_hf_probe.py \
  --bench-json PointRubric/data/bench.json \
  --judge-model Qwen/Qwen3-1.7B \
  --output-dir outputs/pointrubric/probe \
  --split-dir PointRubric/data/fixed_split_seed42_train40_dev10_test50 \
  --eval-split test --layers auto --probe-classifier linear --seed 42
```

The Probe command writes `probe_artifact.pt` plus predictions, metrics, and a run config. All output paths are ignored by Git.

The exact paper bank and GPT-4o criterion labels are committed at
`datasets/RaR-Science-Static/reference_scored_bank.jsonl`; the disjoint
1,000/200/300 question split is stored alongside it. The workflow that produced
and consumes these artifacts is implemented under `rubric/scripts/transfer/`:

1. Convert RaR-Science with `rubric/data/prepare_data.py`.
2. Recreate the paper's 1,000/200/300 seed-42 split with `build_rarscience_gpt_verdict_sft_splits.py`.
3. Build a fixed response bank with `build_rarscience_response_bank.py`.
4. Score the same bank with the Generative, Logprob, or Probe entrypoints.
5. Compare criterion labels using `evaluate_rarscience_judge_alignment.py`.

To score the exact bank with the paper's Probe configuration, use the command in
[`datasets/RaR-Science-Static/README.md`](datasets/RaR-Science-Static/README.md).
Candidate judge outputs can be checked against the released reference labels
with `evaluate_rarscience_judge_alignment.py`.

The committed split can be reproduced from the released scored bank with:

```bash
python rubric/scripts/transfer/build_rarscience_gpt_verdict_sft_splits.py \
  --gpt-scored-bank datasets/RaR-Science-Static/reference_scored_bank.jsonl \
  --output-root outputs/rarscience/paper_split \
  --train-sizes 1000 --dev-count 200 --heldout-count 300 --seed 42
```

## 5. SFT baseline

Create the train/dev/test JSONL files from the frozen PointRubric split:

```bash
python SFT/prepare_sft_data.py \
  --bench PointRubric/data/bench.json \
  --out-dir SFT/data \
  --fixed-split-dir PointRubric/data/fixed_split_seed42_train40_dev10_test50
```

The portable LLaMA-Factory configs are in `SFT/configs/`; they preserve the paper hyperparameters and write checkpoints under ignored `outputs/sft/` paths. For example:

```bash
llamafactory-cli train SFT/configs/qwen3_1p7b_sft.yaml
```

## 6. RL data and reward integration

Prepare RaR-Science in VERL’s parquet schema:

```bash
python rubric/data/prepare_data.py \
  --dataset science \
  --output_dir outputs/data/rubric_rl \
  --privileged_info_mode rubric
```

For a Generative judge, configure VERL’s custom reward function as:

- path: `rubric/rubric_rl/reward_function.py`
- callable: `compute_score`

The judge client reads `JUDGE_MODEL`, `JUDGE_API_BASE`, and `JUDGE_API_KEY`. It supports OpenAI directly and OpenAI-compatible local backends through LiteLLM.

For a Probe judge, first produce a RaR-Science `probe_artifact.pt` using `score_rarscience_response_bank_hf_probe.py`. Start the frozen Probe server on its own GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python -m rubric_rl.probe_reward_server \
  --artifact outputs/rarscience/probe/probe_artifact.pt \
  --host 127.0.0.1 --port 8621 \
  --score-mode probability
```

Then configure VERL’s custom reward function as:

- path: `rubric/rubric_rl/probe_reward_function.py`
- callable: `compute_score`
- environment: `PROBE_REWARD_API_BASE=http://127.0.0.1:8621`

Before allocating a full training run, validate the reward path against prepared parquet data:

```bash
python rubric/scripts/rl/preflight_reward_function.py \
  --repo-dir rubric \
  --reward-func-path rubric_rl/probe_reward_function.py \
  --data-file outputs/data/rubric_rl/train.parquet
```

Use `configs/paper_rl.json` as the authoritative paper settings record. It deliberately avoids cluster-specific resource directives; map GPU type, partition, account, storage, and container settings to your own system.

## 7. Human audit

`rubric/scripts/bench/sample_orb_human_audit.py` creates a blinded audit sample, and `analyze_orb_human_audit.py` summarizes completed annotations. Audit sheets and annotator metadata are not committed.

## Release boundaries

- **Committed:** source code, prompts, PointRubric, the sanitized RaR-Science-Static paper bank and reference labels, frozen splits, checksums, portable configs, and the paper.
- **Regenerated/downloaded:** source-dataset conversions, Hugging Face model weights, new API annotations or response banks, Probe artifacts, checkpoints, and result files.
- **Never committed:** credentials, private paths, cluster account/email settings, environment dumps, logs, caches, and experiment-tracker metadata.
