# Public Release Manifest

This repository is the organized public release for **Small Language Models as Judges for Rubric-Based Reinforcement Learning**.

## Paper

- `paper/Small_Language_Models_as_Judges_for_Rubric_Based_Reinforcement_Learning.pdf`: camera-ready PDF, rebuilt with all citations resolved.

## PointRubric

- `PointRubric/data/bench.json`: full 1,042-question benchmark.
- `PointRubric/data/fixed_split_seed42_train40_dev10_test50/`: canonical question-level train/dev/test split.
- `PointRubric/answer_prompts/` and `PointRubric/rubric_prompts/`: construction and evaluation prompts.
- `PointRubric/scripts/`: API-based construction, final filtering, frozen splitting, evaluator, and summary scripts.
- `datasets/PointRubric/`: provenance, counts, exact question IDs, and SHA-256 records.

## RaR-Science and transfer

- `rubric/data/prepare_data.py`: external dataset conversion to VERL parquet.
- `rubric/scripts/transfer/`: response-bank, static readout, calibration, diagnostics, and alignment analysis.
- `datasets/RaR-Science-Static/`: frozen dev/test question IDs and provenance.

## Judges and RL

- `rubric/scripts/bench/`: PointRubric Logprob and Probe evaluators plus human-audit utilities.
- `rubric/rubric_rl/`: generative judge client/reward, Probe server/reward, prompt construction, and evaluation helpers.
- `rubric/scripts/rl/preflight_reward_function.py`: fail-fast reward integration check.
- `configs/paper_rl.json`: exact paper-level RaR-Science GRPO settings.

## SFT

- `SFT/prepare_sft_data.py`: fixed-split conversion.
- `SFT/configs/`: portable Qwen3 0.6B, 1.7B, 4B, and 8B LLaMA-Factory configs.
- `SFT/eval.py`: tuned-judge evaluation.
- `SFT/dataset_info.json`: local dataset registration.

## Environment and documentation

- `README.md`: project overview and headline results.
- `REPRODUCIBILITY.md`: environment, data, static evaluation, SFT, and RL guide.
- `requirements.txt`: versions captured from the final RL environment.
- `.env.example`: variable names only; no credential values.
- `CITATION.cff`: citation metadata without private contact information.

## Intentionally absent

Credentials, populated environment files, private machine paths, Slurm account and email settings, raw API construction dumps, third-party model weights, checkpoints, Probe artifacts, logs, caches, W&B metadata, and generated result directories are excluded.
