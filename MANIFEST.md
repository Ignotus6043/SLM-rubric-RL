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
- `rubric/scripts/transfer/build_rarscience_gpt_verdict_sft_splits.py`: exact seed-42 paper split and optional SFT-record construction from the released scored bank.
- `datasets/RaR-Science-Static/reference_scored_bank.jsonl`: sanitized exact paper bank with 1,500 questions, 3,000 responses, and 22,550 GPT-4o criterion decisions.
- `datasets/RaR-Science-Static/`: exact 1,000/200/300 train/dev/test split, provenance, and SHA-256 records.

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
- `scripts/verify_release.py`: checksum, split-integrity, label, and score validation for the committed datasets.
- `.env.example`: variable names only; no credential values.
- `CITATION.cff`: citation metadata without private contact information.
- `LICENSE`: CC BY 4.0 license for original project contributions.
- `THIRD_PARTY_DATA.md`: upstream dataset, model, API, and software terms that are not superseded by the project license.

## Intentionally absent

Credentials, populated environment files, private machine paths, Slurm account and email settings, raw judge prose and API timing, third-party model weights, checkpoints, Probe artifacts, logs, caches, W&B metadata, and generated result directories are excluded.
