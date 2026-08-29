# RaR-Science-Static Dataset Record

This folder records the exact RaR-Science static-transfer dataset protocol used by the paper.

## Source

- Source dataset: `anisha2102/RaR-Science` from Hugging Face Datasets
- Conversion script in this repository: `rubric/data/prepare_data.py`
- Static evaluation source split: converted `val.parquet`
- Fixed split script in the bundle: `rubric/scripts/transfer/prepare_rarscience_eval_split.py`

## Fixed Split

The static evaluator experiments use `val.parquet` as a source pool, then freeze a seed-42 split:

- Unique source questions available: 2247
- Dev diagnostics split: 200 questions
- Test transfer split: 500 questions

## Paper Static Banks

- `rarsci_refonly_seed42_test500`: test500, reference-answer-only response bank, GPT-4o pseudo-gold labels.
- `rarsci_static_dev_seed42_dev200`: dev200, multi-candidate diagnostics bank with reference answer plus Qwen3-4B-Base, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, and Qwen3-8B greedy answers.

## Files

- `dataset_manifest.json`: source, split protocol, static bank definitions, and SHA-256 checksums for the local converted parquet files.
- `question_ids.json`: exact `dev200` and `test500` question IDs.

Raw RaR-Science parquet files are not copied into the submission bundle because they are an external dataset dependency.
