# RaR-Science-Static Dataset Record

This directory contains the exact fixed response bank and reference labels used
for the paper's primary RaR-Science-Static evaluation.

## Source and construction

- Source: [`anisha2102/RaR-Science`](https://huggingface.co/datasets/anisha2102/RaR-Science), converted from its validation split.
- Sampling: 1,500 questions selected with seed 42 from 2,247 unique converted questions.
- Candidates: the source reference answer and one greedy `Qwen/Qwen3-4B` response per question.
- Reference labels: criterion-level verdicts from `gpt-4o-2024-11-20`.
- Split: 1,000 training, 200 development, and 300 held-out test questions.

The response bank and reference labels are combined in
`reference_scored_bank.jsonl`. Raw judge prose, API timing, credentials, and
machine-specific paths have been removed. Criterion labels, criterion weights,
parse status, aggregate scores, rubrics, and both candidate responses are
preserved.

## Files

- `reference_scored_bank.jsonl`: 1,500 questions, 3,000 responses, and 22,550 criterion decisions.
- `question_ids.json`: complete train/dev/test split in one JSON record.
- `train_question_ids.txt`, `dev_question_ids.txt`, `test_question_ids.txt`: line-oriented IDs accepted by the released evaluation scripts.
- `dataset_manifest.json`: provenance, model snapshots, decoding settings, counts, and SHA-256 checksums.

The split is reproducible with
`rubric/scripts/transfer/build_rarscience_gpt_verdict_sft_splits.py` using
`--train-sizes 1000 --dev-count 200 --heldout-count 300 --seed 42`.

## Validate or reuse the reference labels

The released file is directly compatible with the RaR-Science Probe and
alignment scripts. For example, a candidate scored bank can be compared with
the released reference labels using:

```bash
python rubric/scripts/transfer/evaluate_rarscience_judge_alignment.py \
  --reference-scored-bank datasets/RaR-Science-Static/reference_scored_bank.jsonl \
  --candidate-scored-bank outputs/rarscience/candidate/scored_bank.jsonl \
  --output-dir outputs/rarscience/candidate/alignment
```

To fit the paper's 1.7B Probe on the exact question split:

```bash
python rubric/scripts/transfer/score_rarscience_response_bank_hf_probe.py \
  --gpt-scored-bank datasets/RaR-Science-Static/reference_scored_bank.jsonl \
  --output-dir outputs/rarscience/probe \
  --judge-model Qwen/Qwen3-1.7B \
  --candidate-ids reference_answer qwen3_4b_greedy \
  --train-question-ids-file datasets/RaR-Science-Static/train_question_ids.txt \
  --dev-question-ids-file datasets/RaR-Science-Static/dev_question_ids.txt \
  --heldout-question-ids-file datasets/RaR-Science-Static/test_question_ids.txt \
  --eval-split heldout --layers auto --pooling last \
  --probe-classifier linear --score-mode probability --seed 42
```

See `THIRD_PARTY_DATA.md` for upstream terms. Original project contributions
are released under CC BY 4.0 to the extent the authors hold the applicable
rights.
