# PointRubric Dataset Record

This folder records the exact PointRubric/OpenRubricBench benchmark used by the paper.

## Source

- Source dataset loader: `OpenRubrics/OpenRubric-v2`, split `train`
- Fallback in the construction script: `OpenRubrics/OpenRubrics`
- Construction script in the bundle: `PointRubric/scripts/prepare_bench_fixed_split.py`
- Full released benchmark: `PointRubric/data/bench.json`
- Fixed paper split: `PointRubric/data/fixed_split_seed42_train40_dev10_test50/`

## Construction

The current benchmark was built by merging two local construction passes, keeping problems where `answer1.total_score == 10`, `answer2.total_score <= 3`, and at least one answer score is in `[2, 8]`, then deduplicating by `question_id`.

## Counts

- Full benchmark: 1042 questions, 4168 answer instances
- Fixed split seed: 42
- Train: 417 questions, 1668 answer instances
- Dev: 104 questions, 416 answer instances
- Test: 521 questions, 2084 answer instances

## Files

- `dataset_manifest.json`: source, construction, counts, and SHA-256 checksums.
- `question_ids.json`: full/train/dev/test question IDs used by the benchmark.
