# Bench Benchmark Metrics

This file defines every column in `report.csv`.

## Setup
- Unit of evaluation: one `(question, answer)` sample from `data/bench.json`.
- Ground truth: rubric verdict vector from `answers[*].grade`.
- Prediction: parsed model output Y/N vector.
- Rule weights: hard rule = `3`, soft rule = `1`.
- Total score: weighted sum of satisfied rules.

## Columns in `report.csv`
- `samples_total`: Number of evaluated samples.
- `samples_parse_ok`: Samples where predicted Y/N vector was parsed successfully.
- `parse_success_rate`: `samples_parse_ok / samples_total`.
- `samples_parse_failed`: `samples_total - samples_parse_ok`.

- `rule_exact_match_rate`: Fraction of parsed samples where the full predicted Y/N vector exactly matches gold.
- `rule_hamming_accuracy`: Rule-level agreement over all parsed samples.
- `rule_hamming_error`: `1 - rule_hamming_accuracy`.
- `rule_hamming_error_per_sample`: Mean normalized Hamming distance per sample.

- `rule_weighted_accuracy`: Weighted rule agreement using hard=3, soft=1.
- `rule_weighted_error`: `1 - rule_weighted_accuracy`.
- `rule_weighted_error_per_sample`: Mean per-sample weighted disagreement rate.

- `yes_precision`: Among predicted `Y`, fraction truly `Y`.
- `yes_recall`: Among true `Y`, fraction predicted `Y`.
- `yes_f1`: F1 for `Y` class.
- `no_precision`: Among predicted `N`, fraction truly `N`.
- `no_recall`: Among true `N`, fraction predicted `N`.
- `no_f1`: F1 for `N` class.
- `macro_f1`: `(yes_f1 + no_f1) / 2`.
- `balanced_accuracy`: `(yes_recall + no_recall) / 2`.
- `jaccard_yes`: Jaccard/IoU for `Y` labels.
- `mcc`: Matthews correlation coefficient on rule-level binary labels.

- `score_exact_match_rate`: Fraction of parsed samples where predicted total score equals gold total score.
- `score_mae`: Mean absolute error of total score.
- `score_rmse`: Root mean squared error of total score.
- `score_mean_error_bias`: Mean signed error `(pred - gold)`; negative means under-scoring.
- `score_within_1_rate`: Fraction with `|pred - gold| <= 1`.
- `score_within_2_rate`: Fraction with `|pred - gold| <= 2`.
- `score_pearson`: Pearson correlation between predicted and gold total scores.
- `score_spearman`: Spearman rank correlation between predicted and gold total scores.

- `pairwise_order_accuracy`: Within each question, fraction of answer pairs whose relative order matches gold order.
- `top1_set_accuracy`: Question-level fraction where predicted top-scoring answer overlaps with gold top answer set.
- `bottom1_set_accuracy`: Question-level fraction where predicted bottom-scoring answer overlaps with gold bottom answer set.
- `questions_with_parseable_answers`: Number of questions that had enough parsed answers for ranking metrics.

- `model`: Evaluated model name.
- `batch_size`: Effective generation batch size used for that model.

## Directionality (quick guide)
- Higher is better: most `*_accuracy`, `*_rate`, precision/recall/F1, `macro_f1`, `balanced_accuracy`, `jaccard_yes`, `mcc`, `score_pearson`, `score_spearman`.
- Lower is better: `rule_hamming_error`, `rule_hamming_error_per_sample`, `rule_weighted_error`, `rule_weighted_error_per_sample`, `score_mae`, `score_rmse`, and `|score_mean_error_bias|`.
