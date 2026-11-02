# Results: synthetic predictive-maintenance audit

The UCI source contains **10,000 synthetic rows** and 339 machine-failure labels (3.39%). It is a generated benchmark, not field telemetry.

## Locked test-set metrics

| Split | Model | AP | ROC-AUC | Brier | Failure rate |
|---|---|---:|---:|---:|---:|
| Source-order holdout | Prior-rate baseline | 0.020 | 0.500 | 0.0203 | 2.03% |
| Source-order holdout | Logistic regression | 0.415 | 0.892 | 0.0158 | 2.03% |
| Source-order holdout | Random forest | 0.697 | 0.976 | 0.0102 | 2.03% |
| Stratified shuffled holdout | Prior-rate baseline | 0.034 | 0.500 | 0.0328 | 3.40% |
| Stratified shuffled holdout | Logistic regression | 0.399 | 0.890 | 0.0254 | 3.40% |
| Stratified shuffled holdout | Random forest | 0.842 | 0.984 | 0.0124 | 3.40% |

On the source-order test block (3,000 rows, 61 failures), the fixed Random forest had the highest average precision (0.697). This benchmark score does not establish performance on real equipment.

## Five-percent review-capacity illustration

| Model | Alerts reviewed | Failures captured | Recall | Precision |
|---|---:|---:|---:|---:|
| Prior-rate baseline | 150 | 3 | 4.92% | 2.00% |
| Logistic regression | 150 | 38 | 62.30% | 25.33% |
| Random forest | 150 | 50 | 81.97% | 33.33% |

The capacity result ranks locked test rows at a fixed review budget. The budget is set independently of test labels; this does not estimate failures prevented by maintenance.

## Interpretation limits

The source-order holdout preserves increasing UDI, while UCI describes generated temperature signals as random walks. UDI is not a verified timestamp or persistent machine identifier, so this is a source-order stress test rather than a validated future deployment simulation. The stratified shuffled split is a sensitivity and can mix source-order neighborhoods.

UCI documents the primary target as the union of five component-mode labels, but the downloaded CSV has 27 disagreements. The audit preserves the primary target and excludes all component labels, UDI, and Product ID from model inputs. The synthetic results cannot support real maintenance decisions without field validation and intervention evidence.

Reports contain only aggregate metrics and split summaries. Row-level sensor records and predicted scores are not published.

See [methodology](../docs/METHODOLOGY.md), [data dictionary](../docs/DATA_DICTIONARY.md), and [source notice](../NOTICE.md).
