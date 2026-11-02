# Data dictionary

## UCI source columns

| Source column | Use | Handling |
|---|---|---|
| UDI | Source ordering and checks only | Excluded from predictors |
| Product ID | Identifier | Excluded from predictors |
| Type | Predictor | One-hot encoded product variant |
| Air temperature [K] | Predictor | Numeric |
| Process temperature [K] | Predictor | Numeric |
| Rotational speed [rpm] | Predictor | Numeric |
| Torque [Nm] | Predictor | Numeric |
| Tool wear [min] | Predictor | Numeric |
| Machine failure | Primary target | Binary; never a predictor |
| TWF, HDF, PWF, OSF, RNF | Component target labels | Used for consistency checks only; excluded from predictors |

The five component labels represent tool-wear, heat-dissipation, power,
overstrain, and random failures. UCI states that the primary target is their
union, but the downloaded CSV contains 27 rows where the primary target and
the component-label union disagree. The audit reports the mismatch and keeps
`Machine failure` as the primary target. See `data/source_manifest.json` for
the dataset citation and license.

## Aggregate reports

- `reports/model_metrics.csv`: test-set metrics by split and model.
- `reports/capacity_metrics.csv`: captured failures and precision/recall at
  fixed test-set review fractions.
- `reports/split_summary.csv`: row counts and failure prevalence by split.
- `reports/feature_drift.csv`: aggregate train/test feature differences.
- `reports/summary.json`: modeling scope and headline aggregate metrics.
- `reports/data_quality.json`: aggregate source and target checks.
- `reports/RESULTS.md`: narrative results and limitations.
