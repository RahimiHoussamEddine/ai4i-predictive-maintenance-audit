# Methodology

## Scope

This project audits a synthetic benchmark and a fixed risk-ranking workflow.
It is not a deployed machine-failure model, and it cannot show that alerts or
maintenance would prevent failures. UCI describes the dataset as synthetic,
with generated sensor variables and a binary machine-failure label formed from
five failure modes. Those mode labels are excluded from model features.

## Source and checks

- Fetch the official UCI archive and verify its byte count and SHA-256.
- Confirm the exact 14-column schema, 10,000 rows, unique increasing UDI,
  nonmissing values, and binary target/mode labels.
- Compare the machine-failure target with the row-wise OR of the five
  failure-mode labels and report mismatches. Preserve the source's primary
  target; do not reconstruct or overwrite it from the component labels.
- Use `Type` and five numeric readings as predictors. Exclude `UDI`, `Product
  ID`, the main target, and component failure labels.
- Keep raw data and row-level model scores ignored by Git.

## Evaluation splits

The primary source-order stress split trains on the first 70% of increasing
UDI values and tests on the final 30%. UCI describes the data as multivariate
time-series and describes generated temperature random walks, but UDI is only
a row identifier, not a verified timestamp or persistent asset identifier.
This split therefore tests sensitivity to source-order distribution shifts;
it is not a real future deployment evaluation.

A deterministic stratified 70/30 shuffled holdout is included as a sensitivity.
It can mix rows from different source-order regions and should not be treated
as a stronger estimate of deployment performance.

## Models and measures

Fixed models are a prior-rate baseline, standardized logistic regression, and
a bounded random forest (300 trees, depth 8, minimum leaf size 10). No
hyperparameters are tuned on the test sets. Report average precision,
ROC-AUC, Brier score, and test-set failure prevalence.

For each test set, rank records by predicted risk and report failures captured
at fixed review capacities of 1%, 2%, 5%, and 10%. This is a ranking exercise;
it does not estimate a maintenance intervention effect or failures prevented.
Feature-distribution shifts are summarized as standardized mean differences
for numeric variables and share differences for product type.

## Limits

- The source is synthetic, generated, and not actual factory or fleet telemetry.
- One row is a generated operating point; the data do not provide a verified
  asset-level maintenance timeline or intervention outcomes.
- UDI order is not documented as operational time, and source-order scores do
  not establish future or cross-site performance.
- The failure target is constructed from five component labels; these labels
  are not predictors but their known construction makes high performance on
  this synthetic benchmark insufficient evidence of field usefulness.
- The downloaded CSV has rows where its primary failure target and the union
  of component labels disagree. This mismatch is reported as a source-quality
  finding; the primary target remains authoritative for evaluation.
- Review-capacity results assume a ranking budget only. They do not quantify
  service costs, false-alarm burden over time, or avoided failures.
- No record-level sensors, identifiers, or predicted scores are published.
