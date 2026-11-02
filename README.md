# AI4I Synthetic Predictive-Maintenance Risk Audit

An auditable risk-ranking study using the UCI AI4I 2020 predictive-maintenance
dataset. It compares a prior-rate baseline, logistic regression, and a bounded
random forest on a source-order holdout and a stratified shuffled sensitivity.

**This dataset is synthetic.** Its scores measure performance on a generated
benchmark, not real industrial equipment or maintenance outcomes. The project
does not estimate the causal effect of inspection or maintenance.

## Key result

On the source-order test block, the fixed random forest reached average
precision **0.697** and ROC-AUC **0.976**. A 5% review budget captured 50 of 61
failure-labeled rows (82.0% recall, 33.3% precision). These are synthetic
benchmark results only.

## Run locally

Use Python 3.12:

~~~bash
python3 -m venv .venv
make install-dev
make fetch
make build
make verify
make test
make quality
make check-deps
make app
~~~

The models exclude UDI, Product ID, and the five component failure-mode labels.
The main source-order split trains on the first 70% of UDI order and evaluates
on the final 30%; because UDI is not a verified timestamp or persistent asset
identifier, this is a stress test rather than a real forecast validation. A
stratified shuffled split is shown as a sensitivity.

The downloaded CSV has **27 rows** where its primary machine-failure target
disagrees with the union of the five component labels, despite the source
documentation describing that union. The audit reports the discrepancy and
keeps the primary target as supplied.

Reports and API responses expose only test-set metrics, split summaries, and
capacity aggregates. Raw rows and row-level scores stay local and are ignored
by Git. Dataset and code licenses are documented separately.

## API

~~~text
GET /health
GET /summary
GET /models
GET /review-capacity?evaluation=source_order&model=random_forest
GET /split-summary
~~~

Run with `make api`; launch the dashboard with `make app`. See
`reports/RESULTS.md`, `docs/METHODOLOGY.md`, and `NOTICE.md` for results,
limitations, and source credit.
