"""Source-order and shuffled holdouts with capacity-limited review metrics."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from maintenance_audit.constants import (
    CAPACITIES,
    CATEGORICAL_FEATURES,
    EVALUATIONS,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    SOURCE_ID,
    TARGET,
)
from maintenance_audit.modeling import MODEL_LABELS, make_models

EVALUATION_LABELS = {
    "source_order": "Source-order holdout",
    "shuffled": "Stratified shuffled holdout",
}


def split_indices(
    frame: pd.DataFrame, evaluation: str
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    if evaluation == "source_order":
        ordered = np.argsort(frame[SOURCE_ID].to_numpy())
        cutoff = int(0.70 * len(ordered))
        return ordered[:cutoff], ordered[cutoff:]
    if evaluation == "shuffled":
        return train_test_split(
            indices,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=frame[TARGET].to_numpy(),
        )
    raise ValueError(f"Unknown evaluation mode: {evaluation}")


def _classification_metrics(actual: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(actual, score)),
        "roc_auc": float(roc_auc_score(actual, score)),
        "brier_score": float(brier_score_loss(actual, score)),
        "failure_rate": float(np.mean(actual)),
    }


def capacity_metrics(
    actual: np.ndarray,
    score: np.ndarray,
    capacities: list[float] | None = None,
) -> list[dict[str, float | int]]:
    capacities = capacities or CAPACITIES
    order = np.argsort(-score, kind="stable")
    total_failures = int(actual.sum())
    results: list[dict[str, float | int]] = []
    for capacity in capacities:
        selected_count = max(1, math.ceil(capacity * len(actual)))
        selected = order[:selected_count]
        captured = int(actual[selected].sum())
        results.append(
            {
                "capacity": capacity,
                "n_reviewed": selected_count,
                "failures_captured": captured,
                "precision": captured / selected_count,
                "recall": captured / total_failures if total_failures else 0.0,
                "test_failure_count": total_failures,
            }
        )
    return results


def _feature_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    evaluation: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in NUMERIC_FEATURES:
        train_mean = float(train[column].mean())
        test_mean = float(test[column].mean())
        train_sd = float(train[column].std(ddof=1))
        rows.append(
            {
                "evaluation": evaluation,
                "feature": column,
                "measure": "standardized_mean_difference",
                "train_value": train_mean,
                "test_value": test_mean,
                "difference": (test_mean - train_mean) / train_sd if train_sd else 0.0,
            }
        )
    categories = sorted(
        set(train[CATEGORICAL_FEATURES[0]].unique())
        | set(test[CATEGORICAL_FEATURES[0]].unique())
    )
    for category in categories:
        train_share = float((train["Type"] == category).mean())
        test_share = float((test["Type"] == category).mean())
        rows.append(
            {
                "evaluation": evaluation,
                "feature": f"Type={category}",
                "measure": "share_difference",
                "train_value": train_share,
                "test_value": test_share,
                "difference": test_share - train_share,
            }
        )
    return rows


def evaluate(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    y = frame[TARGET].to_numpy(dtype=int)
    x = frame[FEATURES]

    for evaluation in EVALUATIONS:
        train_index, test_index = split_indices(frame, evaluation)
        train_frame = frame.iloc[train_index]
        test_frame = frame.iloc[test_index]
        train_x, test_x = x.iloc[train_index], x.iloc[test_index]
        train_y, test_y = y[train_index], y[test_index]
        split_rows.extend(
            [
                {
                    "evaluation": evaluation,
                    "split": "train",
                    "n": len(train_y),
                    "failure_count": int(train_y.sum()),
                    "failure_rate": float(train_y.mean()),
                },
                {
                    "evaluation": evaluation,
                    "split": "test",
                    "n": len(test_y),
                    "failure_count": int(test_y.sum()),
                    "failure_rate": float(test_y.mean()),
                },
            ]
        )
        drift_rows.extend(_feature_drift(train_frame, test_frame, evaluation))

        for model_name, estimator in make_models().items():
            estimator.fit(train_x, train_y)
            score = estimator.predict_proba(test_x)[:, 1]
            model_rows.append(
                {
                    "evaluation": evaluation,
                    "model": model_name,
                    "n_test": len(test_y),
                    "failure_count": int(test_y.sum()),
                    **_classification_metrics(test_y, score),
                }
            )
            for row in capacity_metrics(test_y, score):
                capacity_rows.append(
                    {"evaluation": evaluation, "model": model_name, **row}
                )

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(capacity_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(drift_rows),
    )


def _write_results(
    metrics: pd.DataFrame,
    capacity: pd.DataFrame,
    splits: pd.DataFrame,
    quality: dict[str, object],
) -> None:
    display = metrics.copy()
    display["model_label"] = display["model"].map(MODEL_LABELS)
    display["evaluation_label"] = display["evaluation"].map(EVALUATION_LABELS)
    source_order = display[display["evaluation"] == "source_order"].set_index("model")
    best_model = source_order["average_precision"].idxmax()
    block = splits[
        (splits["evaluation"] == "source_order") & (splits["split"] == "test")
    ].iloc[0]
    capacity_rows = capacity[
        (capacity["evaluation"] == "source_order") & (capacity["capacity"] == 0.05)
    ].copy()
    lines = [
        "# Results: synthetic predictive-maintenance audit",
        "",
        (
            f"The UCI source contains **{quality['rows']:,} synthetic rows** and "
            f"{quality['machine_failure_count']} machine-failure labels "
            f"({quality['machine_failure_rate']:.2%}). It is a generated "
            "benchmark, not field telemetry."
        ),
        "",
        "## Locked test-set metrics",
        "",
        "| Split | Model | AP | ROC-AUC | Brier | Failure rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_label} | {row.model_label} | "
            f"{row.average_precision:.3f} | {row.roc_auc:.3f} | "
            f"{row.brier_score:.4f} | {row.failure_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            (
                f"On the source-order test block ({int(block['n']):,} rows, "
                f"{int(block['failure_count'])} failures), the fixed "
                f"{MODEL_LABELS[best_model]} had the highest average precision "
                f"({source_order.loc[best_model, 'average_precision']:.3f}). "
                "This benchmark score does not establish performance on real "
                "equipment."
            ),
            "",
            "## Five-percent review-capacity illustration",
            "",
            "| Model | Alerts reviewed | Failures captured | Recall | Precision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in capacity_rows.itertuples(index=False):
        lines.append(
            f"| {MODEL_LABELS[row.model]} | {row.n_reviewed} | "
            f"{row.failures_captured} | {row.recall:.2%} | "
            f"{row.precision:.2%} |"
        )
    lines.extend(
        [
            "",
            (
                "The capacity result ranks locked test rows at a fixed review "
                "budget. The budget is set independently of test labels; this "
                "does not estimate failures prevented by maintenance."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "The source-order holdout preserves increasing UDI, while UCI "
                "describes generated temperature signals as random walks. "
                "UDI is not a verified timestamp or persistent machine "
                "identifier, so this is a source-order stress test rather "
                "than a validated future deployment simulation. The "
                "stratified shuffled split is a sensitivity and can mix "
                "source-order neighborhoods."
            ),
            "",
            (
                "UCI documents the primary target as the union of five "
                "component-mode labels, but the downloaded CSV has "
                f"{quality['component_label_mismatch_count']} disagreements. "
                "The audit preserves the primary target and excludes all "
                "component labels, UDI, and Product ID from model inputs. "
                "The synthetic results cannot support real maintenance "
                "decisions without field validation and intervention evidence."
            ),
            "",
            (
                "Reports contain only aggregate metrics and split summaries. "
                "Row-level sensor records and predicted scores are not "
                "published."
            ),
            "",
            (
                "See [methodology](../docs/METHODOLOGY.md), "
                "[data dictionary](../docs/DATA_DICTIONARY.md), and "
                "[source notice](../NOTICE.md)."
            ),
            "",
        ]
    )
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(
    frame: pd.DataFrame,
    quality: dict[str, object],
) -> dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics, capacity, splits, drift = evaluate(frame)
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False, float_format="%.8f")
    capacity.to_csv(
        REPORTS_DIR / "capacity_metrics.csv", index=False, float_format="%.8f"
    )
    splits.to_csv(REPORTS_DIR / "split_summary.csv", index=False, float_format="%.8f")
    drift.to_csv(REPORTS_DIR / "feature_drift.csv", index=False, float_format="%.8f")
    (REPORTS_DIR / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    summary: dict[str, object] = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "synthetic": True,
        "rows": len(frame),
        "target": TARGET,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "source_ids_used_as_features": False,
        "component_failure_labels_used_as_features": False,
        "component_label_mismatch_count": quality["component_label_mismatch_count"],
        "evaluations": EVALUATION_LABELS,
        "metrics": metrics.to_dict(orient="records"),
        "capacity_fractions": CAPACITIES,
        "row_level_scores_published": False,
    }
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_results(metrics, capacity, splits, quality)
    return summary
