"""Aggregate-only Streamlit dashboard."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from maintenance_audit.constants import REPORTS_DIR
from maintenance_audit.evaluation import EVALUATION_LABELS
from maintenance_audit.modeling import MODEL_LABELS

st.set_page_config(page_title="AI4I risk audit", layout="wide")
st.title("Synthetic predictive-maintenance risk audit")
st.caption(
    "AI4I is a generated benchmark. These scores do not establish performance "
    "on real equipment or estimate maintenance effects."
)

summary_path = REPORTS_DIR / "summary.json"
if not summary_path.exists():
    st.info("Build the report first with `make build`.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
capacity = pd.read_csv(REPORTS_DIR / "capacity_metrics.csv")
quality = json.loads((REPORTS_DIR / "data_quality.json").read_text(encoding="utf-8"))

if quality["component_label_mismatch_count"]:
    st.warning(
        f"The source file has {quality['component_label_mismatch_count']} "
        "disagreements between its primary target and component labels. "
        "The audit keeps the primary target and excludes the component labels."
    )

left, middle, right = st.columns(3)
left.metric("Synthetic records", f"{summary['rows']:,}")
middle.metric("Machine failures", quality["machine_failure_count"])
right.metric("Overall failure rate", f"{quality['machine_failure_rate']:.2%}")

metric_options = {
    "average_precision": "Average precision",
    "roc_auc": "ROC-AUC",
    "brier_score": "Brier score (lower is better)",
}
metric = st.selectbox(
    "Test metric",
    list(metric_options),
    format_func=metric_options.get,
)
plot_data = metrics.copy()
plot_data["model_label"] = plot_data["model"].map(MODEL_LABELS)
plot_data["evaluation_label"] = plot_data["evaluation"].map(EVALUATION_LABELS)
metric_chart = px.bar(
    plot_data,
    x="model_label",
    y=metric,
    color="evaluation_label",
    barmode="group",
    labels={
        "model_label": "Model",
        metric: metric_options[metric],
        "evaluation_label": "Split",
    },
    title="Locked test-set performance",
)
st.plotly_chart(metric_chart, width="stretch")

evaluation = st.selectbox(
    "Evaluation split",
    list(EVALUATION_LABELS),
    format_func=EVALUATION_LABELS.get,
)
capacity_data = capacity[capacity["evaluation"] == evaluation].copy()
capacity_data["model_label"] = capacity_data["model"].map(MODEL_LABELS)
capacity_data["review_percent"] = capacity_data["capacity"] * 100
capacity_chart = px.line(
    capacity_data,
    x="review_percent",
    y="recall",
    color="model_label",
    markers=True,
    labels={
        "review_percent": "Records reviewed (%)",
        "recall": "Failures captured (%)",
        "model_label": "Model",
    },
    title="Failure capture at fixed review capacity",
)
capacity_chart.update_yaxes(tickformat=".0%")
st.plotly_chart(capacity_chart, width="stretch")

st.subheader("Source and split quality")
split_summary = pd.read_csv(REPORTS_DIR / "split_summary.csv")
split_summary["evaluation"] = split_summary["evaluation"].map(EVALUATION_LABELS)
st.dataframe(split_summary, hide_index=True, width="stretch")
st.markdown(
    "UDI is a source row identifier, not a verified timestamp or persistent "
    "machine ID. Capacity results measure ranking only; they do not estimate "
    "failures prevented by maintenance."
)
