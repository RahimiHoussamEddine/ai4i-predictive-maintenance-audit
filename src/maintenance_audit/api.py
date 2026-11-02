"""Read-only API for aggregate evaluation artifacts."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from maintenance_audit.constants import EVALUATIONS, REPORTS_DIR
from maintenance_audit.modeling import MODEL_LABELS

app = FastAPI(
    title="AI4I Synthetic Predictive-Maintenance Audit API",
    version="0.1.0",
    description="Aggregate-only risk ranking and capacity results.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    ready = (REPORTS_DIR / "summary.json").is_file()
    return {"status": "ok" if ready else "not_ready"}


@app.get("/summary")
async def summary() -> dict[str, object]:
    path = REPORTS_DIR / "summary.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/models")
async def models() -> list[dict[str, object]]:
    path = REPORTS_DIR / "model_metrics.csv"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/review-capacity")
async def review_capacity(
    evaluation: str = Query(...),
    model: str = Query(...),
) -> list[dict[str, object]]:
    path = REPORTS_DIR / "capacity_metrics.csv"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    if evaluation not in EVALUATIONS or model not in MODEL_LABELS:
        raise HTTPException(
            status_code=404,
            detail={"evaluations": EVALUATIONS, "models": sorted(MODEL_LABELS)},
        )
    table = pd.read_csv(path)
    selected = table[(table["evaluation"] == evaluation) & (table["model"] == model)]
    return selected.to_dict(orient="records")


@app.get("/split-summary")
async def split_summary() -> list[dict[str, object]]:
    path = REPORTS_DIR / "split_summary.csv"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    return pd.read_csv(path).to_dict(orient="records")
