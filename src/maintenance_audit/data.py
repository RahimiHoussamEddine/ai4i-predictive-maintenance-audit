"""Source parsing, schema checks, and aggregate audit summary."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from maintenance_audit.constants import (
    ALL_COLUMNS,
    FEATURES,
    MODE_COLUMNS,
    NUMERIC_FEATURES,
    PRODUCT_ID,
    RAW_DIR,
    SOURCE_BYTES,
    SOURCE_DOI,
    SOURCE_ID,
    SOURCE_MEMBER,
    SOURCE_SHA256,
    SOURCE_URL,
    TARGET,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_archive(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "ai4i2020.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run `make fetch` first")
    if path.stat().st_size != SOURCE_BYTES or sha256_file(path) != SOURCE_SHA256:
        raise ValueError("UCI archive differs from the pinned source snapshot")
    with zipfile.ZipFile(path) as archive:
        if SOURCE_MEMBER not in archive.namelist():
            raise ValueError(f"Expected CSV is missing: {SOURCE_MEMBER}")
        payload = archive.read(SOURCE_MEMBER)
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if frame.columns.tolist() != ALL_COLUMNS:
        raise ValueError(f"Unexpected UCI columns: {frame.columns.tolist()}")
    for column in [SOURCE_ID, *NUMERIC_FEATURES, TARGET, *MODE_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["Type"] = frame["Type"].astype("string")
    frame[PRODUCT_ID] = frame[PRODUCT_ID].astype("string")
    if len(frame) != 10_000:
        raise ValueError(f"Expected 10,000 records, received {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("UCI source contains missing values")
    if frame[SOURCE_ID].duplicated().any():
        raise ValueError("Source UDI must be unique")
    if not frame[SOURCE_ID].is_monotonic_increasing:
        raise ValueError("Source order is not monotonic by UDI")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Machine failure target must be binary")
    if any(not set(frame[column].unique()).issubset({0, 1}) for column in MODE_COLUMNS):
        raise ValueError("Component failure labels must be binary")
    return frame


def data_quality_report(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    failure_count = int(frame[TARGET].sum())
    component_union = frame[MODE_COLUMNS].max(axis=1)
    component_mismatch_count = int((component_union != frame[TARGET]).sum())
    return {
        "source": {
            "url": SOURCE_URL,
            "archive": "ai4i2020.zip",
            "member": SOURCE_MEMBER,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "doi": SOURCE_DOI,
            "license": "CC BY 4.0",
        },
        "synthetic_benchmark": True,
        "rows": len(frame),
        "feature_count": len(FEATURES),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_udi": int(frame[SOURCE_ID].duplicated().sum()),
        "duplicate_product_id": int(frame[PRODUCT_ID].duplicated().sum()),
        "duplicate_feature_rows": int(frame[FEATURES].duplicated().sum()),
        "machine_failure_count": failure_count,
        "machine_failure_rate": float(frame[TARGET].mean()),
        "component_failure_counts": {
            column: int(frame[column].sum()) for column in MODE_COLUMNS
        },
        "component_label_mismatch_count": component_mismatch_count,
        "component_labels_reconstruct_machine_failure": component_mismatch_count == 0,
        "modeling": {
            "source_ids_used_as_features": False,
            "component_failure_labels_used_as_features": False,
            "row_level_scores_published": False,
        },
    }


def write_source_manifest(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    manifest = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "source_url": SOURCE_URL,
        "doi": SOURCE_DOI,
        "citation": (
            "AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). "
            "UCI Machine Learning Repository."
        ),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "synthetic": True,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "expected_sha256": SOURCE_SHA256,
        "member": SOURCE_MEMBER,
        "rows": len(frame),
        "features": len(FEATURES),
    }
    (RAW_DIR.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
