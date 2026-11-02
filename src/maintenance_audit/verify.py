"""Validate locked dependencies, source provenance, and aggregate outputs."""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

import pandas as pd

from maintenance_audit.constants import (
    MODE_COLUMNS,
    PRODUCT_ID,
    RAW_DIR,
    REPORTS_DIR,
    SOURCE_ID,
    SOURCE_SHA256,
    TARGET,
)
from maintenance_audit.data import parse_archive, sha256_file

EXPECTED_REPORTS = {
    "RESULTS.md",
    "capacity_metrics.csv",
    "data_quality.json",
    "feature_drift.csv",
    "model_metrics.csv",
    "split_summary.csv",
    "summary.json",
}
DISALLOWED_COLUMNS = {
    SOURCE_ID,
    PRODUCT_ID,
    TARGET,
    *MODE_COLUMNS,
    "prediction",
    "score",
    "y_pred",
}


def _verify_lock(root: Path) -> int:
    lock = root / "requirements.lock"
    if not lock.is_file():
        raise ValueError("requirements.lock is missing")
    count = 0
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line.strip())
        if match is None:
            raise ValueError(f"Unpinned lock entry: {line!r}")
        name, expected = match.groups()
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Locked distribution is not installed: {name}") from error
        if actual != expected:
            raise ValueError(f"Version mismatch for {name}: {expected} != {actual}")
        count += 1
    if count < 20:
        raise ValueError(f"Lock contains too few distributions: {count}")
    return count


def verify(root: Path | None = None) -> dict[str, int | str]:
    root = root or Path(__file__).resolve().parents[2]
    archive = RAW_DIR / "ai4i2020.zip"
    if sha256_file(archive) != SOURCE_SHA256:
        raise ValueError("Raw source checksum does not match the pinned UCI archive")
    frame = parse_archive(archive)
    manifest = json.loads((RAW_DIR.parent / "source_manifest.json").read_text())
    if manifest["archive_sha256"] != SOURCE_SHA256 or manifest["rows"] != len(frame):
        raise ValueError("Source manifest does not match the downloaded archive")

    actual_files = {path.name for path in REPORTS_DIR.iterdir() if path.is_file()}
    if actual_files != EXPECTED_REPORTS:
        raise ValueError(
            f"Unexpected report files: {sorted(actual_files ^ EXPECTED_REPORTS)}"
        )
    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["rows"] != 10_000 or not summary["synthetic"]:
        raise ValueError("Summary must identify the 10,000-row synthetic benchmark")
    if summary["source_ids_used_as_features"]:
        raise ValueError("Source identifiers must be excluded from model features")
    if summary["component_failure_labels_used_as_features"]:
        raise ValueError("Component target labels must be excluded from features")
    mismatch_count = int((frame[MODE_COLUMNS].max(axis=1) != frame[TARGET]).sum())
    if summary["component_label_mismatch_count"] != mismatch_count:
        raise ValueError("Reported component-label mismatch count is incorrect")
    if summary["row_level_scores_published"]:
        raise ValueError("Row-level scores must not be published")

    tables = {
        "model_metrics": pd.read_csv(REPORTS_DIR / "model_metrics.csv"),
        "capacity_metrics": pd.read_csv(REPORTS_DIR / "capacity_metrics.csv"),
        "split_summary": pd.read_csv(REPORTS_DIR / "split_summary.csv"),
        "feature_drift": pd.read_csv(REPORTS_DIR / "feature_drift.csv"),
    }
    for name, table in tables.items():
        exposed = DISALLOWED_COLUMNS.intersection(table.columns)
        if exposed:
            raise ValueError(f"{name} expose disallowed columns: {sorted(exposed)}")
    expected_rows = {
        "model_metrics": 6,
        "capacity_metrics": 24,
        "split_summary": 4,
        "feature_drift": 16,
    }
    for name, expected in expected_rows.items():
        if len(tables[name]) != expected:
            raise ValueError(f"Unexpected {name} row count: {len(tables[name])}")
    lock_count = _verify_lock(root)
    return {
        "source_sha256": SOURCE_SHA256,
        "source_rows": len(frame),
        "models_and_splits": len(tables["model_metrics"]),
        "capacity_rows": len(tables["capacity_metrics"]),
        "locked_distributions": lock_count,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
