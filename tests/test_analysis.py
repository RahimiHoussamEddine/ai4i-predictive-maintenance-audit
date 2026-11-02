"""Tests for source integrity, split boundaries, ranking, and API outputs."""

import asyncio

import httpx
import numpy as np

from maintenance_audit.api import app
from maintenance_audit.constants import (
    FEATURES,
    MODE_COLUMNS,
    PRODUCT_ID,
    RAW_DIR,
    SOURCE_ID,
    TARGET,
)
from maintenance_audit.data import parse_archive
from maintenance_audit.evaluation import capacity_metrics, split_indices


def test_source_parser_and_leakage_exclusions():
    frame = parse_archive(RAW_DIR / "ai4i2020.zip")
    assert frame.shape == (10_000, 14)
    assert frame[SOURCE_ID].is_monotonic_increasing
    assert frame[SOURCE_ID].is_unique
    assert not frame.isna().any().any()
    assert int((frame[MODE_COLUMNS].max(axis=1) != frame[TARGET]).sum()) == 27
    assert SOURCE_ID not in FEATURES
    assert PRODUCT_ID not in FEATURES
    assert TARGET not in FEATURES
    assert not set(MODE_COLUMNS).intersection(FEATURES)


def test_source_order_split_is_forward_and_disjoint():
    frame = parse_archive(RAW_DIR / "ai4i2020.zip")
    train, test = split_indices(frame, "source_order")
    assert len(train) == 7_000
    assert len(test) == 3_000
    assert frame.iloc[train][SOURCE_ID].max() < frame.iloc[test][SOURCE_ID].min()
    assert not set(train).intersection(test)


def test_shuffled_split_is_disjoint_and_preserves_prevalence():
    frame = parse_archive(RAW_DIR / "ai4i2020.zip")
    train, test = split_indices(frame, "shuffled")
    assert len(train) == 7_000
    assert len(test) == 3_000
    assert not set(train).intersection(test)
    assert abs(frame.iloc[train][TARGET].mean() - frame[TARGET].mean()) < 0.002
    assert abs(frame.iloc[test][TARGET].mean() - frame[TARGET].mean()) < 0.002


def test_capacity_metrics_known_values():
    actual = np.array([1, 0, 1, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    result = capacity_metrics(actual, score, capacities=[0.4])[0]
    assert result["n_reviewed"] == 2
    assert result["failures_captured"] == 1
    assert result["precision"] == 0.5
    assert np.isclose(result["recall"], 1 / 3)


def test_api_only_returns_aggregate_artifacts():
    async def request_api():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            summary = await client.get("/summary")
            assert summary.json()["synthetic"] is True
            models = await client.get("/models")
            assert len(models.json()) == 6
            capacity = await client.get(
                "/review-capacity",
                params={"evaluation": "source_order", "model": "random_forest"},
            )
            assert capacity.status_code == 200
            assert len(capacity.json()) == 4
            assert "prediction" not in capacity.json()[0]
            split = await client.get("/split-summary")
            assert len(split.json()) == 4
            unknown = await client.get(
                "/review-capacity",
                params={"evaluation": "invented", "model": "random_forest"},
            )
            assert unknown.status_code == 404

    asyncio.run(request_api())
