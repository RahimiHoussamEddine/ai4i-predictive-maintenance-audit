"""Fixed, leakage-aware machine-failure risk models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from maintenance_audit.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_LABELS = {
    "prior": "Prior-rate baseline",
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
}


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_models() -> dict[str, Pipeline | DummyClassifier]:
    return {
        "prior": DummyClassifier(strategy="prior"),
        "logistic": Pipeline(
            [
                ("features", _preprocessor()),
                ("model", LogisticRegression(C=1.0, max_iter=2_000)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=10,
                        max_features=0.8,
                        random_state=2026,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
