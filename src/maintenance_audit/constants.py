"""Pinned source details and modeling contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
SOURCE_SHA256 = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
SOURCE_BYTES = 522_170
SOURCE_MEMBER = "ai4i2020.csv"
SOURCE_DOI = "10.24432/C5HS5C"
SOURCE_ID = "UDI"
PRODUCT_ID = "Product ID"
TARGET = "Machine failure"
MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
ALL_COLUMNS = [
    SOURCE_ID,
    PRODUCT_ID,
    *FEATURES,
    TARGET,
    *MODE_COLUMNS,
]
RANDOM_STATE = 2026
CAPACITIES = [0.01, 0.02, 0.05, 0.10]
EVALUATIONS = ["source_order", "shuffled"]
