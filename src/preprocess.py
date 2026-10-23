"""Feature preparation and preprocessing pipeline (ColumnTransformer).

Engineering decisions (documented):
- ICD-9 diag codes are grouped into 9 clinical chapters (spec default):
  interpretable, low-cardinality, robust to unseen codes.
- age is mapped to the 10-year-bin midpoint (ordinal -> numeric).
- Drug columns stay categorical with OneHotEncoder (spec default).
- Low-cardinality ids (admission/discharge/source) are cast to str so the
  whole categorical block is string-typed (also required for ONNX export).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_LOW_COLS,
    DIAG_COLS,
    DRUG_COLS,
    ID_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)

# age "[a-b)" -> midpoint
AGE_MIDPOINTS = {f"[{a}-{a+10})": a + 5 for a in range(0, 100, 10)}

DRUG_VALUES = ["No", "Steady", "Down", "Up"]  # implicit dose-order (spec)

# ICD-9-CM chapter ranges (inclusive), checked in order
_ICD9_CHAPTERS = [
    ("Diabetes", lambda p: p == "250" or p.startswith("250")),
    ("Neoplasms", lambda p: p.isdigit() and 140 <= int(p) <= 239),
    ("Circulatory", lambda p: p.isdigit() and 390 <= int(p) <= 459),
    ("Respiratory", lambda p: p.isdigit() and 460 <= int(p) <= 519),
    ("Digestive", lambda p: p.isdigit() and 520 <= int(p) <= 579),
    ("Genitourinary", lambda p: p.isdigit() and 580 <= int(p) <= 629),
    ("Musculoskeletal", lambda p: p.isdigit() and 710 <= int(p) <= 739),
    ("Injury", lambda p: (p.isdigit() and 800 <= int(p) <= 999)
                         or p.upper().startswith("E")),
]


def map_icd9_to_chapter(code) -> str:
    """Map an ICD-9 code (e.g. '250.01', 'V45', 'E888') to a chapter name."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Other"
    s = str(code).strip().upper()
    if s in {"", "?", "NAN", "NONE"}:
        return "Other"
    prefix = s.split(".")[0]
    for name, match in _ICD9_CHAPTERS:
        if match(prefix):
            return name
    return "Other"


def map_age(value):
    """Map age bin '[a-b)' to midpoint; pass through numerics and NaN."""
    if value in AGE_MIDPOINTS:
        return AGE_MIDPOINTS[value]
    return np.nan


def map_drug_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize drug columns to the 4 canonical values (NaN -> 'No')."""
    df = df.copy()
    for col in DRUG_COLS:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), "No")
            df[col] = df[col].astype(str).str.strip()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return X (raw feature space): ICD-9 chapters, age midpoints,
    drug normalization, string categoricals. Drops ids and target."""
    df = df.copy()
    # ICD-9 -> chapters
    for col in DIAG_COLS:
        df[col] = df[col].map(map_icd9_to_chapter)
    # age -> midpoint (numeric)
    df["age"] = df["age"].map(map_age)
    # drugs
    df = map_drug_columns(df)
    # cast low-cardinality categoricals to string (incl. id-like codes)
    for col in CATEGORICAL_LOW_COLS:
        if col == "age":
            continue
        df[col] = df[col].where(df[col].notna(), "Missing").astype(str)
    for col in DIAG_COLS + DRUG_COLS:
        df[col] = df[col].astype(str)
    # drop ids + target
    drop = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    return df.drop(columns=drop)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: numeric / cat-low / cat-high pipelines."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_low_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cat_high_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Other")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS + ["age"]),
        ("cat_low", cat_low_pipeline, CATEGORICAL_LOW_COLS),
        ("cat_high", cat_high_pipeline, DIAG_COLS + DRUG_COLS),
    ], verbose_feature_names_out=False)
