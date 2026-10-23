"""Data loading and cleaning for the UCI Diabetes 130-US Hospitals dataset."""
import pandas as pd

from src.config import DATA_PATH, DEATH_DISCHARGE_IDS, HIGH_MISSING_COLS

INVALID_GENDER = "Unknown/Invalid"


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. '?' encodes missing values in this dataset."""
    return pd.read_csv(path, na_values="?", keep_default_na=True,
                       low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 5 mandatory cleaning steps from the project spec.

    Engineering decision (documented): ALL encounters per patient are kept,
    so GroupShuffleSplit on patient_nbr is essential (same patient appears
    up to 40 times -> random split would leak). This yields 99,340 rows
    with an 11.4% positive rate; the spec's "69k-72k rows" range matches
    the UNIQUE PATIENT count (69,987), not the encounter count.
    """
    n0 = len(df)
    # 1. Drop duplicate encounters
    df = df.drop_duplicates(subset="encounter_id")
    # 2. Drop invalid gender rows (3 rows)
    df = df[df["gender"] != INVALID_GENDER]
    # 3. Drop deceased patients (cannot be readmitted)
    df = df[~df["discharge_disposition_id"].isin(DEATH_DISCHARGE_IDS)]
    # 4. Drop columns with severe missingness (>40%)
    df = df.drop(columns=HIGH_MISSING_COLS)
    # 5. Keep patient_nbr (GroupSplit groups / leakage check)
    df = df.reset_index(drop=True)
    print(f"clean_data: {n0:,} -> {len(df):,} rows "
          f"({n0 - len(df):,} removed)")
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """Binary target: 1 if readmitted within 30 days, else 0."""
    return (df["readmitted"] == "<30").astype(int)


def load_and_clean(path=DATA_PATH) -> pd.DataFrame:
    return clean_data(load_raw(path))
