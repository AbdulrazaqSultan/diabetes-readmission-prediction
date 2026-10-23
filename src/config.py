"""Project configuration: paths, constants, feature groups."""
from pathlib import Path

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target
TARGET_COL = "readmitted"
POSITIVE_LABEL = "<30"  # readmitted within 30 days

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PATH = DATA_RAW_DIR / "diabetic_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Cleaning
DEATH_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}   # deceased -> cannot be readmitted
HIGH_MISSING_COLS = ["weight", "payer_code", "medical_specialty"]  # >40% missing
ID_COLS = ["encounter_id", "patient_nbr"]        # ids; patient_nbr kept for GroupSplit

# Feature groups (post-cleaning, raw names)
NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL_LOW_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
