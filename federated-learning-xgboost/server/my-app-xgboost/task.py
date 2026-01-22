"""my-app: A Flower / XGBoost app (binary Diabetes dataset)."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from preprocessing import PreprocessConfig, preprocess_df


BASE_DIR = Path(__file__).resolve().parent

# Default name (only used if you run locally without env var)
CSV_NAME = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

# Prefer dataset path from environment variable (per-client CSV)
env_path = os.getenv("DATASET_PATH")
if env_path:
    DATA_PATH = Path(env_path)
else:
    # Fallback (local run / dev)
    DATA_PATH = BASE_DIR / "datasets" / CSV_NAME

# Cache for loaded data (per-process)
_X = None
_y = None

PREP_CFG = PreprocessConfig(
    target_col="Diabetes_binary",
    clip_outliers=True,
    add_flags=True,
)

# Preprocessing toggle
USE_PREPROCESSING = True  # default: ON


def _load_dataset(csv_path: Path = DATA_PATH):
    """Load Diabetes dataset as binary labels (0/1) from a *single client-local CSV*."""
    global _X, _y

    # Use cache if already loaded
    if _X is not None and _y is not None:
        return _X, _y

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            f"Tip: set DATASET_PATH to your client-local CSV path"
        )

    print(f"[task.py] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna()

    if "Diabetes_binary" not in df.columns:
        raise ValueError("Column 'Diabetes_binary' not found in dataset")

    # OPTIONAL preprocessing
    if USE_PREPROCESSING:
        df = preprocess_df(df, PREP_CFG)

    y = df["Diabetes_binary"].to_numpy(dtype=int)
    X = df.drop(columns=["Diabetes_binary"]).to_numpy(dtype=float)

    _X, _y = X, y
    return _X, _y


def load_data(partition_id: int, num_clients: int):
    """Return XGBoost DMatrices (binary) for THIS client only.

    NOTE:
    - We keep (partition_id, num_clients) in the signature because Flower passes them,
      but we do NOT use them for splitting between clients anymore.
    - Each client gets its own CSV via DATASET_PATH.
    - Inside the client, we split into train/val/test.
    """
    print(f"[task.py] load_data called with partition_id={partition_id}, num_clients={num_clients}")
    print(f"[task.py] Using DATA_PATH={DATA_PATH}")

    X, y = _load_dataset()

    # Stratify if possible
    stratify_all = y if len(np.unique(y)) > 1 else None

    # Split out TEST first
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=stratify_all,
    )

    # Split remaining into TRAIN/VAL
    stratify_tmp = y_tmp if len(np.unique(y_tmp)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.2,  # 20% of remaining -> val
        random_state=42,
        stratify=stratify_tmp,
    )

    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
    test_dmatrix  = xgb.DMatrix(X_test, label=y_test)

    return train_dmatrix, valid_dmatrix, test_dmatrix, len(y_train), len(y_val), len(y_test)


# ===== OLD LOGIC (kept for reference) =====
# Previously i did client partitioning inside task.py using:
# - StratifiedKFold
# - ratio splits (e.g., 0.2/0.1/0.7)
# Now it's unnecessary because each client has its own CSV.
# =========================================


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
