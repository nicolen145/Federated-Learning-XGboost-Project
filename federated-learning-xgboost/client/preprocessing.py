# preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class PreprocessConfig:
    target_col: str = "Diabetes_binary"
    clip_outliers: bool = True
    bmi_clip: Tuple[float, float] = (10.0, 60.0)
    days_clip: Tuple[float, float] = (0.0, 30.0)
    add_flags: bool = True


def preprocess_df(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    Deterministic preprocessing only (no fitting, no stats learned from data):
    - Clip outliers for BMI and days columns (MentHlth/PhysHlth)
    - Add simple health-related binary flags
    """
    out = df.copy()

    if cfg.clip_outliers:
        if "BMI" in out.columns:
            out["BMI"] = out["BMI"].clip(cfg.bmi_clip[0], cfg.bmi_clip[1])

        for col in ("MentHlth", "PhysHlth"):
            if col in out.columns:
                out[col] = out[col].clip(cfg.days_clip[0], cfg.days_clip[1])

    if cfg.add_flags:
        if "BMI" in out.columns:
            out["HighBMI"] = (out["BMI"] >= 30).astype(int)
        if "MentHlth" in out.columns:
            out["PoorMentalHealth"] = (out["MentHlth"] >= 15).astype(int)
        if "PhysHlth" in out.columns:
            out["PoorPhysicalHealth"] = (out["PhysHlth"] >= 15).astype(int)

    return out


def split_xy(df: pd.DataFrame, cfg: PreprocessConfig):
    """Split df into X (features) and y (label)."""
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")
    y = df[cfg.target_col].astype(int).values
    X = df.drop(columns=[cfg.target_col]).values
    return X, y
