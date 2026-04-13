from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost is required. Install with: pip install xgboost") from exc


# -------------------------
# LOAD DATA
# -------------------------
def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "results": "results.csv",
        "races": "races.csv",
        "qualifying": "qualifying.csv",
        "drivers": "drivers.csv",
    }

    data = {}
    for key, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        data[key] = pd.read_csv(path, na_values=["\\N"])

    return data


# -------------------------
# BUILD DATASET + FEATURES
# -------------------------
def build_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = data["results"].copy()
    races = data["races"][["raceId", "year", "round", "circuitId", "date"]].copy()
    qualifying = data["qualifying"][["raceId", "driverId", "position"]].copy()
    qualifying = qualifying.rename(columns={"position": "quali_position"})
    drivers = data["drivers"][["driverId", "dob"]].copy()

    df = results[["raceId", "driverId", "constructorId", "grid", "positionOrder", "points"]].copy()
    df = df.merge(races, on="raceId", how="left")
    df = df.merge(qualifying, on=["raceId", "driverId"], how="left")
    df = df.merge(drivers, on="driverId", how="left")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    df = df.dropna(subset=["date", "year", "round", "driverId", "constructorId", "positionOrder"])
    df = df.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)

    # Target
    df["win"] = (df["positionOrder"] == 1).astype(int)
    df["finish_pos"] = df["positionOrder"]

    # Group stats
    driver_group = df.groupby("driverId", sort=False)
    constructor_group = df.groupby("constructorId", sort=False)

    df["driver_prior_races"] = driver_group.cumcount()
    df["constructor_prior_races"] = constructor_group.cumcount()

    df["driver_prev_wins"] = driver_group["win"].cumsum() - df["win"]
    df["driver_prev_points"] = driver_group["points"].cumsum() - df["points"]
    df["driver_prev_finish_sum"] = driver_group["finish_pos"].cumsum() - df["finish_pos"]

    df["constructor_prev_wins"] = constructor_group["win"].cumsum() - df["win"]
    df["constructor_prev_points"] = constructor_group["points"].cumsum() - df["points"]
    df["constructor_prev_finish_sum"] = constructor_group["finish_pos"].cumsum() - df["finish_pos"]

    df["driver_prev_avg_finish"] = np.where(
        df["driver_prior_races"] > 0,
        df["driver_prev_finish_sum"] / df["driver_prior_races"],
        np.nan,
    )

    df["constructor_prev_avg_finish"] = np.where(
        df["constructor_prior_races"] > 0,
        df["constructor_prev_finish_sum"] / df["constructor_prior_races"],
        np.nan,
    )

    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

    # Features
    feature_cols = [
        "year", "round", "grid", "quali_position", "driver_age",
        "driver_prior_races", "driver_prev_wins", "driver_prev_points",
        "driver_prev_avg_finish", "constructor_prior_races",
        "constructor_prev_wins", "constructor_prev_points",
        "constructor_prev_avg_finish", "driverId",
        "constructorId", "circuitId",
    ]

    model_df = df[feature_cols + ["win"]].copy()

    numeric_cols = [
        "year", "round", "grid", "quali_position", "driver_age",
        "driver_prior_races", "driver_prev_wins", "driver_prev_points",
        "driver_prev_avg_finish", "constructor_prior_races",
        "constructor_prev_wins", "constructor_prev_points",
        "constructor_prev_avg_finish",
    ]

    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df[numeric_cols] = model_df[numeric_cols].fillna(model_df[numeric_cols].median())
    model_df = model_df.dropna(subset=["driverId", "constructorId", "circuitId", "win"])

    model_df["driverId"] = model_df["driverId"].astype(int).astype(str)
    model_df["constructorId"] = model_df["constructorId"].astype(int).astype(str)
    model_df["circuitId"] = model_df["circuitId"].astype(int).astype(str)

    return model_df


# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_year = int(df["year"].max())
    split_year = max_year - 2

    train_df = df[df["year"] < split_year].copy()
    test_df = df[df["year"] >= split_year].copy()

    if train_df.empty or test_df.empty:
        split_index = int(len(df) * 0.8)
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

    return train_df, test_df


# -------------------------
# EVALUATION
# -------------------------
def evaluate_model(name: str, model: Pipeline,
                   x_train, y_train, x_test, y_test) -> Dict[str, float]:

    model.fit(x_train, y_train)

    if name == "Linear Regression":
        y_score = np.clip(model.predict(x_test), 0.0, 1.0)
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_score = model.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if len(np.unique(y_test)) > 1:
        result["roc_auc"] = roc_auc_score(y_test, y_score)
    else:
        result["roc_auc"] = np.nan

    return result


# -------------------------
# TRAIN + PICK BEST MODEL
# -------------------------
def train_and_compare(model_df: pd.DataFrame):
    train_df, test_df = time_split(model_df)

    y_train = train_df["win"].astype(int)
    y_test = test_df["win"].astype(int)

    x_train = train_df.drop(columns=["win"])
    x_test = test_df.drop(columns=["win"])

    numeric_features = [
        "year","round","grid","quali_position","driver_age",
        "driver_prior_races","driver_prev_wins","driver_prev_points",
        "driver_prev_avg_finish","constructor_prior_races",
        "constructor_prev_wins","constructor_prev_points",
        "constructor_prev_avg_finish"
    ]

    categorical_features = ["driverId", "constructorId", "circuitId"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    models = [
        ("Logistic Regression",
         Pipeline([("preprocessor", preprocessor),
                   ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])),

        ("XGBoost",
         Pipeline([("preprocessor", preprocessor),
                   ("model", XGBClassifier(
                       n_estimators=300,
                       learning_rate=0.05,
                       max_depth=6,
                       subsample=0.9,
                       colsample_bytree=0.9,
                       eval_metric="logloss",
                       random_state=42))])),

        ("Linear Regression",
         Pipeline([("preprocessor", preprocessor),
                   ("model", LinearRegression())]))
    ]

    results = []
    best_model = None
    best_f1 = -1

    for name, model in models:
        metrics = evaluate_model(name, model, x_train, y_train, x_test, y_test)
        results.append(metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model

    comparison_df = pd.DataFrame(results).sort_values(by="f1", ascending=False).reset_index(drop=True)

    return comparison_df, best_model


# -------------------------
# MAIN
# -------------------------
def main():
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "archive (2)"

    data = load_data(data_dir)
    model_df = build_dataset(data)

    comparison_df, best_model = train_and_compare(model_df)

    # Save comparison
    comparison_df.to_csv(project_root / "model_comparison.csv", index=False)

    # Save trained model
    joblib.dump(best_model, project_root / "model.pkl")

    print("Model comparison (sorted by F1):")
    print(comparison_df.to_string(index=False))
    print("\nSaved model as model.pkl")


if __name__ == "__main__":
    main()