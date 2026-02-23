# %%
import sys
import json
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import joblib

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
GOLD_DIR = Path("data/gold")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = list(range(2015, 2024))  # 2015-2023
VAL_YEAR = 2024
TEST_YEAR = 2025

# Features to use for training
# Exclude target, identifiers, and any columns that would leak future info
EXCLUDE_COLS = [
    "score",                # target
    "player_name",          # identifier
    "team",                 # will encode separately
    "opponent",             # will encode separately
    "position",             # will encode separately
    "primary_position",     # will encode separately
    "secondary_position",   # will encode separately
    "season_price_change",  # leaks full season info
    "avg_score",            # leaks future rounds
    "avg_2rd",              # site-calculated, may leak
    "avg_3rd",              # site-calculated, may leak
    "avg_5rd",              # site-calculated, may leak
    "be_gap",
    "ppm",
    "avg_be_gap_3rd",
    "base",
    "attack", 
    "playmaking",
    "power",
    "negative",
    "base_power",
    "base_avg",
    "scoring_avg",
    "create_avg",
    "evade_avg",
    "negative_avg",
    "base_power_avg",
    "base_pct",
    "base_power_ppm",
    "h8_pct",
    "tb_pct",
    "mt_pct",
    "ol_il_pct",
    # Current round action stats — not available before the game
    "TR", "TS", "LT", "GO", "MG", "FG", "MF", "TA", "MT",
    "TB", "FD", "OL", "IO", "LB", "LA", "FT", "KB", "H8",
    "HU", "HG", "IT", "KD", "PC", "ER", "SS",
    "mins", "bppm", "cv",
    # Current round scoring breakdown
    "base", "attack", "playmaking", "power", "negative",
    "base_power", "base_avg", "scoring_avg", "create_avg",
    "evade_avg", "negative_avg", "base_power_avg", "base_pct",
    "base_power_ppm", "h8_pct", "tb_pct", "mt_pct", "ol_il_pct",
    # Other current-round leakers
    "be_gap", "avg_be_gap_3rd", "ppm", "avg_pc", "avg_er",
    "avg_pc_er", "sixty_sixty", "score_rank_in_position",
]


# %%
def load_data() -> pl.DataFrame:
    """Load Gold feature dataset."""
    path = GOLD_DIR / "player_rounds_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Gold parquet not found at {path}")
    df = pl.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# %%
def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """Label encode categorical columns."""
    cat_cols = ["team", "opponent", "position", "primary_position", "secondary_position"]
    
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].fillna("unknown").astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


# %%
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns to use for training."""
    return [c for c in df.columns if c not in EXCLUDE_COLS + ["year"]]


# %%
def prepare_splits(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    train = df.filter(pl.col("year").is_in(TRAIN_YEARS)).to_pandas()
    val = df.filter(pl.col("year") == VAL_YEAR).to_pandas()
    test = df.filter(pl.col("year") == TEST_YEAR).to_pandas()

    logger.info(f"Train: {len(train)} rows ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})")
    logger.info(f"Val:   {len(val)} rows ({VAL_YEAR})")
    logger.info(f"Test:  {len(test)} rows ({TEST_YEAR})")

    return train, val, test


# %%
def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    """Train XGBoost model."""
    logger.info("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


# %%
def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
    """Train LightGBM model."""
    logger.info("Training LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    return model


# %%
def evaluate(model, X, y, label: str, position_col: pd.Series = None):
    """Evaluate model performance overall and per position."""
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    logger.info(f"{label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    if position_col is not None:
        results = {}
        for pos in position_col.unique():
            mask = position_col == pos
            if mask.sum() < 10:
                continue
            pos_mae = mean_absolute_error(y[mask], preds[mask])
            pos_rmse = root_mean_squared_error(y[mask], preds[mask])
            results[pos] = {"mae": round(pos_mae, 2), "rmse": round(pos_rmse, 2)}
            logger.info(f"  {str(pos):6s} — MAE: {pos_mae:.2f}, RMSE: {pos_rmse:.2f}")

    return preds


# %%
def ensemble_predict(xgb_preds, lgb_preds, xgb_weight=0.5) -> np.ndarray:
    """Blend XGBoost and LightGBM predictions."""
    return xgb_weight * xgb_preds + (1 - xgb_weight) * lgb_preds


# %%
def run_training():
    """Full training pipeline."""
    df = load_data()

    # Drop rows with null scores (DNPs)
    df = df.filter(pl.col("score").is_not_null())

    train, val, test = prepare_splits(df)

    # Encode categoricals
    train, encoders = encode_categoricals(train, fit=True)
    val, _ = encode_categoricals(val, encoders=encoders, fit=False)
    test, _ = encode_categoricals(test, encoders=encoders, fit=False)

    feature_cols = get_feature_cols(train)
    logger.info(f"Training with {len(feature_cols)} features")

    X_train = train[feature_cols].fillna(-999)
    y_train = train["score"]
    X_val = val[feature_cols].fillna(-999)
    y_val = val["score"]
    X_test = test[feature_cols].fillna(-999)
    y_test = test["score"]

    # Train models
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

    # Evaluate individually
    logger.info("\n--- XGBoost ---")
    xgb_val_preds = evaluate(xgb_model, X_val, y_val, "Val", val["primary_position"])
    xgb_test_preds = evaluate(xgb_model, X_test, y_test, "Test", test["primary_position"])

    logger.info("\n--- LightGBM ---")
    lgb_val_preds = evaluate(lgb_model, X_val, y_val, "Val", val["primary_position"])
    lgb_test_preds = evaluate(lgb_model, X_test, y_test, "Test", test["primary_position"])

    # Evaluate ensemble
    logger.info("\n--- Ensemble (50/50) ---")
    ens_val_preds = ensemble_predict(xgb_val_preds, lgb_val_preds)
    ens_test_preds = ensemble_predict(xgb_test_preds, lgb_test_preds)
    ens_val_mae = mean_absolute_error(y_val, ens_val_preds)
    ens_test_mae = mean_absolute_error(y_test, ens_test_preds)
    logger.info(f"Val  — MAE: {ens_val_mae:.2f}")
    logger.info(f"Test — MAE: {ens_test_mae:.2f}")

    # Save models and encoders
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(lgb_model, MODELS_DIR / "lgb_model.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    logger.success("Models saved to data/models/")

    # Save feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "xgb_importance": xgb_model.feature_importances_,
        "lgb_importance": lgb_model.feature_importances_,
    }).sort_values("xgb_importance", ascending=False)

    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    logger.success("Feature importance saved")

    return xgb_model, lgb_model, encoders, feature_cols


# %%
run_training()