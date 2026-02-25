# %%
import sys
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
import mlflow
import mlflow.xgboost
import mlflow.lightgbm

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
GOLD_DIR = Path("data/gold")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = list(range(2015, 2024))
VAL_YEAR = 2024
TEST_YEAR = 2025

MLFLOW_EXPERIMENT = "nrl_supercoach_score_prediction"

# %%
# Current round features — not available before the game
EXCLUDE_COLS = [
    # Target
    "score",
    # Identifiers
    "player_name",
    # Site-calculated season aggregates that leak future rounds
    "season_price_change", "avg_score",
    "avg_2rd", "avg_3rd", "avg_5rd",
    # Current round action stats
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

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
    "eval_metric": "mae",
}

LGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


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
    """Get feature columns — exclude target, identifiers, and leaky columns."""
    return [c for c in df.columns if c not in EXCLUDE_COLS + ["year"]]


# %%
def prepare_splits(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, validation, and test sets by year."""
    train = df.filter(pl.col("year").is_in(TRAIN_YEARS)).to_pandas()
    val = df.filter(pl.col("year") == VAL_YEAR).to_pandas()
    test = df.filter(pl.col("year") == TEST_YEAR).to_pandas()

    logger.info(f"Train: {len(train)} rows ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})")
    logger.info(f"Val:   {len(val)} rows ({VAL_YEAR})")
    logger.info(f"Test:  {len(test)} rows ({TEST_YEAR})")

    return train, val, test


# %%
def evaluate(model, X, y, label: str, position_col: pd.Series = None) -> tuple[float, float, np.ndarray]:
    """Evaluate model — returns MAE, RMSE, and predictions."""
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    logger.info(f"{label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    if position_col is not None:
        for pos in sorted(position_col.unique()):
            if pos is None:
                continue
            mask = position_col == pos
            if mask.sum() < 10:
                continue
            pos_mae = mean_absolute_error(y[mask], preds[mask])
            pos_rmse = root_mean_squared_error(y[mask], preds[mask])
            logger.info(f"  {str(pos):6s} — MAE: {pos_mae:.2f}, RMSE: {pos_rmse:.2f}")

    return mae, rmse, preds


# %%
def run_training():
    """Full training pipeline with MLflow tracking."""

    # Set up MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_data()
    df = df.filter(pl.col("score").is_not_null())

    train, val, test = prepare_splits(df)

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

    with mlflow.start_run(run_name="xgb_lgb_ensemble"):

        # Log parameters
        mlflow.log_params({f"xgb_{k}": v for k, v in XGB_PARAMS.items()})
        mlflow.log_params({f"lgb_{k}": v for k, v in LGB_PARAMS.items()})
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_years", f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
        mlflow.log_param("val_year", VAL_YEAR)
        mlflow.log_param("test_year", TEST_YEAR)
        mlflow.log_param("train_rows", len(X_train))

        # Train XGBoost
        logger.info("\n--- Training XGBoost ---")
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # Train LightGBM
        logger.info("\n--- Training LightGBM ---")
        lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
        )

        # Evaluate
        logger.info("\n--- XGBoost ---")
        xgb_val_mae, xgb_val_rmse, xgb_val_preds = evaluate(xgb_model, X_val, y_val, "Val", val["primary_position"])
        xgb_test_mae, xgb_test_rmse, xgb_test_preds = evaluate(xgb_model, X_test, y_test, "Test", test["primary_position"])

        logger.info("\n--- LightGBM ---")
        lgb_val_mae, lgb_val_rmse, lgb_val_preds = evaluate(lgb_model, X_val, y_val, "Val", val["primary_position"])
        lgb_test_mae, lgb_test_rmse, lgb_test_preds = evaluate(lgb_model, X_test, y_test, "Test", test["primary_position"])

        # Ensemble
        ens_val_preds = 0.5 * xgb_val_preds + 0.5 * lgb_val_preds
        ens_test_preds = 0.5 * xgb_test_preds + 0.5 * lgb_test_preds
        ens_val_mae = mean_absolute_error(y_val, ens_val_preds)
        ens_val_rmse = root_mean_squared_error(y_val, ens_val_preds)
        ens_test_mae = mean_absolute_error(y_test, ens_test_preds)
        ens_test_rmse = root_mean_squared_error(y_test, ens_test_preds)

        logger.info(f"\n--- Ensemble (50/50) ---")
        logger.info(f"Val  — MAE: {ens_val_mae:.2f}, RMSE: {ens_val_rmse:.2f}")
        logger.info(f"Test — MAE: {ens_test_mae:.2f}, RMSE: {ens_test_rmse:.2f}")

        # Log metrics
        mlflow.log_metrics({
            "xgb_val_mae": xgb_val_mae,
            "xgb_val_rmse": xgb_val_rmse,
            "xgb_test_mae": xgb_test_mae,
            "xgb_test_rmse": xgb_test_rmse,
            "lgb_val_mae": lgb_val_mae,
            "lgb_val_rmse": lgb_val_rmse,
            "lgb_test_mae": lgb_test_mae,
            "lgb_test_rmse": lgb_test_rmse,
            "ensemble_val_mae": ens_val_mae,
            "ensemble_val_rmse": ens_val_rmse,
            "ensemble_test_mae": ens_test_mae,
            "ensemble_test_rmse": ens_test_rmse,
        })

        # Log feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "xgb_importance": xgb_model.feature_importances_,
            "lgb_importance": lgb_model.feature_importances_,
        }).sort_values("xgb_importance", ascending=False)

        importance_path = MODELS_DIR / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        # Log models
        mlflow.xgboost.log_model(xgb_model, "xgb_model")
        mlflow.lightgbm.log_model(lgb_model, "lgb_model")

        # Save locally too
        joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
        joblib.dump(lgb_model, MODELS_DIR / "lgb_model.pkl")
        joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
        joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

        logger.success(f"Run logged to MLflow experiment: {MLFLOW_EXPERIMENT}")
        logger.success("Models saved to data/models/")

    return xgb_model, lgb_model, encoders, feature_cols


# %%
run_training()