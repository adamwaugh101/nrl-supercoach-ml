# %%
import os
import polars as pl
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import argparse
import pandas as pd
import matplotlib.pyplot as plt


load_dotenv(Path(__file__).parent.parent / ".env")

# %%
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
)

ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
)

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm"])
parser.add_argument("--description", type=str, default="No description provided")
parser.add_argument("--changes", type=str, default="None")
args = parser.parse_args()

mlflow.set_tracking_uri(os.getenv("AZURE_ML_TRACKING_URI"))
mlflow.set_experiment("nrl-supercoach-score-prediction")
mlflow.set_tag("run_description", args.description)
mlflow.set_tag("changes", args.changes)
mlflow.set_tag("data_version", "gold_v1")

# %%
df = pl.read_parquet(Path(__file__).parent.parent / "data/gold/player_rounds_features.parquet")
print(f"Loaded: {df.shape}")

# %%
# Shift score forward by 1 within each player+year group to create target
df = df.sort(["player_name", "year", "round"])

df = df.with_columns(
    pl.col("score")
      .shift(-1)
      .over(["player_name", "year"])
      .alias("target_score")
)

# Drop last round of each season (no next round to predict)
df = df.filter(pl.col("target_score").is_not_null())

print(f"After target shift: {df.shape}")
# %%
# Filter out non-playing rows (injured, rested, DNP)
pre_filter = df.shape[0]
df = df.filter(pl.col("mins") >= 5)
post_filter = df.shape[0]
print(f"Filtered {pre_filter - post_filter} low-minute rows ({pre_filter} → {post_filter})")

# %%
# Columns to drop from features
DROP_COLS = [
    "target_score",
    "score",           # current round score — would be leakage
    "player_name",
    "team",
    "opponent",
    "position",
    "primary_position",
    "secondary_position",
    "ground_condition",
    "weather_condition",
]

# Encode categorical columns before dropping
df = df.with_columns([
    pl.col("position").cast(pl.Categorical).to_physical().alias("position_enc"),
    pl.col("primary_position").cast(pl.Categorical).to_physical().alias("primary_position_enc"),
    pl.col("team").cast(pl.Categorical).to_physical().alias("team_enc"),
    pl.col("opponent").cast(pl.Categorical).to_physical().alias("opponent_enc"),
    # --- add these ---
    pl.col("is_home").cast(pl.Int8),
    pl.col("is_bye").cast(pl.Int8),
    pl.col("ground_condition").cast(pl.Categorical).to_physical().alias("ground_condition_enc"),
    pl.col("weather_condition").cast(pl.Categorical).to_physical().alias("weather_condition_enc"),
])

# %%
# Time-based train/val/test split
train = df.filter(pl.col("year") <= 2023)
val   = df.filter(pl.col("year") == 2024)
test  = df.filter(pl.col("year") == 2025)

feature_cols = [c for c in df.columns if c not in DROP_COLS and c != "year" and c != "round"]

X_train = train.select(feature_cols).to_numpy()
y_train = train["target_score"].to_numpy()

X_val = val.select(feature_cols).to_numpy()
y_val = val["target_score"].to_numpy()

X_test = test.select(feature_cols).to_numpy()
y_test = test["target_score"].to_numpy()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %%
params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

mlflow.end_run()

with mlflow.start_run(run_name=f"{args.model}-baseline"):
    mlflow.log_params(params)


    if args.model == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(**params, eval_metric="rmse", early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    elif args.model == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
        )

    val_preds = model.predict(X_val)
    val_mae  = mean_absolute_error(y_val, val_preds)
    val_rmse = root_mean_squared_error(y_val, val_preds)

    test_preds = model.predict(X_test)
    test_mae  = mean_absolute_error(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)

    mlflow.log_metrics({
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    })

    if args.model == "xgboost":
        model.save_model("model.json")
        mlflow.log_artifact("model.json")
    elif args.model == "lightgbm":
        model.booster_.save_model("model.txt")
        mlflow.log_artifact("model.txt")

    print(f"Val  — MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
    print(f"Test — MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")

    # Baseline comparisons
    val_raw = val.select(["target_score", "score_lag_1", "rolling_avg_3"]).to_pandas()

    baseline_last_week_mae  = mean_absolute_error(val_raw["target_score"], val_raw["score_lag_1"].fillna(0))
    baseline_last_week_rmse = root_mean_squared_error(val_raw["target_score"], val_raw["score_lag_1"].fillna(0))

    baseline_rolling_mae  = mean_absolute_error(val_raw["target_score"], val_raw["rolling_avg_3"].fillna(0))
    baseline_rolling_rmse = root_mean_squared_error(val_raw["target_score"], val_raw["rolling_avg_3"].fillna(0))

    print(f"Baseline last week  — MAE: {baseline_last_week_mae:.2f}, RMSE: {baseline_last_week_rmse:.2f}")
    print(f"Baseline 3rd avg    — MAE: {baseline_rolling_mae:.2f}, RMSE: {baseline_rolling_rmse:.2f}")
    print(f"{args.model.capitalize()} (val)       — MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")

    mlflow.log_metrics({
        "baseline_last_week_mae": baseline_last_week_mae,
        "baseline_last_week_rmse": baseline_last_week_rmse,
        "baseline_rolling_mae": baseline_rolling_mae,
        "baseline_rolling_rmse": baseline_rolling_rmse,
    })    

    if args.model == "xgboost":
        importance = model.feature_importances_
    elif args.model == "lightgbm":
        importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(feature_importance_df["feature"][::-1], feature_importance_df["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"{args.model.capitalize()} — Top 30 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    mlflow.log_artifact("feature_importance.png")
    plt.show()

    print(feature_importance_df.to_string(index=False))