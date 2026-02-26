# %%
import os
import argparse
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

load_dotenv(Path(__file__).parent.parent / ".env")

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lightgbm", choices=["xgboost", "lightgbm"])
parser.add_argument("--description", type=str, default="Position-specific models")
parser.add_argument("--changes", type=str, default="None")
args = parser.parse_args()

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

mlflow.set_tracking_uri(os.getenv("AZURE_ML_TRACKING_URI"))
mlflow.set_experiment("nrl-supercoach-by-position")

# %%
df = pl.read_parquet(Path(__file__).parent.parent / "data/gold/player_rounds_features.parquet")

df = df.sort(["player_name", "year", "round"])
df = df.with_columns(
    pl.col("score")
      .shift(-1)
      .over(["player_name", "year"])
      .alias("target_score")
)
df = df.filter(pl.col("target_score").is_not_null())
df = df.filter(pl.col("mins") >= 5)

df = df.with_columns([
    pl.col("primary_position").cast(pl.Categorical).to_physical().alias("primary_position_enc"),
    pl.col("team").cast(pl.Categorical).to_physical().alias("team_enc"),
    pl.col("opponent").cast(pl.Categorical).to_physical().alias("opponent_enc"),
])

DROP_COLS = [
    "target_score", "score", "player_name", "team", "opponent",
    "position", "primary_position", "secondary_position",
]

feature_cols = [c for c in df.columns if c not in DROP_COLS and c != "year" and c != "round"]

positions = df["primary_position_enc"].unique().to_list()
position_map = (
    df.select(["primary_position", "primary_position_enc"])
    .unique()
    .to_pandas()
    .set_index("primary_position_enc")["primary_position"]
    .to_dict()
)

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

results = []

for pos_enc in sorted(positions):
    pos_name = position_map[pos_enc]
    print(f"\n--- Training {pos_name} ---")

    pos_df = df.filter(pl.col("primary_position_enc") == pos_enc)

    train = pos_df.filter(pl.col("year") <= 2023)
    val   = pos_df.filter(pl.col("year") == 2024)
    test  = pos_df.filter(pl.col("year") == 2025)

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["target_score"].to_numpy()
    X_val   = val.select(feature_cols).to_numpy()
    y_val   = val["target_score"].to_numpy()
    X_test  = test.select(feature_cols).to_numpy()
    y_test  = test["target_score"].to_numpy()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    with mlflow.start_run(run_name=f"{args.model}-{pos_name}"):
        mlflow.log_params(params)
        mlflow.set_tag("position", pos_name)
        mlflow.set_tag("model", args.model)
        mlflow.set_tag("run_description", args.description)
        mlflow.set_tag("changes", args.changes)

        if args.model == "xgboost":
            model = XGBRegressor(**params, eval_metric="rmse", early_stopping_rounds=20)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif args.model == "lightgbm":
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(False)],
            )

        val_mae  = mean_absolute_error(y_val, model.predict(X_val))
        val_rmse = root_mean_squared_error(y_val, model.predict(X_val))
        test_mae  = mean_absolute_error(y_test, model.predict(X_test))
        test_rmse = root_mean_squared_error(y_test, model.predict(X_test))

        mlflow.log_metrics({
            "val_mae": val_mae, "val_rmse": val_rmse,
            "test_mae": test_mae, "test_rmse": test_rmse,
        })

        # Feature importance
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
        ax.set_title(f"{args.model.upper()} {pos_name} — Top 20 Features")
        plt.tight_layout()
        plot_path = f"feature_importance_{pos_name.replace('/', '_')}.png"
        plt.savefig(plot_path, dpi=150)
        mlflow.log_artifact(plot_path)
        plt.close()

        if args.model == "xgboost":
            model.save_model(f"model_{pos_name}.json")
            mlflow.log_artifact(f"model_{pos_name}.json")
        elif args.model == "lightgbm":
            model.booster_.save_model(f"model_{pos_name}.txt")
            mlflow.log_artifact(f"model_{pos_name}.txt")

        print(f"{pos_name} — Val MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f} | Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
        results.append({"position": pos_name, "val_mae": val_mae, "val_rmse": val_rmse, "test_mae": test_mae, "test_rmse": test_rmse})

# %%
results_df = pd.DataFrame(results).sort_values("val_mae")
print("\n--- Summary ---")
print(results_df.to_string(index=False))