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
import optuna
from optuna.integration.mlflow import MLflowCallback
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

load_dotenv(Path(__file__).parent.parent / ".env")

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--description", type=str, default="Optuna tuning for volatile positions")
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
mlflow.set_experiment("nrl-supercoach-tuning")

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

TUNE_POSITIONS = ["CTW", "HFB", "FLB"]

# %%
mlflow.end_run()
results = []

for pos_name in TUNE_POSITIONS:
    print(f"\n--- Tuning {pos_name} ---")

    pos_df = df.filter(pl.col("primary_position") == pos_name)

    train = pos_df.filter(pl.col("year") <= 2023)
    val   = pos_df.filter(pl.col("year") == 2024)
    test  = pos_df.filter(pl.col("year") == 2025)

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["target_score"].to_numpy()
    X_val   = val.select(feature_cols).to_numpy()
    y_val   = val["target_score"].to_numpy()
    X_test  = test.select(feature_cols).to_numpy()
    y_test  = test["target_score"].to_numpy()

    with mlflow.start_run(run_name=f"tuning-{pos_name}"):
        mlflow.set_tag("position", pos_name)
        mlflow.set_tag("run_description", args.description)
        mlflow.set_tag("changes", args.changes)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(False)],
            )

            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)

            with mlflow.start_run(run_name=f"trial-{pos_name}-{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("val_mae", mae)

            return mae

        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=args.trials)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1

        print(f"Best params: {best_params}")
        print(f"Best val MAE: {study.best_value:.2f}")

        # Retrain with best params
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(False)],
        )

        val_mae  = mean_absolute_error(y_val, final_model.predict(X_val))
        val_rmse = root_mean_squared_error(y_val, final_model.predict(X_val))
        test_mae  = mean_absolute_error(y_test, final_model.predict(X_test))
        test_rmse = root_mean_squared_error(y_test, final_model.predict(X_test))

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({
            "val_mae": val_mae, "val_rmse": val_rmse,
            "test_mae": test_mae, "test_rmse": test_rmse,
        })

        final_model.booster_.save_model(f"model_tuned_{pos_name}.txt")
        mlflow.log_artifact(f"model_tuned_{pos_name}.txt")

        print(f"{pos_name} tuned — Val MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f} | Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
        results.append({"position": pos_name, "val_mae": val_mae, "val_rmse": val_rmse, "test_mae": test_mae, "test_rmse": test_rmse})

# %%
results_df = pd.DataFrame(results).sort_values("val_mae")
print("\n--- Tuning Summary ---")
print(results_df.to_string(index=False))