# %%
import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
import mlflow
from pathlib import Path

if os.path.isfile(Path(__file__).parent.parent / ".env"):
    print("Loading .env file")
    load_dotenv(Path(__file__).parent.parent / ".env")
else:
    print(".env file not found")

print(f"TENANT: {os.getenv('AZURE_TENANT_ID')}")
print(f"CLIENT: {os.getenv('AZURE_CLIENT_ID')}")
print(f"SECRET: {os.getenv('AZURE_CLIENT_SECRET')[:5] if os.getenv('AZURE_CLIENT_SECRET') else 'None'}")

# %%
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
)

ml_client = MLClient(
    credential=credential,
    subscription_id="b90ab5b7-3c61-467e-a0b5-d2ddaccbfe0b",
    resource_group_name="nrl-supercoach-rg",
    workspace_name="nrl-supercoach-ml",
)

# %%
mlflow.set_tracking_uri(os.getenv("AZURE_ML_TRACKING_URI"))
mlflow.set_experiment("nrl-supercoach-connection-test")

with mlflow.start_run():
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 1.0)
    print("MLflow run logged successfully")

# %%
print(f"Workspace: {ml_client.workspace_name}")
print("Connection test passed!")