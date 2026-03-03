"""
Autoencoder for player state representation and clustering.

Learns a compressed latent representation of player-round states from gold
features, then clusters players in latent space to surface archetypes like:
  - Rising cheapies (low price, upward trajectory)
  - Declining premiums (high price, downward trajectory)
  - Consistent premiums (high price, low variance)
  - Boom-bust players (high variance)

Outputs:
  data/gold/player_states.parquet  — one row per player-round with cluster
                                     label, anomaly score, and latent vector
  models/autoencoder.pt            — trained model weights
  models/autoencoder_scaler.pkl    — fitted StandardScaler
"""

# %%
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger.remove()
logger.add(sys.stdout, level="INFO")

# ── Config ───────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # Score history (lagged — no current-round leakage)
    "score_lag_1", "score_lag_2", "score_lag_3",
    "rolling_avg_3", "rolling_std_3",
    "career_avg", "season_avg_to_date",
    # Price & value signals
    "price", "ppm", "price_momentum_3", "price_change_lag_1",
    "be_gap", "be_gap_lag_1",
    # Play style (what kind of scorer)
    "base_avg", "scoring_avg", "create_avg", "bppm",
    # Matchup context
    "opponent_avg_pts_allowed", "matchup_adjusted_avg",
    # Minutes (proxy for role security)
    "mins_lag_1",
]

IDENTITY_COLS = ["player_name", "year", "round", "position", "team", "price"]

N_LATENT = 8
N_CLUSTERS = 6
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
MIN_GAMES = 3  # require at least this many lag rounds (score_lag_1/2/3 not null)

# ── Load ─────────────────────────────────────────────────────────────────────

logger.info("Loading gold features...")
df = pl.read_parquet("data/gold/player_rounds_features.parquet")
df = df.filter(pl.col("score_lag_3").is_not_null())
logger.info(f"Rows after filtering score_lag_3 not null (>= {MIN_GAMES} prior games): {len(df):,}")

# ── Feature prep ─────────────────────────────────────────────────────────────

# One-hot encode primary position
positions = df["primary_position"].fill_null("UNK").to_list()
unique_positions = sorted(set(positions))
pos_map = {p: i for i, p in enumerate(unique_positions)}

pos_encoded = np.zeros((len(df), len(unique_positions)), dtype=np.float32)
for i, p in enumerate(positions):
    pos_encoded[i, pos_map[p]] = 1.0

# Numeric features → numpy, impute nulls with column median
X_numeric = df.select(NUMERIC_FEATURES).to_numpy().astype(np.float32)
for j in range(X_numeric.shape[1]):
    col = X_numeric[:, j]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        X_numeric[nan_mask, j] = float(np.nanmedian(col))

# Combine and scale
X = np.hstack([X_numeric, pos_encoded])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
X_tensor = torch.from_numpy(X_scaled)

INPUT_DIM = X_scaled.shape[1]
logger.info(f"Feature matrix: {X.shape}  ({len(NUMERIC_FEATURES)} numeric + {len(unique_positions)} position)")

# ── Model ─────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


model = Autoencoder(INPUT_DIM, N_LATENT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ── Train ─────────────────────────────────────────────────────────────────────

logger.info(f"Training autoencoder ({EPOCHS} epochs, batch={BATCH_SIZE}, latent={N_LATENT})...")

dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for (batch,) in loader:
        optimizer.zero_grad()
        reconstructed, _ = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        logger.info(f"  Epoch {epoch + 1:3d}/{EPOCHS} — loss: {total_loss / len(loader):.4f}")

# ── Extract latent vectors + anomaly scores ───────────────────────────────────

logger.info("Extracting latent representations...")
model.eval()
with torch.no_grad():
    reconstructed, latent = model(X_tensor)
    reconstruction_error = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()
    latent_np = latent.numpy()

# ── Cluster ───────────────────────────────────────────────────────────────────

logger.info(f"Clustering into {N_CLUSTERS} clusters (k-means)...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(latent_np)

# ── Output dataframe ──────────────────────────────────────────────────────────

df_out = df.select(IDENTITY_COLS + ["score", "be_gap", "games_played"]).with_columns(
    [
        pl.Series("cluster", cluster_labels),
        pl.Series("anomaly_score", reconstruction_error),
        *[pl.Series(f"latent_{i}", latent_np[:, i]) for i in range(N_LATENT)],
    ]
)

# ── Cluster profiles ──────────────────────────────────────────────────────────

logger.info("\n--- Cluster profiles ---")
logger.info(f"  {'Cluster':<10} {'n':>6} {'price':>8} {'score':>7} {'be_gap':>8}")
logger.info(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
for c in sorted(df_out["cluster"].unique().to_list()):
    subset = df_out.filter(pl.col("cluster") == c)
    logger.info(
        f"  {c:<10} {len(subset):>6,} "
        f"{subset['price'].mean():>8,.0f} "
        f"{subset['score'].cast(pl.Float64).mean():>7.1f} "
        f"{subset['be_gap'].cast(pl.Float64).mean():>8.1f}"
    )

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = Path("data/gold/player_states.parquet")
df_out.write_parquet(out_path)
logger.info(f"\nSaved: {out_path}  ({len(df_out):,} rows)")

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/autoencoder.pt")
joblib.dump(scaler, "models/autoencoder_scaler.pkl")
joblib.dump(kmeans, "models/autoencoder_kmeans.pkl")
joblib.dump({"pos_map": pos_map, "unique_positions": unique_positions, "numeric_features": NUMERIC_FEATURES}, "models/autoencoder_meta.pkl")
logger.info("Saved model artefacts: models/autoencoder.pt, autoencoder_scaler.pkl, autoencoder_kmeans.pkl, autoencoder_meta.pkl")
