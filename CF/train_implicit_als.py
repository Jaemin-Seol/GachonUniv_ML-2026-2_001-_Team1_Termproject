# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# Implicit ALS Model Training
#   1. Load training interactions and create user/item ID mappings
#   2. Build a sparse user-item matrix using preference scores
#   3. Initialize and train the Implicit ALS model
#   4. Save the trained factors, ID mappings, and hyperparameters
#
# Inputs  : sgd_train.csv
# Outputs : implicit_als_model.npz


import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


# =========================
# 1. Hyperparameters
# =========================

SEED = 42
LATENT_FACTORS = 64
REGULARIZATION = 0.01
ALPHA = 40
ITERATIONS = 20


# =========================
# 2. Load training data
# =========================

# Load the prepared training interactions
train_df = pd.read_csv("sgd_train.csv")

print("Dataset loaded")
print("Train interactions:", len(train_df))

print("\nColumns:")
print(train_df.columns.tolist())


# =========================
# 3. Create user/item mappings
# =========================

# Convert original IDs into consecutive integer indices
user_ids = np.sort(train_df["steam_id"].unique())
item_ids = np.sort(train_df["item_id"].unique())

user_index = pd.Index(user_ids).get_indexer(train_df["steam_id"])
item_index = pd.Index(item_ids).get_indexer(train_df["item_id"])

num_users = len(user_ids)
num_items = len(item_ids)

print("\nUsers:", num_users)
print("Items:", num_items)


# =========================
# 4. Build user-item matrix
# =========================

# Use preference scores as confidence weights for observed interactions
confidence_values = train_df["preference_score"].to_numpy(
    dtype=np.float32
)

user_items = coo_matrix(
    (
        confidence_values,
        (user_index, item_index)
    ),
    shape=(num_users, num_items),
    dtype=np.float32
).tocsr()

print("\nUser-item matrix created")
print("Matrix shape:", user_items.shape)
print("Non-zero interactions:", user_items.nnz)


# =========================
# 5. Initialize Implicit ALS
# =========================

model = AlternatingLeastSquares(
    factors=LATENT_FACTORS,
    regularization=REGULARIZATION,
    alpha=ALPHA,
    iterations=ITERATIONS,
    random_state=SEED,
    num_threads=-1
)


# =========================
# 6. Train model
# =========================

print("\n========== ALS TRAINING START ==========")

# Fit the model using the user-item interaction matrix
model.fit(
    user_items,
    show_progress=True
)

print("\n========== ALS TRAINING COMPLETE ==========")


# =========================
# 7. Save model
# =========================

# Save learned factors, ID mappings, and training hyperparameters
np.savez_compressed(
    "implicit_als_model.npz",

    user_factors=model.user_factors,
    item_factors=model.item_factors,

    user_ids=user_ids,
    item_ids=item_ids,

    latent_factors=LATENT_FACTORS,
    regularization=REGULARIZATION,
    alpha=ALPHA,
    iterations=ITERATIONS,
    seed=SEED
)

print("\nSaved: implicit_als_model.npz")

print("User factors shape:", model.user_factors.shape)
print("Item factors shape:", model.item_factors.shape)
