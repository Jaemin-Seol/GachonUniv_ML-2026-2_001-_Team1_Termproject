# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# Implicit ALS Hyperparameter Tuning - Round 3
#   1. Define Alpha and Regularization configurations for comparison
#   2. Build the user-item matrix and prepare validation data
#   3. Train and evaluate each configuration using Precision@K and Recall@K
#   4. Select the best model based on Recall@20
#   5. Save the best model, recommendations, and experiment results
#
# Inputs  : sgd_train.csv, sgd_validation.csv
# Outputs : implicit_als_round3_results.csv,
#           implicit_als_best_round3.npz,
#           implicit_als_best_round3_recommendations.csv


import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


# ============================================================
# 1. Experiment settings
# ============================================================

SEED = 42
ITERATIONS = 20
FACTORS = 64
K_VALUES = [10, 20]
MAX_K = 20
BATCH_SIZE = 256

# Refine Alpha around 5 and test Regularization.
CONFIGS = [
    {"name": "a3_r0005", "alpha": 3, "regularization": 0.005},
    {"name": "a3_r001",  "alpha": 3, "regularization": 0.01},
    {"name": "a3_r002",  "alpha": 3, "regularization": 0.02},
    {"name": "a4_r0005", "alpha": 4, "regularization": 0.005},
    {"name": "a4_r001",  "alpha": 4, "regularization": 0.01},
    {"name": "a4_r002",  "alpha": 4, "regularization": 0.02},
    {"name": "a5_r0005", "alpha": 5, "regularization": 0.005},
    {"name": "a5_r001",  "alpha": 5, "regularization": 0.01},
    {"name": "a5_r002",  "alpha": 5, "regularization": 0.02},
    {"name": "a6_r0005", "alpha": 6, "regularization": 0.005},
    {"name": "a6_r001",  "alpha": 6, "regularization": 0.01},
    {"name": "a6_r002",  "alpha": 6, "regularization": 0.02},
    {"name": "a8_r0005", "alpha": 8, "regularization": 0.005},
    {"name": "a8_r001",  "alpha": 8, "regularization": 0.01},
    {"name": "a8_r002",  "alpha": 8, "regularization": 0.02},
]


# ============================================================
# 2. Load data and build the user-item matrix
# ============================================================

# Load training and validation interactions.
train_df = pd.read_csv("sgd_train.csv")
valid_df = pd.read_csv("sgd_validation.csv")

print("Dataset loaded")
print("Train interactions:", len(train_df))
print("Validation interactions:", len(valid_df))

# Create mappings from original IDs to consecutive matrix indices.
user_ids = np.sort(train_df["steam_id"].unique())
item_ids = np.sort(train_df["item_id"].unique())
user_to_idx = {v: i for i, v in enumerate(user_ids)}
item_to_idx = {v: i for i, v in enumerate(item_ids)}

# Convert interaction IDs into matrix indices.
user_index = pd.Index(user_ids).get_indexer(train_df["steam_id"])
item_index = pd.Index(item_ids).get_indexer(train_df["item_id"])

# Build a sparse user-item matrix using preference scores as confidence weights.
user_items = coo_matrix(
    (train_df["preference_score"].to_numpy(dtype=np.float32),
     (user_index, item_index)),
    shape=(len(user_ids), len(item_ids)),
    dtype=np.float32
).tocsr()


# ============================================================
# 3. Prepare training and validation interaction sets
# ============================================================

# Store previously interacted items for each user.
train_df["user_idx"] = user_index
train_df["item_idx"] = item_index
train_seen = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

# Keep only validation interactions whose users and games exist in the training data.
valid_eval = valid_df[
    valid_df["steam_id"].isin(user_to_idx)
    & valid_df["item_id"].isin(item_to_idx)
].copy()

# Convert validation IDs into matrix indices.
valid_eval["user_idx"] = valid_eval["steam_id"].map(user_to_idx)
valid_eval["item_idx"] = valid_eval["item_id"].map(item_to_idx)

# Build the set of relevant validation games for each user.
valid_relevant = valid_eval.groupby("user_idx")["item_idx"].apply(set).to_dict()
eval_users = np.array(sorted(valid_relevant), dtype=np.int64)

print("User-item matrix:", user_items.shape)
print("Non-zero interactions:", user_items.nnz)
print("Evaluation users:", len(eval_users))


# ============================================================
# 4. Train and evaluate each configuration
# ============================================================

all_results = []
best_recall20 = -1.0
best_config = None

for config in CONFIGS:
    print("\n" + "=" * 60)
    print("Configuration:", config)

    # Initialize and train ALS with the current hyperparameter configuration.
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=config["regularization"],
        alpha=config["alpha"],
        iterations=ITERATIONS,
        random_state=SEED,
        num_threads=-1
    )
    model.fit(user_items, show_progress=True)

    # Initialize metric accumulators and recommendation records.
    precision_sum = {k: 0.0 for k in K_VALUES}
    recall_sum = {k: 0.0 for k in K_VALUES}
    rec_rows = []

    # Evaluate users in batches to reduce memory usage.
    for start in range(0, len(eval_users), BATCH_SIZE):
        batch_users = eval_users[start:start + BATCH_SIZE]

        # Calculate prediction scores for all games in the catalog.
        scores = model.user_factors[batch_users] @ model.item_factors.T

        # Exclude games that each user has already interacted with.
        for row, user_idx in enumerate(batch_users):
            seen = train_seen.get(int(user_idx), set())
            if seen:
                scores[row, list(seen)] = -np.inf

        # Select the top-K candidate games for each user.
        top_indices = np.argpartition(-scores, kth=MAX_K - 1, axis=1)[:, :MAX_K]

        for row, user_idx in enumerate(batch_users):
            selected = top_indices[row]
            order = np.argsort(-scores[row, selected])
            recommended = selected[order]
            relevant = valid_relevant[int(user_idx)]

            # Accumulate Precision@K and Recall@K.
            for k in K_VALUES:
                hits = len(set(recommended[:k]) & relevant)
                precision_sum[k] += hits / k
                recall_sum[k] += hits / len(relevant)

            # Store the ranked recommendation list for the current configuration.
            for rank, item_idx in enumerate(recommended, start=1):
                rec_rows.append({
                    "config": config["name"],
                    "steam_id": user_ids[user_idx],
                    "rank": rank,
                    "item_id": item_ids[item_idx],
                    "predicted_score": float(scores[row, item_idx])
                })

        if (start // BATCH_SIZE) % 50 == 0:
            print(f"Evaluated {min(start + BATCH_SIZE, len(eval_users))} / {len(eval_users)} users")

    # Calculate macro-averaged evaluation metrics across all evaluation users.
    n = len(eval_users)
    result = {
        "config": config["name"],
        "factors": FACTORS,
        "regularization": config["regularization"],
        "alpha": config["alpha"],
        "iterations": ITERATIONS
    }
    for k in K_VALUES:
        result[f"precision@{k}"] = precision_sum[k] / n
        result[f"recall@{k}"] = recall_sum[k] / n

    all_results.append(result)

    print("\nValidation results")
    for k in K_VALUES:
        print(f"Precision@{k}: {result[f'precision@{k}']:.6f}")
        print(f"Recall@{k}:    {result[f'recall@{k}']:.6f}")

    # Save the model and recommendations if Recall@20 improves.
    if result["recall@20"] > best_recall20:
        best_recall20 = result["recall@20"]
        best_config = config.copy()

        np.savez_compressed(
            "implicit_als_best_round3.npz",
            user_factors=model.user_factors,
            item_factors=model.item_factors,
            user_ids=user_ids,
            item_ids=item_ids,
            latent_factors=FACTORS,
            regularization=config["regularization"],
            alpha=config["alpha"],
            iterations=ITERATIONS,
            seed=SEED
        )

        pd.DataFrame(rec_rows).to_csv(
            "implicit_als_best_round3_recommendations.csv", index=False
        )

        print("New best round-3 model saved.")

    del model


# ============================================================
# 5. Save experiment results
# ============================================================

# Sort configurations by Recall@20 and save the comparison results.
results_df = pd.DataFrame(all_results).sort_values("recall@20", ascending=False)
results_df.to_csv("implicit_als_round3_results.csv", index=False)

print("\n========== ROUND 3 COMPLETE ==========")
print(results_df.to_string(index=False))
print("\nBest configuration by Recall@20:", best_config)
print(f"Best Recall@20: {best_recall20:.6f}")

print("\nSaved:")
print("- implicit_als_round3_results.csv")
print("- implicit_als_best_round3.npz")
print("- implicit_als_best_round3_recommendations.csv")