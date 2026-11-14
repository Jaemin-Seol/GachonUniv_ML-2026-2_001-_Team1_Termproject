# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# Implicit ALS Hyperparameter Tuning
#   1. Define hyperparameter configurations for comparison
#   2. Build the user-item matrix and prepare validation data
#   3. Train and evaluate each configuration using Precision@K and Recall@K
#   4. Select the best model based on Recall@20
#   5. Save the best model, recommendations, and experiment results
#
# Inputs  : sgd_train.csv, sgd_validation.csv
# Outputs : implicit_als_tuning_results.csv,
#           implicit_als_best_tuned.npz,
#           implicit_als_best_tuned_recommendations.csv


import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


# ============================================================
# 1. Experiment settings
# ============================================================

SEED = 42
ITERATIONS = 20
K_VALUES = [10, 20]
MAX_K = max(K_VALUES)
BATCH_SIZE = 256

# Round 1: one-factor-at-a-time experiments.
# The original setting (64, 0.01, 40) is included as a reference.
CONFIGS = [
    {"name": "baseline", "factors": 64, "regularization": 0.01, "alpha": 40},
    {"name": "factors_32", "factors": 32, "regularization": 0.01, "alpha": 40},
    {"name": "factors_128", "factors": 128, "regularization": 0.01, "alpha": 40},
    {"name": "alpha_10", "factors": 64, "regularization": 0.01, "alpha": 10},
    {"name": "alpha_20", "factors": 64, "regularization": 0.01, "alpha": 20},
    {"name": "reg_0.1", "factors": 64, "regularization": 0.1, "alpha": 40},
]


# ============================================================
# 2. Load data and build the user-item matrix once
# ============================================================

# Load the training and validation datasets
train_df = pd.read_csv("sgd_train.csv")
valid_df = pd.read_csv("sgd_validation.csv")

print("Dataset loaded")
print("Train interactions:", len(train_df))
print("Validation interactions:", len(valid_df))

# Create mappings from original IDs to consecutive indices
user_ids = np.sort(train_df["steam_id"].unique())
item_ids = np.sort(train_df["item_id"].unique())

user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

user_index = pd.Index(user_ids).get_indexer(train_df["steam_id"])
item_index = pd.Index(item_ids).get_indexer(train_df["item_id"])

num_users = len(user_ids)
num_items = len(item_ids)

# Build a sparse user-item matrix using preference scores
confidence_values = train_df["preference_score"].to_numpy(dtype=np.float32)

user_items = coo_matrix(
    (confidence_values, (user_index, item_index)),
    shape=(num_users, num_items),
    dtype=np.float32
).tocsr()

print("\nUser-item matrix:", user_items.shape)
print("Non-zero interactions:", user_items.nnz)


# ============================================================
# 3. Prepare evaluation data once
# ============================================================

# Store training interactions for excluding previously owned games
train_df["user_idx"] = user_index
train_df["item_idx"] = item_index

train_seen = (
    train_df.groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

# Keep validation interactions belonging to users and games in the training catalog
valid_eval = valid_df[
    valid_df["steam_id"].isin(user_to_idx)
    & valid_df["item_id"].isin(item_to_idx)
].copy()

valid_eval["user_idx"] = valid_eval["steam_id"].map(user_to_idx)
valid_eval["item_idx"] = valid_eval["item_id"].map(item_to_idx)

# Group relevant validation games by user
valid_relevant = (
    valid_eval.groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

eval_users = np.array(sorted(valid_relevant.keys()), dtype=np.int64)

print("Evaluation users:", len(eval_users))


# ============================================================
# 4. Train and evaluate one configuration
# ============================================================

def train_and_evaluate(config):
    print("\n" + "=" * 60)
    print("Configuration:", config["name"])
    print(config)
    print("=" * 60)

    # Initialize and train ALS with the current hyperparameters
    model = AlternatingLeastSquares(
        factors=config["factors"],
        regularization=config["regularization"],
        alpha=config["alpha"],
        iterations=ITERATIONS,
        random_state=SEED,
        num_threads=-1
    )

    model.fit(user_items, show_progress=True)

    precision_sum = {k: 0.0 for k in K_VALUES}
    recall_sum = {k: 0.0 for k in K_VALUES}
    recommendation_rows = []

    # Generate recommendations in batches to reduce memory usage
    for start in range(0, len(eval_users), BATCH_SIZE):
        batch_users = eval_users[start:start + BATCH_SIZE]

        scores = (
            model.user_factors[batch_users]
            @ model.item_factors.T
        )

        # Exclude games already present in each user's training data.
        for row_idx, user_idx in enumerate(batch_users):
            seen_items = train_seen.get(int(user_idx), set())
            if seen_items:
                scores[row_idx, list(seen_items)] = -np.inf

        # Select the highest-scoring MAX_K items for each user
        top_indices = np.argpartition(
            -scores,
            kth=MAX_K - 1,
            axis=1
        )[:, :MAX_K]

        for row_idx, user_idx in enumerate(batch_users):
            selected = top_indices[row_idx]
            order = np.argsort(-scores[row_idx, selected])
            recommended_items = selected[order]

            relevant_items = valid_relevant[int(user_idx)]

            # Accumulate Precision@K and Recall@K
            for k in K_VALUES:
                top_k = recommended_items[:k]
                hits = len(set(top_k) & relevant_items)

                precision_sum[k] += hits / k
                recall_sum[k] += hits / len(relevant_items)

            # Keep recommendations for this configuration.
            original_user_id = user_ids[user_idx]
            for rank, item_idx in enumerate(recommended_items, start=1):
                recommendation_rows.append({
                    "config": config["name"],
                    "steam_id": original_user_id,
                    "rank": rank,
                    "item_id": item_ids[item_idx],
                    "predicted_score": float(scores[row_idx, item_idx])
                })

        if (start // BATCH_SIZE) % 50 == 0:
            print(
                f"Evaluated {min(start + BATCH_SIZE, len(eval_users))}"
                f" / {len(eval_users)} users"
            )

    # Calculate macro-averaged evaluation metrics
    n = len(eval_users)
    result = {
        "config": config["name"],
        "factors": config["factors"],
        "regularization": config["regularization"],
        "alpha": config["alpha"],
        "iterations": ITERATIONS,
    }

    for k in K_VALUES:
        result[f"precision@{k}"] = precision_sum[k] / n
        result[f"recall@{k}"] = recall_sum[k] / n

    recommendations_df = pd.DataFrame(recommendation_rows)

    print("\nValidation results")
    for k in K_VALUES:
        print(f"Precision@{k}: {result[f'precision@{k}']:.6f}")
        print(f"Recall@{k}:    {result[f'recall@{k}']:.6f}")

    return model, result, recommendations_df


# ============================================================
# 5. Run all experiments
# ============================================================

all_results = []
best_recall_at_20 = -1.0
best_config = None

for config in CONFIGS:
    model, result, recommendations_df = train_and_evaluate(config)
    all_results.append(result)

    # Select the best configuration by Recall@20.
    # This criterion is fixed before comparing the experiments.
    if result["recall@20"] > best_recall_at_20:
        best_recall_at_20 = result["recall@20"]
        best_config = config.copy()

        # Save the current best model and its ID mappings
        np.savez_compressed(
            "implicit_als_best_tuned.npz",
            user_factors=model.user_factors,
            item_factors=model.item_factors,
            user_ids=user_ids,
            item_ids=item_ids,
            latent_factors=config["factors"],
            regularization=config["regularization"],
            alpha=config["alpha"],
            iterations=ITERATIONS,
            seed=SEED
        )

        # Save recommendations generated by the best configuration
        recommendations_df.to_csv(
            "implicit_als_best_tuned_recommendations.csv",
            index=False
        )

        print("\nNew best model saved.")

    del model


# ============================================================
# 6. Save and display comparison
# ============================================================

# Sort experiment results by Recall@20 and save the comparison table
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(
    by="recall@20",
    ascending=False
)

results_df.to_csv(
    "implicit_als_tuning_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nBest configuration by Recall@20:")
print(best_config)
print(f"Best Recall@20: {best_recall_at_20:.6f}")

print("\nSaved files:")
print("- implicit_als_tuning_results.csv")
print("- implicit_als_best_tuned.npz")
print("- implicit_als_best_tuned_recommendations.csv")