# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# Implicit ALS Validation Evaluation
#   1. Load the trained ALS model and validation data
#   2. Generate top-K recommendations for each user
#   3. Calculate Precision@K and Recall@K
#   4. Save evaluation metrics and recommendation results
#
# Inputs  : sgd_train.csv, sgd_validation.csv, implicit_als_model.npz
# Outputs : implicit_als_validation_metrics.csv,
#           implicit_als_validation_recommendations.csv


import pandas as pd
import numpy as np


# =========================
# 1. Load data and model
# =========================

# Load training and validation interaction data
train_df = pd.read_csv("sgd_train.csv")
valid_df = pd.read_csv("sgd_validation.csv")

# Load the trained Implicit ALS model and its user/item mappings
checkpoint = np.load(
    "implicit_als_model.npz"
)

user_factors = checkpoint["user_factors"]
item_factors = checkpoint["item_factors"]

user_ids = checkpoint["user_ids"]
item_ids = checkpoint["item_ids"]

print("Dataset loaded")
print("Train interactions:", len(train_df))
print("Validation interactions:", len(valid_df))

print("Users:", len(user_ids))
print("Items:", len(item_ids))


# =========================
# 2. Create ID mappings
# =========================

# Map original user and item IDs to matrix indices
user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

item_to_idx = {
    item_id: idx
    for idx, item_id in enumerate(item_ids)
}


# =========================
# 3. Prepare training data
# =========================

# Convert original IDs to model matrix indices
train_df["user_idx"] = train_df["steam_id"].map(user_to_idx)
train_df["item_idx"] = train_df["item_id"].map(item_to_idx)

# Store each user's previously observed games for filtering
train_seen = (
    train_df.groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)


# =========================
# 4. Prepare validation data
# =========================

# Keep only users and games included in the training catalog
valid_eval = valid_df[
    valid_df["steam_id"].isin(user_to_idx)
    & valid_df["item_id"].isin(item_to_idx)
].copy()

valid_eval["user_idx"] = valid_eval["steam_id"].map(user_to_idx)
valid_eval["item_idx"] = valid_eval["item_id"].map(item_to_idx)

# Build the set of relevant validation games for each user
valid_relevant = (
    valid_eval.groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

eval_users = list(valid_relevant.keys())

print("\nEvaluation users:", len(eval_users))


# =========================
# 5. Generate recommendations
# =========================

K_VALUES = [10, 20]
MAX_K = max(K_VALUES)

BATCH_SIZE = 256

# Accumulate per-user Precision and Recall for macro-averaging
precision_sum = {k: 0.0 for k in K_VALUES}
recall_sum = {k: 0.0 for k in K_VALUES}

recommendation_rows = []

print("\n========== EVALUATION START ==========")

# Process users in batches to reduce memory usage
for start in range(0, len(eval_users), BATCH_SIZE):

    batch_users = eval_users[start:start + BATCH_SIZE]

    batch_indices = np.array(batch_users, dtype=np.int64)

    # Calculate predicted scores for all candidate games
    scores = (
        user_factors[batch_indices]
        @ item_factors.T
    )

    # Exclude games already owned by each user in the training set
    for row_idx, user_idx in enumerate(batch_users):

        seen_items = train_seen.get(user_idx, set())

        if seen_items:
            scores[row_idx, list(seen_items)] = -np.inf

    # Select top 20 candidates for each user
    top_indices = np.argpartition(
        -scores,
        kth=MAX_K - 1,
        axis=1
    )[:, :MAX_K]

    # Sort selected games by predicted score
    for row_idx, user_idx in enumerate(batch_users):

        selected = top_indices[row_idx]

        selected_scores = scores[row_idx, selected]

        order = np.argsort(-selected_scores)

        recommended_items = selected[order]

        relevant_items = valid_relevant[user_idx]

        # Calculate Precision and Recall K=10 and K=20
        for k in K_VALUES:

            top_k = recommended_items[:k]

            hits = len(
                set(top_k) & relevant_items
            )

            precision_sum[k] += hits / k

            recall_sum[k] += hits / len(relevant_items)

        # Save top-20 recommendations for each user
        original_user_id = user_ids[user_idx]

        for rank, item_idx in enumerate(
            recommended_items,
            start=1
        ):

            recommendation_rows.append({
                "steam_id": original_user_id,
                "rank": rank,
                "item_id": item_ids[item_idx],
                "predicted_score": scores[row_idx, item_idx]
            })

    if (start // BATCH_SIZE) % 50 == 0:
        print(
            f"Processed {min(start + BATCH_SIZE, len(eval_users))}"
            f" / {len(eval_users)} users"
        )


# =========================
# 6. Calculate macro metrics
# =========================

num_users = len(eval_users)

metrics = []

# Average each user's metrics to obtain macro-averaged results
for k in K_VALUES:

    precision_avg = precision_sum[k] / num_users
    recall_avg = recall_sum[k] / num_users

    metrics.append({
        "K": k,
        "Precision": precision_avg,
        "Recall": recall_avg
    })

metrics_df = pd.DataFrame(metrics)


# =========================
# 7. Save results
# =========================

# Save validation metrics and top-20 recommendations
metrics_df.to_csv(
    "implicit_als_validation_metrics.csv",
    index=False
)

recommendations_df = pd.DataFrame(
    recommendation_rows
)

recommendations_df.to_csv(
    "implicit_als_validation_recommendations.csv",
    index=False
)


# =========================
# 8. Print results
# =========================

print("\n========== FINAL RESULTS ==========")

for _, row in metrics_df.iterrows():

    k = int(row["K"])

    print(f"\nK = {k}")
    print(f"Precision@{k}: {row['Precision']:.6f}")
    print(f"Recall@{k}: {row['Recall']:.6f}")

print("\nSaved:")
print("implicit_als_validation_metrics.csv")
print("implicit_als_validation_recommendations.csv")

print("\n========== EVALUATION COMPLETE ==========")
