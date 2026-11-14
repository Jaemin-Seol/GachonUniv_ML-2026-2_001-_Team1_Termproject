# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# Popularity-based Recommendation Evaluation
#   1. Calculate game popularity based on distinct training users
#   2. Generate top-K recommendations excluding previously owned games
#   3. Evaluate recommendations using macro Precision@K and Recall@K
#   4. Save evaluation metrics and recommendation lists
#
# Inputs  : sgd_train.csv, sgd_validation.csv
# Outputs : popularity_validation_metrics.csv,
#           popularity_validation_recommendations.csv


import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================

# Load training and validation interaction data
train_df = pd.read_csv("sgd_train.csv")
valid_df = pd.read_csv("sgd_validation.csv")

print("Train data:", train_df.shape)
print("Validation data:", valid_df.shape)


# =========================
# 2. Calculate game popularity
# =========================

# Count the number of distinct users who own each game in the training set
popularity = (
    train_df.groupby("item_id")["steam_id"]
    .nunique()
    .reset_index(name="popularity")
)

# Rank games by popularity, using item_id to break ties consistently
popularity = popularity.sort_values(
    by=["popularity", "item_id"],
    ascending=[False, True]
).reset_index(drop=True)

# Define the training games as the candidate recommendation catalog
all_items = popularity["item_id"].tolist()

print("Number of candidate games:", len(all_items))


# =========================
# 3. Prepare evaluation data
# =========================

# Store games already owned by each user in the training set
train_seen = (
    train_df.groupby("steam_id")["item_id"]
    .apply(set)
    .to_dict()
)

# Collect relevant games from the validation set for each user
valid_relevant = (
    valid_df.groupby("steam_id")["item_id"]
    .apply(set)
    .to_dict()
)

# Include users with at least one relevant validation game
eval_users = list(valid_relevant.keys())

print("Number of evaluation users:", len(eval_users))


# =========================
# 4. Generate recommendations
# =========================

K_VALUES = [10, 20]
MAX_K = max(K_VALUES)

recommendation_rows = []

# Accumulate per-user Precision and Recall for macro-averaging
precision_sum = {k: 0.0 for k in K_VALUES}
recall_sum = {k: 0.0 for k in K_VALUES}

for user in eval_users:

    seen_items = train_seen.get(user, set())
    relevant_items = valid_relevant[user]

    # Exclude games already owned by the user
    candidates = [
        item for item in all_items
        if item not in seen_items
    ]

    # Recommend the most popular unseen games
    recommended_items = candidates[:MAX_K]

    # Calculate Precision and Recall at each K
    for k in K_VALUES:

        top_k = recommended_items[:k]
        hits = len(set(top_k) & relevant_items)

        precision = hits / k
        recall = hits / len(relevant_items)

        precision_sum[k] += precision
        recall_sum[k] += recall

    # Store top-20 recommendations for each user
    for rank, item in enumerate(recommended_items, start=1):

        item_popularity = popularity.loc[
            popularity["item_id"] == item, "popularity"
        ].iloc[0]

        recommendation_rows.append({
            "steam_id": user,
            "rank": rank,
            "item_id": item,
            "popularity": item_popularity
        })


# =========================
# 5. Calculate macro-average metrics
# =========================

num_users = len(eval_users)

metrics = []

# Calculate the average metric across all evaluation users
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
# 6. Save results
# =========================

# Save validation metrics and top-20 recommendation lists
metrics_df.to_csv(
    "popularity_validation_metrics.csv",
    index=False
)

recommendations_df = pd.DataFrame(recommendation_rows)

recommendations_df.to_csv(
    "popularity_validation_recommendations.csv",
    index=False
)


# =========================
# 7. Print results
# =========================

print("\nPopularity Baseline - Validation Results")
print("----------------------------------------")

for _, row in metrics_df.iterrows():

    print(f"K = {int(row['K'])}")
    print(f"Precision@{int(row['K'])}: {row['Precision']:.6f}")
    print(f"Recall@{int(row['K'])}:    {row['Recall']:.6f}")
    print()

print("Saved popularity_validation_metrics.csv")
print("Saved popularity_validation_recommendations.csv")
