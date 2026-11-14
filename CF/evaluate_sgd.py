# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# SGD Matrix Factorization Evaluation
#   1. Load the trained SGD MF model and evaluation datasets
#   2. Reconstruct the model and calculate user-item prediction scores
#   3. Generate top-K recommendations, excluding training interactions
#   4. Evaluate recommendations using macro Precision@K and Recall@K
#   5. Save evaluation metrics and recommendation lists
#
# Inputs  : sgd_mf_best_model.pt, sgd_train.csv, sgd_validation.csv
# Outputs : sgd_validation_metrics.csv,
#           sgd_validation_recommendations.csv


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


# ==========================================
# 1. Basic settings
# ==========================================

MODEL_PATH = "sgd_mf_best_model.pt"
TRAIN_PATH = "sgd_train.csv"
VALID_PATH = "sgd_validation.csv"

K_VALUES = [10, 20]
BATCH_SIZE = 256

# Use CPU for model evaluation
device = torch.device("cpu")

print("Device:", device)


# ==========================================
# 2. Load datasets and model
# ==========================================

train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

# Load the model checkpoint, including its ID mappings and metadata
# weights_only=False is required for checkpoints containing non-weight objects
try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

user_ids = checkpoint["user_ids"]
item_ids = checkpoint["item_ids"]

# Map original user and item IDs to model matrix indices
user_to_idx = {
    user_id: idx for idx, user_id in enumerate(user_ids)
}

item_to_idx = {
    item_id: idx for idx, item_id in enumerate(item_ids)
}

n_users = len(user_ids)
n_items = len(item_ids)
factors = checkpoint["latent_factors"]

print("\nDataset loaded")
print("Train interactions:", len(train_df))
print("Validation interactions:", len(valid_df))
print("Users:", n_users)
print("Items:", n_items)


# ==========================================
# 3. Define the same MF model architecture
# ==========================================

# Reconstruct the model architecture to load the trained parameters
class MatrixFactorization(nn.Module):

    def __init__(self, n_users, n_items, factors, global_mean):
        super().__init__()

        self.user_factors = nn.Embedding(
            n_users, factors, sparse=True
        )

        self.item_factors = nn.Embedding(
            n_items, factors, sparse=True
        )

        self.user_bias = nn.Embedding(
            n_users, 1, sparse=True
        )

        self.item_bias = nn.Embedding(
            n_items, 1, sparse=True
        )

        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.global_bias = nn.Parameter(
            torch.tensor(global_mean, dtype=torch.float32)
        )

    def forward(self, users, items):

        user_vec = self.user_factors(users)
        item_vec = self.item_factors(items)

        user_b = self.user_bias(users).squeeze(-1)
        item_b = self.item_bias(items).squeeze(-1)

        # Predict preference scores using latent factors and bias terms
        prediction = (
            self.global_bias
            + user_b
            + item_b
            + (user_vec * item_vec).sum(dim=1)
        )

        return prediction


model = MatrixFactorization(
    n_users,
    n_items,
    factors,
    checkpoint["global_mean"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nTrained model loaded successfully")


# ==========================================
# 4. Build user-item interaction dictionaries
# ==========================================

train_seen = defaultdict(set)
valid_relevant = defaultdict(set)

# Record games already observed by each user in the training set
for row in train_df[["steam_id", "item_id"]].itertuples(
    index=False,
    name=None
):
    user_id, item_id = row

    u = user_to_idx[user_id]
    i = item_to_idx[item_id]

    train_seen[u].add(i)

# Collect relevant validation games for each user
for row in valid_df[["steam_id", "item_id"]].itertuples(
    index=False,
    name=None
):
    user_id, item_id = row

    # Exclude users and games outside the training catalog
    if user_id not in user_to_idx:
        continue

    if item_id not in item_to_idx:
        continue

    u = user_to_idx[user_id]
    i = item_to_idx[item_id]

    valid_relevant[u].add(i)

# Evaluate only users with at least one validation interaction
# in the training user and item catalog
eval_users = sorted(valid_relevant.keys())

print("\nEvaluation users:", len(eval_users))


# ==========================================
# 5. Prepare model parameters
# ==========================================

with torch.no_grad():

    # Extract trained user/item embeddings and bias terms
    user_factors = model.user_factors.weight
    item_factors = model.item_factors.weight

    user_bias = model.user_bias.weight.squeeze(-1)
    item_bias = model.item_bias.weight.squeeze(-1)

    global_bias = model.global_bias

    # Precompute the transposed item factor matrix for efficient scoring
    item_factors_t = item_factors.T.contiguous()


# ==========================================
# 6. Generate top-K recommendations
# ==========================================

max_k = max(K_VALUES)

hit_counts = {
    k: 0 for k in K_VALUES
}

# Accumulate per-user Precision and Recall for macro-averaging
precision_sum = {
    k: 0.0 for k in K_VALUES
}

recall_sum = {
    k: 0.0 for k in K_VALUES
}

recommendation_rows = []

print("\n========== EVALUATION START ==========")

# Process users in batches to improve evaluation efficiency
for start in range(0, len(eval_users), BATCH_SIZE):

    batch_user_indices = eval_users[
        start:start + BATCH_SIZE
    ]

    users_tensor = torch.tensor(
        batch_user_indices,
        dtype=torch.long
    )

    with torch.no_grad():

        # Calculate predicted preference scores for all candidate games
        # Score = global bias + user bias + item bias + user-item dot product
        scores = torch.matmul(
            user_factors[users_tensor],
            item_factors_t
        )

        scores += global_bias
        scores += user_bias[users_tensor].unsqueeze(1)
        scores += item_bias.unsqueeze(0)

        # Exclude games already observed in the training set
        for row_idx, user_idx in enumerate(batch_user_indices):

            seen_items = list(train_seen[user_idx])

            if seen_items:
                scores[row_idx, seen_items] = float("-inf")

        # Select the top-K candidate games for each user
        top_scores, top_items = torch.topk(
            scores,
            k=max_k,
            dim=1
        )

    # Evaluate recommendations for each user in the batch
    for row_idx, user_idx in enumerate(batch_user_indices):

        recommended_items = top_items[row_idx].tolist()
        recommended_scores = top_scores[row_idx].tolist()

        relevant_items = valid_relevant[user_idx]

        # Calculate Precision and Recall at each K
        for k in K_VALUES:

            top_k_items = recommended_items[:k]

            hits = len(
                set(top_k_items) & relevant_items
            )

            precision = hits / k
            recall = hits / len(relevant_items)

            precision_sum[k] += precision
            recall_sum[k] += recall
            hit_counts[k] += hits

        # Store the top-20 recommendations for each user
        original_user_id = user_ids[user_idx]

        for rank, (item_idx, score) in enumerate(
            zip(recommended_items, recommended_scores),
            start=1
        ):

            recommendation_rows.append({
                "steam_id": original_user_id,
                "item_id": item_ids[item_idx],
                "rank": rank,
                "predicted_score": score
            })

    processed = min(
        start + BATCH_SIZE,
        len(eval_users)
    )

    print(
        f"Processed users: {processed}/{len(eval_users)}"
    )


# ==========================================
# 7. Calculate final metrics
# ==========================================

n_eval_users = len(eval_users)

print("\n========== FINAL RESULTS ==========")

results = []

# Calculate macro-averaged Precision and Recall across evaluation users
for k in K_VALUES:

    precision_at_k = precision_sum[k] / n_eval_users
    recall_at_k = recall_sum[k] / n_eval_users

    print(f"\nK = {k}")
    print(f"Precision@{k}: {precision_at_k:.6f}")
    print(f"Recall@{k}: {recall_at_k:.6f}")

    results.append({
        "K": k,
        "Precision": precision_at_k,
        "Recall": recall_at_k,
        "Users": n_eval_users
    })


# ==========================================
# 8. Save results
# ==========================================

# Save validation metrics and top-20 recommendation lists
results_df = pd.DataFrame(results)

results_df.to_csv(
    "sgd_validation_metrics.csv",
    index=False
)

recommendations_df = pd.DataFrame(
    recommendation_rows
)

recommendations_df.to_csv(
    "sgd_validation_recommendations.csv",
    index=False
)

print("\nSaved:")
print("sgd_validation_metrics.csv")
print("sgd_validation_recommendations.csv")

print("\n========== EVALUATION COMPLETE ==========")