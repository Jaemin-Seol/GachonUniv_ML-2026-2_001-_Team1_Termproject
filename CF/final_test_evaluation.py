import pandas as pd
import numpy as np
import torch
import torch.nn as nn

TRAIN_PATH = "sgd_train.csv"
TEST_PATH = "sgd_test.csv"
SGD_MODEL_PATH = "sgd_mf_best_model.pt"
ALS_MODEL_PATH = "implicit_als_best_round3.npz"

K_VALUES = [10, 20]
MAX_K = max(K_VALUES)
BATCH_SIZE = 256
DEVICE = torch.device("cpu")

print("Device:", DEVICE)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

try:
    sgd_checkpoint = torch.load(
        SGD_MODEL_PATH, map_location=DEVICE, weights_only=False
    )
except TypeError:
    sgd_checkpoint = torch.load(SGD_MODEL_PATH, map_location=DEVICE)

als_checkpoint = np.load(ALS_MODEL_PATH)

sgd_user_ids = list(sgd_checkpoint["user_ids"])
sgd_item_ids = list(sgd_checkpoint["item_ids"])
als_user_ids = als_checkpoint["user_ids"]
als_item_ids = als_checkpoint["item_ids"]

print("\nDatasets loaded")
print("Train interactions:", len(train_df))
print("Test interactions:", len(test_df))
print("SGD users/items:", len(sgd_user_ids), len(sgd_item_ids))
print("ALS users/items:", len(als_user_ids), len(als_item_ids))

train_users = set(train_df["steam_id"].unique())
train_items = set(train_df["item_id"].unique())

if set(sgd_user_ids) != train_users or set(als_user_ids) != train_users:
    raise ValueError("Model user IDs do not match training users.")
if set(sgd_item_ids) != train_items or set(als_item_ids) != train_items:
    raise ValueError("Model item IDs do not match the training catalog.")

# Common warm-start Test ground truth for all three methods.
test_eval = test_df[
    test_df["steam_id"].isin(train_users)
    & test_df["item_id"].isin(train_items)
].copy()

test_relevant = (
    test_eval.groupby("steam_id")["item_id"].apply(set).to_dict()
)
eval_users = sorted(test_relevant.keys())
train_seen = train_df.groupby("steam_id")["item_id"].apply(set).to_dict()
catalog_items = sorted(train_items)

print("\nEvaluation users:", len(eval_users))
print("Candidate games:", len(catalog_items))
print("Test interactions retained:", len(test_eval))

if not eval_users:
    raise ValueError("No eligible users found in the Test set.")


def calculate_user_metrics(recommended_items, relevant_items):
    result = {}
    for k in K_VALUES:
        hits = len(set(recommended_items[:k]) & relevant_items)
        result[k] = (hits / k, hits / len(relevant_items))
    return result


def get_top_k(scores, seen_indices, item_ids):
    if seen_indices:
        scores[list(seen_indices)] = -np.inf
    selected = np.argpartition(-scores, MAX_K - 1)[:MAX_K]
    order = np.lexsort((item_ids[selected], -scores[selected]))
    return selected[order]


def save_outputs(model_name, precision_sum, recall_sum, rows):
    metrics = []
    for k in K_VALUES:
        metrics.append({
            "Model": model_name,
            "K": k,
            "Precision": precision_sum[k] / len(eval_users),
            "Recall": recall_sum[k] / len(eval_users),
            "Users": len(eval_users)
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{model_name.lower()}_test_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(
        f"{model_name.lower()}_test_recommendations.csv", index=False
    )
    return metrics_df


# ------------------------------------------------------------
# Popularity baseline
# ------------------------------------------------------------
print("\n========== POPULARITY BASELINE ==========")

popularity = (
    train_df.groupby("item_id")["steam_id"]
    .nunique()
    .reindex(catalog_items)
    .fillna(0)
)
popularity_order = np.lexsort((
    np.asarray(catalog_items), -popularity.to_numpy()
))
popularity_ranked_items = np.asarray(catalog_items)[popularity_order]

pop_precision = {k: 0.0 for k in K_VALUES}
pop_recall = {k: 0.0 for k in K_VALUES}
pop_rows = []

for user in eval_users:
    seen = train_seen.get(user, set())
    relevant = test_relevant[user]
    recommended = [
        item for item in popularity_ranked_items if item not in seen
    ][:MAX_K]

    user_metrics = calculate_user_metrics(recommended, relevant)
    for k in K_VALUES:
        p, r = user_metrics[k]
        pop_precision[k] += p
        pop_recall[k] += r

    for rank, item in enumerate(recommended, start=1):
        pop_rows.append({
            "steam_id": user,
            "rank": rank,
            "item_id": item,
            "popularity": int(popularity.loc[item])
        })

pop_metrics_df = save_outputs(
    "popularity", pop_precision, pop_recall, pop_rows
)

# ------------------------------------------------------------
# SGD Matrix Factorization
# ------------------------------------------------------------
print("\n========== SGD MATRIX FACTORIZATION ==========")


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, factors, global_mean):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, factors, sparse=True)
        self.item_factors = nn.Embedding(n_items, factors, sparse=True)
        self.user_bias = nn.Embedding(n_users, 1, sparse=True)
        self.item_bias = nn.Embedding(n_items, 1, sparse=True)
        self.global_bias = nn.Parameter(
            torch.tensor(global_mean, dtype=torch.float32)
        )

    def forward(self, users, items):
        user_vec = self.user_factors(users)
        item_vec = self.item_factors(items)
        user_b = self.user_bias(users).squeeze(-1)
        item_b = self.item_bias(items).squeeze(-1)
        return (
            self.global_bias + user_b + item_b
            + (user_vec * item_vec).sum(dim=1)
        )


sgd_model = MatrixFactorization(
    len(sgd_user_ids),
    len(sgd_item_ids),
    sgd_checkpoint["latent_factors"],
    sgd_checkpoint["global_mean"]
).to(DEVICE)
sgd_model.load_state_dict(sgd_checkpoint["model_state_dict"])
sgd_model.eval()

sgd_user_to_idx = {v: i for i, v in enumerate(sgd_user_ids)}
sgd_item_to_idx = {v: i for i, v in enumerate(sgd_item_ids)}
sgd_catalog_ids = np.asarray(sgd_item_ids)

with torch.no_grad():
    sgd_user_factors = sgd_model.user_factors.weight.detach()
    sgd_item_factors_t = sgd_model.item_factors.weight.detach().T.contiguous()
    sgd_user_bias = sgd_model.user_bias.weight.detach().squeeze(-1)
    sgd_item_bias = sgd_model.item_bias.weight.detach().squeeze(-1)
    sgd_global_bias = float(sgd_model.global_bias.item())

sgd_precision = {k: 0.0 for k in K_VALUES}
sgd_recall = {k: 0.0 for k in K_VALUES}
sgd_rows = []

for start in range(0, len(eval_users), BATCH_SIZE):
    batch_users = eval_users[start:start + BATCH_SIZE]
    batch_indices = torch.tensor(
        [sgd_user_to_idx[u] for u in batch_users], dtype=torch.long
    )

    with torch.no_grad():
        scores_batch = sgd_user_factors[batch_indices] @ sgd_item_factors_t
        scores_batch += sgd_global_bias
        scores_batch += sgd_user_bias[batch_indices].unsqueeze(1)
        scores_batch += sgd_item_bias.unsqueeze(0)

    scores_batch = scores_batch.numpy()

    for row_idx, user in enumerate(batch_users):
        seen_indices = {
            sgd_item_to_idx[item] for item in train_seen.get(user, set())
        }
        selected = get_top_k(
            scores_batch[row_idx], seen_indices, sgd_catalog_ids
        )
        recommended = [sgd_item_ids[i] for i in selected]
        relevant = test_relevant[user]

        user_metrics = calculate_user_metrics(recommended, relevant)
        for k in K_VALUES:
            p, r = user_metrics[k]
            sgd_precision[k] += p
            sgd_recall[k] += r

        for rank, (idx, item) in enumerate(zip(selected, recommended), start=1):
            sgd_rows.append({
                "steam_id": user,
                "rank": rank,
                "item_id": item,
                "predicted_score": float(scores_batch[row_idx, idx])
            })

    print(f"SGD processed users: {min(start + BATCH_SIZE, len(eval_users))}/{len(eval_users)}")

sgd_metrics_df = save_outputs("sgd", sgd_precision, sgd_recall, sgd_rows)

# ------------------------------------------------------------
# Implicit ALS - selected Round 3 model
# ------------------------------------------------------------
print("\n========== IMPLICIT ALS ==========")

als_user_factors = als_checkpoint["user_factors"]
als_item_factors = als_checkpoint["item_factors"]
als_user_to_idx = {v: i for i, v in enumerate(als_user_ids)}
als_item_to_idx = {v: i for i, v in enumerate(als_item_ids)}
als_item_ids_array = np.asarray(als_item_ids)

als_precision = {k: 0.0 for k in K_VALUES}
als_recall = {k: 0.0 for k in K_VALUES}
als_rows = []

for start in range(0, len(eval_users), BATCH_SIZE):
    batch_users = eval_users[start:start + BATCH_SIZE]
    batch_indices = np.asarray(
        [als_user_to_idx[u] for u in batch_users], dtype=np.int64
    )
    scores_batch = als_user_factors[batch_indices] @ als_item_factors.T

    for row_idx, user in enumerate(batch_users):
        seen_indices = {
            als_item_to_idx[item] for item in train_seen.get(user, set())
        }
        selected = get_top_k(
            scores_batch[row_idx], seen_indices, als_item_ids_array
        )
        recommended = [als_item_ids[i] for i in selected]
        relevant = test_relevant[user]

        user_metrics = calculate_user_metrics(recommended, relevant)
        for k in K_VALUES:
            p, r = user_metrics[k]
            als_precision[k] += p
            als_recall[k] += r

        for rank, (idx, item) in enumerate(zip(selected, recommended), start=1):
            als_rows.append({
                "steam_id": user,
                "rank": rank,
                "item_id": item,
                "predicted_score": float(scores_batch[row_idx, idx])
            })

    print(f"ALS processed users: {min(start + BATCH_SIZE, len(eval_users))}/{len(eval_users)}")

als_metrics_df = save_outputs(
    "implicit_als", als_precision, als_recall, als_rows
)

# ------------------------------------------------------------
# Combined final results
# ------------------------------------------------------------
all_metrics = pd.concat(
    [pop_metrics_df, sgd_metrics_df, als_metrics_df],
    ignore_index=True
)
all_metrics.to_csv("final_test_metrics_all_models.csv", index=False)

print("\n========== FINAL TEST RESULTS ==========")
print(all_metrics.to_string(index=False))

print("\nSaved files:")
print("- popularity_test_metrics.csv")
print("- popularity_test_recommendations.csv")
print("- sgd_test_metrics.csv")
print("- sgd_test_recommendations.csv")
print("- implicit_als_test_metrics.csv")
print("- implicit_als_test_recommendations.csv")
print("- final_test_metrics_all_models.csv")
print("\n========== TEST EVALUATION COMPLETE ==========")
