# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# Interaction Data Preparation and Splitting
#   1. Load interaction data and filter users with fewer than 3 games
#   2. Calculate preference scores based on game ownership and playtime
#   3. Split interactions into training, validation, and test sets
#   4. Filter validation and test games not present in the training catalog
#   5. Save the processed datasets for model training and evaluation
#
# Inputs  : steam_interactions.csv
# Outputs : sgd_train.csv, sgd_validation.csv, sgd_test.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ==========================================
# 1. Load dataset
# ==========================================

# Load the preprocessed user-game interaction data
df = pd.read_csv("../Preprocessing/steam_interactions.csv")

print("Original dataset")
print("Interactions:", len(df))
print("Users:", df["steam_id"].nunique())
print("Games:", df["item_id"].nunique())


# ==========================================
# 2. Filter eligible users
# ==========================================

# Retain users with at least 3 game interactions
user_counts = df.groupby("steam_id").size()

eligible_users = user_counts[user_counts >= 3].index

df = df[df["steam_id"].isin(eligible_users)].copy()

print("\nAfter filtering")
print("Interactions:", len(df))
print("Eligible users:", df["steam_id"].nunique())
print("Games:", df["item_id"].nunique())


# ==========================================
# 3. Calculate preference scores
# ==========================================

ALPHA = 1.0
BETA = 0.1

# Calculate preference scores using game ownership and playtime
df["preference_score"] = (
    ALPHA
    + BETA * np.log1p(df["playtime_forever"])
)

print("\nPreference score statistics")
print(df["preference_score"].describe())


# ==========================================
# 4. Train / Validation / Test split
# ==========================================

train_list = []
valid_list = []
test_list = []

# Use a fixed random seed to ensure reproducible splits
rng = np.random.RandomState(42)

# Split interactions separately for each user
for user_id, group in df.groupby("steam_id", sort=False):

    indices = group.index.to_numpy(copy=True)

    # Shuffle each user's interactions before splitting
    rng.shuffle(indices)

    n = len(indices)

    # Allocate approximately 80% to training and 10% each to validation and test
    n_test = max(1, round(n * 0.1))
    n_valid = max(1, round(n * 0.1))

    test_indices = indices[:n_test]
    valid_indices = indices[n_test:n_test + n_valid]
    train_indices = indices[n_test + n_valid:]

    # Ensure that each user has at least one training interaction
    if len(train_indices) == 0:
        train_indices = valid_indices[-1:]
        valid_indices = valid_indices[:-1]

    train_list.extend(train_indices)
    valid_list.extend(valid_indices)
    test_list.extend(test_indices)


train_df = df.loc[train_list].copy()
valid_df = df.loc[valid_list].copy()
test_df = df.loc[test_list].copy()


# ==========================================
# 5. Check item coverage
# ==========================================

# Identify games available in the training catalog
train_items = set(train_df["item_id"])

valid_before = len(valid_df)
test_before = len(test_df)

# Exclude validation and test interactions for games unseen during training
valid_df = valid_df[
    valid_df["item_id"].isin(train_items)
].copy()

test_df = test_df[
    test_df["item_id"].isin(train_items)
].copy()


# ==========================================
# 6. Save datasets
# ==========================================

# Save the datasets for use across recommendation models
train_df.to_csv("sgd_train.csv", index=False)
valid_df.to_csv("sgd_validation.csv", index=False)
test_df.to_csv("sgd_test.csv", index=False)


# ==========================================
# 7. Print final summary
# ==========================================

print("\n========== SPLIT SUMMARY ==========")

print("Train interactions:", len(train_df))
print("Validation interactions:", len(valid_df))
print("Test interactions:", len(test_df))

print("Train users:", train_df["steam_id"].nunique())
print("Train games:", train_df["item_id"].nunique())

# Report the proportion of validation and test interactions
# retained after filtering out unseen games
print("\nValidation warm-start coverage:",
      f"{len(valid_df) / valid_before:.2%}")

print("Test warm-start coverage:",
      f"{len(test_df) / test_before:.2%}")

print("\nSaved:")
print("sgd_train.csv")
print("sgd_validation.csv")
print("sgd_test.csv")
