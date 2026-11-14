# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 조윤성
#
# SGD Matrix Factorization Model Training
#   1. Load training and validation interactions and create ID mappings
#   2. Define the Matrix Factorization model with user/item embeddings and biases
#   3. Train the model using SGD with L2 regularization
#   4. Evaluate each epoch using validation RMSE
#   5. Save the model with the best validation performance
#
# Inputs  : sgd_train.csv, sgd_validation.csv
# Outputs : sgd_mf_best_model.pt


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import copy

# ==========================================
# 1. Basic settings
# ==========================================

SEED = 42
LATENT_FACTORS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 8192
EPOCHS = 10
REGULARIZATION = 0.001

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device)

# ==========================================
# 2. Load datasets
# ==========================================

# Load the prepared training and validation datasets
train_df = pd.read_csv("sgd_train.csv")
valid_df = pd.read_csv("sgd_validation.csv")

print("\nDataset loaded")
print("Train:", len(train_df))
print("Validation:", len(valid_df))

# ==========================================
# 3. Create ID mappings
# ==========================================

# Map original user and item IDs to consecutive integer indices
user_ids = sorted(train_df["steam_id"].unique())
item_ids = sorted(train_df["item_id"].unique())

user_to_idx = {v: i for i, v in enumerate(user_ids)}
item_to_idx = {v: i for i, v in enumerate(item_ids)}

n_users = len(user_ids)
n_items = len(item_ids)

# Convert dataset IDs and preference scores into PyTorch tensors
def convert_to_indices(df):
    user_idx = df["steam_id"].map(user_to_idx)
    item_idx = df["item_id"].map(item_to_idx)

    if user_idx.isna().any() or item_idx.isna().any():
        raise ValueError("Unknown ID found.")

    return (
        torch.tensor(user_idx.to_numpy(), dtype=torch.long),
        torch.tensor(item_idx.to_numpy(), dtype=torch.long),
        torch.tensor(
            df["preference_score"].to_numpy(dtype=np.float32)
        )
    )

train_users, train_items, train_scores = convert_to_indices(train_df)
valid_users, valid_items, valid_scores = convert_to_indices(valid_df)

# Create mini-batches for training
train_loader = DataLoader(
    TensorDataset(train_users, train_items, train_scores),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ==========================================
# 4. Define Matrix Factorization model
# ==========================================

class MatrixFactorization(nn.Module):

    def __init__(self, n_users, n_items, factors, global_mean):
        super().__init__()

        # Learnable user and item latent factor vectors
        self.user_factors = nn.Embedding(
            n_users, factors, sparse=True
        )
        self.item_factors = nn.Embedding(
            n_items, factors, sparse=True
        )

        # Learnable user and item bias terms
        self.user_bias = nn.Embedding(
            n_users, 1, sparse=True
        )
        self.item_bias = nn.Embedding(
            n_items, 1, sparse=True
        )

        # Initialize latent factors and biases
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # Global average preference score
        self.global_bias = nn.Parameter(
            torch.tensor(global_mean, dtype=torch.float32)
        )

    def forward(self, users, items):

        user_vec = self.user_factors(users)
        item_vec = self.item_factors(items)

        user_b = self.user_bias(users).squeeze()
        item_b = self.item_bias(items).squeeze()

        # Predict preference scores using latent factors and bias terms
        prediction = (
            self.global_bias
            + user_b
            + item_b
            + (user_vec * item_vec).sum(dim=1)
        )

        return prediction


# Initialize the model using the mean training preference score
global_mean = train_df["preference_score"].mean()

model = MatrixFactorization(
    n_users,
    n_items,
    LATENT_FACTORS,
    global_mean
).to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

# ==========================================
# 5. Evaluation
# ==========================================

# Calculate MSE and RMSE on the validation dataset
def evaluate(model, users, items, scores):

    model.eval()

    with torch.no_grad():
        predictions = model(users, items)
        mse = criterion(predictions, scores).item()
        rmse = np.sqrt(mse)

    return mse, rmse


# ==========================================
# 6. Training
# ==========================================

# Track the model with the lowest validation RMSE
best_rmse = float("inf")
best_state = None

print("\nGlobal mean:", global_mean)
print("\n========== TRAINING START ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total_samples = 0

    for users, items, scores in train_loader:

        optimizer.zero_grad()

        predictions = model(users, items)
        mse_loss = criterion(predictions, scores)

        user_vec = model.user_factors(users)
        item_vec = model.item_factors(items)

        user_b = model.user_bias(users)
        item_b = model.item_bias(items)

        # Apply L2 regularization to latent factors and bias terms
        reg_loss = REGULARIZATION * (
            user_vec.pow(2).sum(dim=1).mean()
            + item_vec.pow(2).sum(dim=1).mean()
            + user_b.pow(2).mean()
            + item_b.pow(2).mean()
        )

        loss = mse_loss + reg_loss

        loss.backward()
        optimizer.step()

        batch_size = len(scores)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss = total_loss / total_samples

    # Evaluate the current model using validation RMSE
    _, valid_rmse = evaluate(
        model,
        valid_users,
        valid_items,
        valid_scores
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Validation RMSE: {valid_rmse:.6f}"
    )

    # Save a copy of the model if validation RMSE improves
    if valid_rmse < best_rmse:
        best_rmse = valid_rmse
        best_state = copy.deepcopy(model.state_dict())

# ==========================================
# 7. Save best model
# ==========================================

# Restore the best-performing model before saving
model.load_state_dict(best_state)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "user_ids": user_ids,
        "item_ids": item_ids,
        "latent_factors": LATENT_FACTORS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "regularization": REGULARIZATION,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "best_validation_rmse": best_rmse,
        "global_mean": global_mean
    },
    "sgd_mf_best_model.pt"
)

print("\n========== TRAINING COMPLETE ==========")
print("Best Validation RMSE:", best_rmse)
print("Saved: sgd_mf_best_model.pt")