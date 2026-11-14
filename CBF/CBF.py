# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 김인현
#
# Content-based Filtering (CBF)
#   1. Split: the CF teammate's train / validation / test (or our own split_v1)
#   2. Game vectors from tags (binary or TF-IDF)
#   3. User profile = sum of train game vectors weighted by S_ug = alpha*O_ug + beta*log(1+t_ug)
#   4. Top-K candidates by similarity to the profile
#   5. Pick settings on validation, evaluate once on test
#   6. Score the CF teammate's saved Top-K lists with the same metric code
#
# Inputs : Preprocessing/steam_games_tag.csv (+ steam_interactions.csv for our own split)
#          CF/sgd_{train,validation,test}.csv, CF/*_test_recommendations.csv
# Outputs: CBF/results/<split>/*.csv, *.png
# Run    : python CBF/CBF.py  (from the repository root)

import ast
import itertools
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# =============================================================================
# Settings
# =============================================================================
# "cf" : CF teammate's split (users with >= 3 games, 80/10/10). Used for the comparison.
# "own": our earlier split_v1 (users with >= 5 games, 70/10/20). Reference only.
SPLIT_SOURCE = "cf"

CBF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CBF_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "Preprocessing")
CF_DIR = os.path.join(ROOT_DIR, "CF")
RESULT_DIR = os.path.join(CBF_DIR, "results", {"cf": "cf_split", "own": "split_v1"}[SPLIT_SOURCE])
SPLIT_FILE = os.path.join(DATA_DIR, "split_v1.csv")
CF_MODELS = {          # CF teammate's test Top-20 lists (steam_id, rank, item_id, score)
    "Popularity (CF code)": "popularity_test_recommendations.csv",
    "SGD MF": "sgd_test_recommendations.csv",
    "Implicit ALS": "implicit_als_test_recommendations.csv",
}

# Own split only
SEED = 42
MIN_GAMES = 5          # users with fewer games are not evaluated
TEST_RATIO = 0.2
VAL_RATIO = 0.1

K_LIST = [10, 20]      # same K as the CF side
MAIN_K = 10            # K used to pick the best validation setting
POPULAR_SHARE = 0.2    # top 20% of games by train owners = "popular"
BATCH = 2000           # users scored at once (memory: BATCH x n_games floats)

# Only beta / alpha matters (scaling one user's weights does not change the ranking).
# log(1 + minutes) is usually 2-8, so small betas are tried too.
GRID = {
    "weighting": ["binary", "tfidf"],
    "similarity": ["cosine", "dot"],
    "alpha_beta": [(1.0, 0.0), (1.0, 0.05), (1.0, 0.1), (1.0, 0.2), (1.0, 0.5), (1.0, 1.0)],
}


# =============================================================================
# Data
# =============================================================================
def normalize_tag(tag):
    """Unify spellings: "Free To Play"/"Free to Play", "Base-Building"/"Base Building", "Parody "."""
    return " ".join(tag.lower().replace("-", " ").split())


def load_games():
    games = pd.read_csv(os.path.join(DATA_DIR, "steam_games_tag.csv"), dtype={"item_id": str})
    # tags is a list string like "['FPS', 'Action']"; ast.literal_eval(s) safely parses it to a list
    games["tags"] = games["tags"].apply(
        lambda value: sorted({normalize_tag(tag) for tag in ast.literal_eval(value) if tag.strip()})
    )
    return games


def load_cf_split():
    """CF teammate's split (CF_prepare_sgd_data.py). Val/test keep only games seen in train."""
    parts = []
    for name, split in [("sgd_train.csv", "train"), ("sgd_validation.csv", "val"), ("sgd_test.csv", "test")]:
        part = pd.read_csv(os.path.join(CF_DIR, name), dtype={"steam_id": str, "item_id": str},
                           usecols=["steam_id", "item_id", "playtime_forever"])
        part["split"] = split
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def make_split():
    """Own split: hold out a random part of each eligible user's games.

    Saved to SPLIT_FILE and reused; raises if the saved file does not match the settings.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "steam_interactions.csv"),
                     dtype={"steam_id": str, "item_id": str})
    # groupby(...).transform("size"): the user's game count, repeated on each of the user's rows
    size = df.groupby("steam_id")["steam_id"].transform("size")
    eligible = size >= MIN_GAMES
    # round to a count, at least 1 game
    n_test = (size * TEST_RATIO).round().clip(lower=1)
    n_val = (size * VAL_RATIO).round().clip(lower=1)

    if os.path.exists(SPLIT_FILE):
        split = pd.read_csv(SPLIT_FILE, dtype={"steam_id": str, "item_id": str})
        df = df.merge(split, on=["steam_id", "item_id"], how="left")
        actual_test = (df["split"] == "test").groupby(df["steam_id"]).transform("sum")
        actual_val = (df["split"] == "val").groupby(df["steam_id"]).transform("sum")
        if (df["split"].isna().any() or (actual_test != n_test.where(eligible, 0)).any()
                or (actual_val != n_val.where(eligible, 0)).any()):
            raise ValueError(f"{SPLIT_FILE} does not match the split settings. Delete it to make a new one.")
        print("Split loaded:", SPLIT_FILE)
        return df

    # np.random.default_rng(seed): random generator with a fixed seed
    # rank(method="first"): random order 1..n per user, ties broken by position
    rank = pd.Series(np.random.default_rng(SEED).random(len(df))).groupby(df["steam_id"]).rank(method="first")
    df["split"] = "train"
    df.loc[eligible & (rank <= n_test), "split"] = "test"
    df.loc[eligible & (rank > n_test) & (rank <= n_test + n_val), "split"] = "val"
    df[["steam_id", "item_id", "split"]].to_csv(SPLIT_FILE, index=False)
    print("Split saved:", SPLIT_FILE)
    return df


# =============================================================================
# Model
# =============================================================================
def game_vectors(games, weighting):
    """(games x tags) sparse matrix.

    TfidfVectorizer (scikit-learn):
      analyzer=callable : input is already a tag list, so no tokenizing
      binary=True       : a tag counts once per game
      use_idf           : True multiplies by IDF = ln((1+n)/(1+df)) + 1 (common tags weigh less),
                          False gives a plain binary vector
      norm=None         : no row normalization (cosine does it later)
    """
    vectorizer = TfidfVectorizer(analyzer=lambda tags: tags, binary=True,
                                 use_idf=(weighting == "tfidf"), norm=None)
    return vectorizer.fit_transform(games["tags"]).astype(np.float32).tocsr()


def score_matrix(df, user_index, game_index, alpha, beta):
    """(users x games) sparse S_ug. Unowned pairs are not stored (unobserved, not "dislike").

    np.log1p(t) = log(1 + t).
    sp.csr_matrix((data, (row, col)), shape): sparse matrix with data[i] at (row[i], col[i]).
    """
    rows = df["steam_id"].map(user_index).to_numpy()
    cols = df["item_id"].map(game_index).to_numpy()
    s = alpha + beta * np.log1p(df["playtime_forever"].to_numpy())
    return sp.csr_matrix((s.astype(np.float32), (rows, cols)), shape=(len(user_index), len(game_index)))


def new_sums(k_list):
    return {f"{m}@{k}": 0.0 for m in ["precision", "recall", "longtail"] for k in k_list}


def add_metrics(sums, top, target, popular, k_list):
    """Add one batch to the metric sums (macro average: divide by #users at the end).

    top: (batch x max_k) game indices, best first. target: (batch x games) bool, hidden games.
    precision = hits / K, recall = hits / #hidden, longtail = share of Top-K not popular.
    np.take_along_axis(a, idx, axis=1): per row, a[row, idx[row, j]].
    """
    hit = np.take_along_axis(target, top, axis=1)
    n_target = target.sum(axis=1)
    for k in k_list:
        hits = hit[:, :k].sum(axis=1)
        sums[f"precision@{k}"] += (hits / k).sum()
        sums[f"recall@{k}"] += (hits / n_target).sum()
        sums[f"longtail@{k}"] += (~popular[top[:, :k]]).mean(axis=1).sum()


def recommend_and_evaluate(profile_source, seen, held_out, X, similarity, popular, users,
                           k_list=K_LIST, blocked=None):
    """Top-K for `users` and their metrics.

    profile_source : (users x games) S_ug used for the profile
    seen           : (users x games) owned games, removed from candidates
    held_out       : (users x games) hidden games to recover
    X              : (games x tags) game vectors
    blocked        : bool per game, never a candidate (games not in train)

    On validation, test games stay candidates and count as misses. This lowers every
    setting alike, so the winner is unchanged, but val and test numbers are not comparable.
    """
    # Cosine: unit-length game vectors; the profile length is fixed per user, so ranking by
    # the dot product then equals ranking by cosine. normalize(X, norm="l2", axis=1) scales each row.
    if similarity == "cosine":
        X = normalize(X, norm="l2", axis=1)
    X_dense_T = X.T.toarray()                     # (tags x games)
    profiles = (profile_source @ X).toarray()     # (users x tags)

    max_k = max(k_list)
    sums = new_sums(k_list)
    for start in range(0, len(users), BATCH):
        batch = users[start:start + BATCH]
        scores = profiles[batch] @ X_dense_T
        scores[seen[batch].toarray() > 0] = -np.inf
        if blocked is not None:
            scores[:, blocked] = -np.inf
        # np.argpartition(a, kth, axis=1): per row, the kth smallest in place, smaller ones before it
        # (unsorted). On -scores this gives the max_k best without sorting all games; then sort them.
        top = np.argpartition(-scores, max_k, axis=1)[:, :max_k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        add_metrics(sums, top, held_out[batch].toarray() > 0, popular, k_list)
    return {name: value / len(users) for name, value in sums.items()}


def popularity_baseline(seen, held_out, owners, popular, users, blocked=None):
    """Most-owned games the user does not have.

    One feature per game = its owner count, so the score is proportional to popularity.
    """
    X = sp.csr_matrix(owners.astype(np.float32).reshape(-1, 1))
    return recommend_and_evaluate(seen, seen, held_out, X, "dot", popular, users, blocked=blocked)


def evaluate_lists(rec_file, held_out, popular, user_index, game_index, users, k_list=K_LIST):
    """Score a saved Top-K list (CF teammate's model) with the same add_metrics()."""
    max_k = max(k_list)
    rec = pd.read_csv(rec_file, dtype={"steam_id": str, "item_id": str}, usecols=["steam_id", "rank", "item_id"])
    rec = rec[rec["rank"] <= max_k]
    rec["u"] = rec["steam_id"].map(user_index)
    rec["g"] = rec["item_id"].map(game_index)
    # DataFrame.pivot(index, columns, values): one row per user, one column per rank
    top = rec.pivot(index="u", columns="rank", values="g")
    if top.isna().any().any() or set(top.index) != set(users):
        raise ValueError(f"{rec_file}: lists do not cover the evaluated users with {max_k} games each")
    top = top.loc[users].to_numpy(dtype=np.int64)

    sums = new_sums(k_list)
    for start in range(0, len(users), BATCH):
        part = slice(start, start + BATCH)
        add_metrics(sums, top[part], held_out[users[part]].toarray() > 0, popular, k_list)
    return {name: value / len(users) for name, value in sums.items()}


def self_check():
    # Games 0, 1 are shooters, 2 is a puzzle game. The user owns 0 and hides 1 -> 1 ranks first.
    X = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
    owned = sp.csr_matrix(np.array([[1, 0, 0]], dtype=np.float32))
    hidden = sp.csr_matrix(np.array([[0, 1, 0]], dtype=np.float32))
    popular = np.array([True, False, False])
    m = recommend_and_evaluate(owned, owned, hidden, X, "cosine", popular, np.array([0]), k_list=[1, 2])
    assert m["precision@1"] == 1.0 and m["recall@1"] == 1.0, m
    assert m["precision@2"] == 0.5 and m["longtail@2"] == 1.0, m
    assert normalize_tag("Base-Building") == normalize_tag("Base Building") == "base building"
    assert normalize_tag("Parody ") == normalize_tag("parody")
    print("Self check passed")


def plot_results(result):
    """Precision@K, Recall@K and long-tail share by K, one line per model."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, ["precision", "recall", "longtail"]):
        for _, row in result.iterrows():
            # The two popularity lines coincide; dashed keeps both visible
            ax.plot(K_LIST, [row[f"{metric}@{k}"] for k in K_LIST], marker="o", label=row["model"],
                    linestyle="--" if "CF code" in row["model"] else "-")
        ax.set_xlabel("K")
        ax.set_title({"precision": "Precision@K", "recall": "Recall@K",
                      "longtail": "Share of non-popular games in Top-K"}[metric])
        ax.set_xticks(K_LIST)
    # fig.legend(handles, labels, loc, ncol, frameon): one shared legend for all panels,
    # anchored at `loc`, laid out in `ncol` columns, frameon = box around it
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"Test set ({SPLIT_SOURCE} split)")
    # rect = (left, bottom, right, top) area for the panels; bottom 0.1 is left for the legend
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(RESULT_DIR, "test_metrics.png"), dpi=150)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    self_check()
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("\n" + "=" * 70 + "\n1. LOAD DATA AND SPLIT\n" + "=" * 70)
    games = load_games()
    df = load_cf_split() if SPLIT_SOURCE == "cf" else make_split()
    print("Split source:", SPLIT_SOURCE)
    print(df["split"].value_counts().to_string())

    user_index = {u: i for i, u in enumerate(df["steam_id"].unique())}
    game_index = {g: i for i, g in enumerate(games["item_id"])}
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    # Candidates = games in train (same as the CF side). Popular = top POPULAR_SHARE by owners.
    # np.bincount(x, minlength=n): count of each index 0..n-1. np.quantile(a, q): q-th quantile.
    owners = np.bincount(train["item_id"].map(game_index), minlength=len(game_index))
    blocked = owners == 0
    popular = owners >= np.quantile(owners[~blocked], 1 - POPULAR_SHARE)
    print("Candidate games:", (~blocked).sum(), "/", len(blocked))
    print("Popular games:", (popular & ~blocked).sum())
    print("Candidate games without tags (score 0):",
          ((games["tags"].map(len) == 0).to_numpy() & ~blocked).sum())

    def users_of(part):
        return np.array(sorted(part["steam_id"].map(user_index).unique()))

    val_users, eval_users = users_of(val), users_of(test)
    print("Validation users:", len(val_users), "| Test users:", len(eval_users))

    print("\n" + "=" * 70 + "\n2. VALIDATION GRID\n" + "=" * 70)
    owned_train = score_matrix(train, user_index, game_index, 1.0, 0.0)
    held_val = score_matrix(val, user_index, game_index, 1.0, 0.0)
    vectors = {w: game_vectors(games, w) for w in GRID["weighting"]}
    rows = []
    # itertools.product(a, b, c): every combination, i.e. the full grid
    for weighting, similarity, (alpha, beta) in itertools.product(*GRID.values()):
        profile_source = score_matrix(train, user_index, game_index, alpha, beta)
        m = recommend_and_evaluate(profile_source, owned_train, held_val,
                                   vectors[weighting], similarity, popular, val_users, blocked=blocked)
        rows.append({"weighting": weighting, "similarity": similarity, "alpha": alpha, "beta": beta, **m})
        print(f"{weighting:6} {similarity:6} a={alpha} b={beta}  "
              f"P@{MAIN_K}={m[f'precision@{MAIN_K}']:.4f}  R@{MAIN_K}={m[f'recall@{MAIN_K}']:.4f}")
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(RESULT_DIR, "validation_grid.csv"), index=False)

    # Best setting per similarity. Dot favors games with many tags (often popular ones);
    # cosine removes that. Both go to test to show the accuracy / long-tail trade-off.
    best = {sim: grid[grid["similarity"] == sim].sort_values(f"recall@{MAIN_K}").iloc[-1]
            for sim in GRID["similarity"]}
    for sim, b in best.items():
        print(f"Best {sim} on validation:", b[["weighting", "alpha", "beta"]].to_dict())

    print("\n" + "=" * 70 + "\n3. TEST (best settings, profile from train)\n" + "=" * 70)
    # Same protocol as CF: train-only profile, only train games removed from candidates
    held_test = score_matrix(test, user_index, game_index, 1.0, 0.0)
    result_rows = []
    for sim, b in best.items():
        profile_source = score_matrix(train, user_index, game_index, b["alpha"], b["beta"])
        m = recommend_and_evaluate(profile_source, owned_train, held_test,
                                   vectors[b["weighting"]], sim, popular, eval_users, blocked=blocked)
        result_rows.append({"model": f"CBF ({b['weighting']}, {sim}, beta={b['beta']})", **m})
    pop = popularity_baseline(owned_train, held_test, owners, popular, eval_users, blocked=blocked)
    result_rows.append({"model": "Popularity (CBF code)", **pop})

    # CF models from their saved lists. "Popularity (CF code)" must equal "(CBF code)":
    # that confirms both sides use the same split, candidates and metrics.
    if SPLIT_SOURCE == "cf":
        for model, file in CF_MODELS.items():
            m = evaluate_lists(os.path.join(CF_DIR, file), held_test, popular,
                               user_index, game_index, eval_users)
            result_rows.append({"model": model, **m})

    result = pd.DataFrame(result_rows)
    result.to_csv(os.path.join(RESULT_DIR, "test_metrics.csv"), index=False)
    print(result.round(4).to_string(index=False))
    plot_results(result)
    print("\nSaved:", RESULT_DIR)
