# Steam Game Recommendation: Collaborative Filtering vs. Content-based Filtering

Gachon University, Machine Learning (61375_001), 2026-2 - Team 1 Term Project

We build a Steam game recommender from users' game ownership and playtime, and compare four models on the same data split and metrics:

- **Popularity baseline**
- **SGD Matrix Factorization** (CF, PyTorch)
- **Implicit ALS** (CF)
- **Content-based filtering** (TF-IDF over genres/tags)

## Results (test set)

| Model | P@10 | R@10 | P@20 | R@20 | Long-tail@10 |
|---|---|---|---|---|---|
| Implicit ALS (CF) | 0.1371 | 0.3010 | 0.1040 | 0.4042 | 0.02% |
| Popularity baseline | 0.0821 | 0.1840 | 0.0600 | 0.2500 | 0% |
| SGD MF (CF) | 0.0650 | 0.1330 | 0.0448 | 0.1706 | 0% |
| CBF: TF-IDF + dot | 0.0349 | 0.0846 | 0.0279 | 0.1212 | 6.7% |
| CBF: TF-IDF + cosine | 0.0245 | 0.0650 | 0.0202 | 0.0963 | 11.2% |

Implicit ALS is the most accurate. CBF is less accurate but is the only model that regularly recommends lesser-known games.

## Project Structure

```
Preprocessing/
  Preprocessing.py        data exploration and preprocessing
  SteamSpy.py             fetches missing metadata from the SteamSpy API
  steam_tag_manual.csv    manual metadata rules
CF/
  prepare_sgd_data.py     user filtering, preference scores, train/validation/test split
  train_sgd_mf.py         trains SGD Matrix Factorization
  evaluate_sgd.py         validation evaluation of SGD MF
  evaluate_popularity.py  validation evaluation of the popularity baseline
  train_implicit_als.py   trains Implicit ALS
  evaluate_implicit_als.py  validation evaluation of Implicit ALS
  tune_implicit_als.py, tune_implicit_als_round2.py, tune_implicit_als_round3.py
                          Implicit ALS hyperparameter tuning (best by Recall@20)
  final_test_evaluation.py  final test evaluation of all CF models and the baseline
CBF/
  CBF.py                  CBF training/tuning, evaluation, and comparison with the CF results
```

## Data

Download the following and put the files in `Preprocessing/`. Raw data is not included in this repository.

| Source | Files to use |
|---|---|
| [UCSD Steam data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) | `australian_users_items.json` (User and Item Data v1), `steam_games.json` (Item Metadata v2). The downloads are usually `.json.gz`, so extract them first. |
| [Kaggle Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) | `games_march2025_cleaned.csv` |

Game metadata still missing after these is collected from the [SteamSpy API](https://steamspy.com/about) by `SteamSpy.py` (no download needed, but it needs an internet connection and takes about 30 minutes).

Note: the SteamSpy API is a third-party service and may be unavailable or return different data than when we ran it (some games may have been delisted). If requests fail, `SteamSpy.py` records them as failed and retries them the next time it runs. The results can then differ slightly from the table above.

Final data: about 5.07M interactions, 70,912 users, 10,841 games.

## Setup

Python 3.10 (the pinned versions in `requirements.txt` were tested with Python 3.10).

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Packages: pandas, NumPy, SciPy, scikit-learn, matplotlib, seaborn, requests, PyTorch, implicit.

## How to Run

Run each script from inside its own folder.

**1. Preprocessing** (in `Preprocessing/`)

```
python Preprocessing.py
python SteamSpy.py
python Preprocessing.py
```

The first run stops with a `FileNotFoundError` on purpose: it first saves `steamspy_appids.csv`, which `SteamSpy.py` needs. After `SteamSpy.py` creates `steamspy_api_results.csv`, run `Preprocessing.py` again. The second run ends with three plot windows (close them to finish) and creates `steam_games_tag.csv` and `steam_interactions.csv`.

**2. Collaborative filtering** (in `CF/`)

Run these four to get the final test results. `final_test_evaluation.py` uses `sgd_mf_best_model.pt` and `implicit_als_best_round3.npz`.

```
python prepare_sgd_data.py
python train_sgd_mf.py
python tune_implicit_als_round3.py
python final_test_evaluation.py
```

The other scripts record the validation experiments and the earlier tuning rounds. They are not needed for the final results, and can be run in this order after `prepare_sgd_data.py`:

```
python evaluate_popularity.py        # popularity baseline, validation
python evaluate_sgd.py               # after train_sgd_mf.py
python train_implicit_als.py         # default ALS model before tuning
python evaluate_implicit_als.py      # after train_implicit_als.py
python tune_implicit_als.py          # tuning round 1
python tune_implicit_als_round2.py   # tuning round 2
```

**3. Content-based filtering and comparison** (in `CBF/`)

```
python CBF.py
```

It reads the CF split and recommendation lists from `CF/` and `steam_games_tag.csv` from `Preprocessing/`, then writes `validation_grid.csv`, `test_metrics.csv` and a figure to `CBF/results/cf_split/`. A full run takes about ten minutes.

Generated files (CSV results, trained models) are not stored in this repository; running the scripts above recreates them. Results may differ slightly from the table above if SteamSpy returns different data.

## Evaluation Setup

- Users with fewer than 3 games are excluded (65,827 users). Each user's games are split about 80/10/10 into train/validation/test.
- Preference score: `S = alpha * O + beta * log(1 + playtime)`.
- Each model recommends K games (K = 10, 20) from all games in train, excluding games the user already owns.
- Metrics: Precision@K, Recall@K, and long-tail share (recommendations outside the top 20% most-owned games).

## Team

| Member | Role |
|---|---|
| 설재민 | Preprocessing |
| 조윤성 | Collaborative filtering |
| 김인현 | Content-based filtering |
