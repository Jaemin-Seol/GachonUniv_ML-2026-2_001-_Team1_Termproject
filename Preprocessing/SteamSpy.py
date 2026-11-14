# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 설재민
#
# SteamSpy metadata collection
#
# This script collects game metadata from SteamSpy for the saved App IDs.
import os
import time
import pandas as pd
import requests

# Set the input and output file paths.
input_file = "./steamspy_appids.csv"
results_file = "./steamspy_api_results.csv"

# Stop if the App ID list has not been created.
if not os.path.exists(input_file):
    raise FileNotFoundError(
        "SteamSpy app ID file not found. Run the preprocessing code first."
    )

# Load unique App IDs to check with SteamSpy.
df_api_input = pd.read_csv(input_file, dtype={"appid": str})
app_ids = df_api_input["appid"].drop_duplicates().tolist()

# Skip IDs with a saved result and retry IDs whose requests failed.
if os.path.exists(results_file):
    df_saved = pd.read_csv(results_file, dtype={"appid": str})
    completed_ids = set(
        df_saved.loc[
            df_saved["status"] != "Request failed",
            "appid"
        ]
    )
else:
    completed_ids = set()

print("Total app IDs:", len(app_ids))
print("Already completed:", len(completed_ids))

new_results = []
processed_count = 0

# Request metadata for each App ID that does not have a completed result.
for app_id in app_ids:
    if app_id in completed_ids:
        continue

    try:
        response = requests.get(
            "https://steamspy.com/api.php",
            params={"request": "appdetails", "appid": app_id},
            timeout=20
        )
        response.raise_for_status()
        result = response.json()

        # Record the App ID when SteamSpy returns no usable data.
        if not isinstance(result, dict) or "appid" not in result:
            row = {
                "appid": app_id,
                "status": "No data",
                "name": "",
                "genres": "",
                "tag_count": 0,
                "tags": ""
            }
            print("No data:", app_id)

        # Save the game name, genres, and tags when data is available.
        else:
            tag_votes = result.get("tags", {})
            if not isinstance(tag_votes, dict):
                tag_votes = {}

            row = {
                "appid": app_id,
                "status": "OK",
                "name": result.get("name", ""),
                "genres": result.get("genre", ""),
                "tag_count": len(tag_votes),
                "tags": ", ".join(tag_votes.keys())
            }
            print(result.get("name", app_id), "tags:", len(tag_votes))

    # Record request or response parsing errors so they can be retried later.
    except (requests.RequestException, ValueError) as error:
        row = {
            "appid": app_id,
            "status": "Request failed",
            "name": "",
            "genres": "",
            "tag_count": 0,
            "tags": ""
        }
        print("Request failed:", app_id, error)

    new_results.append(row)
    processed_count += 1

    # Save results every 25 requests to preserve progress.
    if processed_count % 25 == 0:
        pd.DataFrame(new_results).to_csv(
            results_file,
            mode="a",
            header=not os.path.exists(results_file),
            index=False
        )
        new_results = []
        print("Checkpoint saved.")

    time.sleep(1.1)

# Save any remaining results.
if new_results:
    pd.DataFrame(new_results).to_csv(
        results_file,
        mode="a",
        header=not os.path.exists(results_file),
        index=False
    )

# Keep the latest result if a failed request was retried.
if os.path.exists(results_file):
    df_results = pd.read_csv(results_file, dtype={"appid": str})
    df_results = df_results.drop_duplicates(subset="appid", keep="last")
    df_results.to_csv(results_file, index=False)

print("SteamSpy results saved:", results_file)