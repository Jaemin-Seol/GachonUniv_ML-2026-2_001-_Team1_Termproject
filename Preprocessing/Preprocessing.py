# Gachon Univ. Machine Learning (61375_001) 2026-2
# Term project - Recsys
# 설재민
#
# Data exploration and preprocessing
#
# This script prepares the user-game interaction data and game metadata.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

def load_nonstandard_json(file_path):
    records = []

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(ast.literal_eval(line))

    return pd.DataFrame(records)

if __name__ == '__main__':
    # =============================================================================
    # 1. Load and inspect user data
    # =============================================================================
    print("\n" + "=" * 70)
    print("1. Load and inspect user data")
    print("=" * 70)

    # The dataset is not in standard JSON format, so pd.read_json() cannot read it directly.
    # Loading may take some time.
    print("Loading dataset...(User)")
    df = load_nonstandard_json("./australian_users_items.json")

    # Check the DataFrame structure
    print("\n[DataFrame structure]")
    df.info()

    # Check the structure of the nested game records
    print("\n[First user's first two games]")
    print(df.loc[0, "items"][:2]) #show first two items (games)

    # Check for duplicate Steam IDs before processing user records
    print("\n[Duplicate user check]")
    print("Total user records:", len(df))
    print("Unique Steam IDs:", df["steam_id"].nunique())

    # =============================================================================
    # 2. Resolve duplicate user records
    # =============================================================================
    print("\n" + "=" * 70)
    print("2. Resolve duplicate user records")
    print("=" * 70)

    # Find all records with duplicated Steam IDs.
    duplicate_records = df[
        df["steam_id"].duplicated(keep=False)
    ]

    rows_to_drop = []

    # Group duplicated records by Steam ID.
    for steam_id, group in duplicate_records.groupby("steam_id", sort=False):
        seen_records = []

        # Remove exact duplicate records.
        for index, row in group.iterrows():
            current_record = row.to_dict()

            # Mark exact duplicates for removal
            if any(current_record == previous for previous in seen_records):
                rows_to_drop.append(index)
            else:
                seen_records.append(current_record)

    # Remove exact duplicate rows
    df_clean = df.drop(index=rows_to_drop).reset_index(drop=True)

    print("Exact duplicate rows removed:", len(rows_to_drop))
    print("Extra repeated Steam ID rows remaining:",
        df_clean["steam_id"].duplicated().sum())

    # Check duplicated Steam IDs after removing exact duplicate records.
    print("\n[Duplicate check after removing exact duplicates]")
    duplicate_records = df_clean[
        df_clean["steam_id"].duplicated(keep=False)
    ]

    different_item_list_count = 0

    # Check whether records for this user contain different sets of games.
    for steam_id, group in duplicate_records.groupby("steam_id"):
        item_lists = []

        for items in group["items"]:
            # Sort for easy comparison
            item_ids = sorted(item["item_id"] for item in items)
            item_lists.append(item_ids)

        # Count users whose records contain different game lists.
        if any(item_list != item_lists[0] for item_list in item_lists[1:]):
            different_item_list_count += 1

    print("Accounts with different item ID lists:", different_item_list_count)

    # Select the record with the highest total lifetime playtime.
    df_clean["total_playtime"] = df_clean["items"].apply(
        lambda items: sum(item["playtime_forever"] for item in items)
    )

    # Keep the record with the highest total playtime for each Steam ID.
    df_merged = df_clean.loc[
        df_clean.groupby("steam_id")["total_playtime"].idxmax()
    ].drop(columns="total_playtime").reset_index(drop=True)

    print("\n[User data after duplicate handling]")
    df_merged.info()

    # =============================================================================
    # 3. Build the interaction table
    # =============================================================================
    print("\n" + "=" * 70)
    print("3. BUILD THE INTERACTION TABLE")
    print("=" * 70)

    # Remove columns not needed for the interaction table.
    df_merged = df_merged.drop(
        columns=["user_id", "user_url", "items_count"]
    )

    # Expand each user's game list into separate rows.
    df_exploded = df_merged.explode("items", ignore_index=True)
    df_exploded = df_exploded.dropna(subset=["items"])

    # Convert the nested game records into columns.
    item_details = pd.json_normalize(df_exploded["items"])

    # Combine Steam IDs with game details.
    df_interaction = pd.concat(
        [
            df_exploded[["steam_id"]].reset_index(drop=True),
            item_details.reset_index(drop=True)
        ], axis=1
    )

    # Check consistency between recent and lifetime playtime.
    print("\n[Playtime consistency check]")

    invalid_playtime = (
        df_interaction["playtime_2weeks"]
        > df_interaction["playtime_forever"]
    )

    print("Inconsistent records:", invalid_playtime.sum())
    print(df_interaction.loc[
        invalid_playtime,
        ["steam_id", "item_id", "playtime_forever", "playtime_2weeks"]
    ].head())

    # Treat recent playtime as the minimum possible lifetime playtime.
    df_interaction.loc[invalid_playtime, "playtime_forever"] = (
        df_interaction.loc[invalid_playtime, "playtime_2weeks"]
    )

    print(
        "Inconsistent records after correction:",
        (
            df_interaction["playtime_2weeks"]
            > df_interaction["playtime_forever"]
        ).sum()
    )

    # Check interaction counts, duplicate user-game pairs, and missing values.
    print("\n[Interaction table summary]")
    df_interaction.info()
    print(df_interaction.head())

    print("Rows:", len(df_interaction))
    print("Unique users:", df_interaction["steam_id"].nunique())
    print("Unique games:", df_interaction["item_id"].nunique())
    print("Duplicate user-game rows:",
        df_interaction.duplicated(subset=["steam_id", "item_id"]).sum())
    
    print("\nMissing values:")
    print(df_interaction.isna().sum())

    # =============================================================================
    # 4. Load and prepare game metadata
    # =============================================================================
    print("\n" + "=" * 70)
    print("4. LOAD AND PREPARE GAME METADATA")
    print("=" * 70)

    # Create a list of unique games
    df_games = df_interaction[["item_id", "item_name"]].drop_duplicates(subset="item_id")

    print("\n[Interaction game list]")
    print("Unique games:", len(df_games))
    print(df_games.head())

    # Load the primary game metadata.
    print("\nLoading dataset... (Game)")
    df_metadata = load_nonstandard_json("./steam_games.json")

    # Check metadata info.
    print("\n[Primary metadata structure]")
    df_metadata.info()

    # Find metadata records without an App ID.
    print("\n[Games with missing App IDs]")
    missing_id_games = df_metadata[df_metadata["id"].isna()]

    print("Rows with missing game ID:", len(missing_id_games))
    print(missing_id_games.to_string(index=False))

    # Check whether App ID 200260 appears in the primary metadata.
    print("\n[Search primary metadata for App ID 200260]")
    game_200260 = df_metadata[df_metadata["id"] == "200260"]

    print("Matching rows:", len(game_200260))
    print(game_200260.to_string(index=False))

    # Remove records without an App ID.
    df_metadata = df_metadata.dropna(subset=["id"]).reset_index(drop=True)

    # Remove columns not used for game metadata.
    df_metadata = df_metadata.drop(columns=[
        "publisher",
        "title",
        "url",
        "release_date",
        "discount_price",
        "reviews_url",
        "specs",
        "price",
        "early_access",
        "developer",
        "sentiment",
        "metascore"
    ])

    # Preview the selected metadata columns.
    print("\n[Primary metadata after column selection]")
    print(df_metadata.head())

    # Check for duplicate App IDs before matching metadata to interaction games.
    print("\n[Duplicate App ID check]")
    print("Total game records:", len(df_metadata))
    print("Unique game IDs:", df_metadata["id"].nunique())
    print("Duplicate game IDs:", df_metadata["id"].duplicated().sum())

    # Display all rows with duplicate App IDs.
    print("\n[Duplicate metadata rows]")
    duplicate_game_ids = df_metadata[
        df_metadata["id"].duplicated(keep=False)
    ].sort_values("id")

    print(duplicate_game_ids.to_string(index=False))

    # Keep one metadata row for each App ID.
    df_metadata = df_metadata.drop_duplicates(
        subset="id",keep="first"
    ).reset_index(drop=True)

    print("\n[Primary metadata after duplicate removal]")
    print("Total game records:", len(df_metadata))
    print("Unique game IDs:", df_metadata["id"].nunique())

    # Match interaction game IDs with primary metadata IDs.
    print("\n[Match interaction games to primary metadata]")
    matched_game_count = df_games["item_id"].isin(df_metadata["id"]).sum()

    print("Unique interaction games:", len(df_games))
    print("Matched unique games:", matched_game_count)
    print(
        "Unique-game matching rate:",
        f"{matched_game_count / len(df_games):.2%}"
    )

    # =============================================================================
    # 5. Supplement missing game metadata
    # =============================================================================
    print("\n" + "=" * 70)
    print("5. SUPPLEMENT MISSING GAME METADATA")
    print("=" * 70)

    # Load the supplementary metadata dataset.
    print("\nLoading dataset... (Supplementary game metadata)")
    df_supplement = pd.read_csv("./games_march2025_cleaned.csv")

    print("\n[Supplementary metadata structure]")
    df_supplement.info()

    # Keep the columns needed for matching and metadata features.
    print("\n[Selected supplementary metadata columns]")
    df_supplement = df_supplement[["appid", "name", "genres", "tags"]].copy()
    df_supplement.info()

    print("\n[Supplementary metadata examples]")
    print(df_supplement.head())

    # Check whether the supplementary data contains duplicate App IDs.
    print("\n[Supplementary metadata duplicate App ID check]")
    print("Total game records:", len(df_supplement))
    print("Unique game IDs:", df_supplement["appid"].nunique())
    print("Duplicate game IDs:", df_supplement["appid"].duplicated().sum())

    # Use the same column names as the primary metadata.
    df_supplement = df_supplement.rename(columns={
        "appid": "id",
        "name": "app_name"
    })

    # Convert App IDs to strings and parse genres and tags into lists.
    df_supplement["id"] = df_supplement["id"].astype(str)
    df_supplement["genres"] = df_supplement["genres"].apply(ast.literal_eval)
    df_supplement["tags"] = df_supplement["tags"].apply(
        lambda value: list(ast.literal_eval(value).keys())
        if value.strip().startswith("{")
        else ast.literal_eval(value)
    )

    # Keep the columns used in the final game metadata.
    df_supplement = df_supplement[["app_name", "genres", "tags", "id"]]

    print("\n[Filter supplementary metadata]")

    # Remove games with no tags.
    has_tags = df_supplement["tags"].map(len) > 0
    print("Rows removed for missing tags:", (~has_tags).sum())
    df_supplement = df_supplement[has_tags].copy()

    # Keep games that appear in the interaction data but not in the primary metadata.
    is_new_game = (
        ~df_supplement["id"].isin(df_metadata["id"])
        & df_supplement["id"].isin(df_games["item_id"])
    )

    print("Supplementary games kept:", is_new_game.sum())
    df_supplement = df_supplement[is_new_game].copy()

    # =============================================================================
    # 6. Supplement metadata with SteamSpy
    # =============================================================================
    print("\n" + "=" * 70)
    print("6. SUPPLEMENT METADATA WITH STEAMSPY")
    print("=" * 70)

    # Find interaction games that still have no tags in either metadata source.
    primary_has_tags = df_metadata["tags"].fillna("").map(len).gt(0)

    content_ids = set(df_metadata.loc[primary_has_tags, "id"].astype(str))
    content_ids.update(df_supplement["id"].astype(str))

    df_missing_games = df_games[
        ~df_games["item_id"].astype(str).isin(content_ids)
    ].copy()

    # Save App IDs that still need metadata for the SteamSpy lookup.
    df_missing_games[["item_id"]].rename(
        columns={"item_id": "appid"}
    ).to_csv("steamspy_appids.csv", index=False)
    print("steamspy_appids.csv saved")

    steamspy_file = "./steamspy_api_results.csv"

    # Stop if the SteamSpy API results have not been saved yet.
    if not os.path.exists(steamspy_file):
        raise FileNotFoundError(
            "SteamSpy result file not found. Run the SteamSpy API script first."
        )

    # Load the saved SteamSpy results.
    df_steamspy = pd.read_csv(
        steamspy_file,
        dtype={"appid": str}
    )

    print("SteamSpy results loaded:", len(df_steamspy))

    # =============================================================================
    # 7. Merge metadata sources
    # =============================================================================
    print("\n" + "=" * 70)
    print("7. MERGE METADATA SOURCES")
    print("=" * 70)

    # Select the fields needed from the primary metadata.
    df_primary_metadata = df_metadata[
        ["id", "genres", "tags"]
    ].copy()

    df_primary_metadata = df_primary_metadata.rename(
        columns={"id": "item_id"}
    )

    # Use string App IDs to match the interaction game list.
    df_primary_metadata["item_id"] = (
        df_primary_metadata["item_id"].astype(str)
    )

    # Add primary genres and tags to the interaction game list.
    df_games = df_games.merge(
        df_primary_metadata,
        on="item_id",
        how="left"
    )

    # Prepare the supplementary metadata for matching by App ID.
    df_supplement_metadata = df_supplement[
        ["id", "genres", "tags"]
    ].rename(
        columns={"id": "item_id"}
    ).set_index("item_id")

    # Add supplementary metadata to games
    df_games = df_games.set_index("item_id")
    df_games.update(df_supplement_metadata)
    df_games = df_games.reset_index()

    # Keep only successful SteamSpy results and the metadata fields used below.
    df_steamspy_metadata = df_steamspy[
        df_steamspy["status"] == "OK"
    ][["appid", "genres", "tags"]].copy()

    # Convert comma separated SteamSpy genres and tags into lists.
    for column in ["genres", "tags"]:
        df_steamspy_metadata[column] = (
            df_steamspy_metadata[column]
            .fillna("").apply(
                lambda value: [
                    part.strip()
                    for part in value.split(",")
                    if part.strip()
                ]
            )
        )

    # Match SteamSpy metadata to games
    df_steamspy_metadata = df_steamspy_metadata.rename(
        columns={"appid": "item_id"}
    ).set_index("item_id")

    # Add SteamSpy metadata to the game list.
    df_games = df_games.set_index("item_id")
    df_games.update(df_steamspy_metadata)
    df_games = df_games.reset_index()

    # =============================================================================
    # 8. Apply manual metadata recovery rules
    # =============================================================================
    print("\n" + "=" * 70)
    print("8. APPLY MANUAL METADATA RECOVERY RULES")
    print("=" * 70)

    # Load the manual rules for final preprocessing
    manual = pd.read_csv("./steam_tag_manual.csv", dtype=str).fillna("")
    manual = manual[manual["Target"].str.strip() != ""]

    # Keep a copy of the original game metadata for source lookups.
    games = df_games.set_index("item_id")
    source_games = games.copy()
    items_to_delete = []

    # Apply each manual rule to its target game.
    for _, row in manual.iterrows():
        target = row["Target"].strip()
        source = row["Source"].strip()

        # Mark the target for deletion when the source is -1.
        if source == "-1":
            items_to_delete.append(target)

        # Use the manually supplied tags when the source is 0.
        elif source == "0":
            games.at[target, "tags"] = [
                tag.strip()
                for tag in row["tags"].split(",")
                if tag.strip()
            ]

        # Copy genres and tags from another game when a source App ID is provided.
        else:
            if source not in source_games.index:
                raise ValueError(
                    f"Source App ID {source} is not in df_games. "
                )

            games.at[target, "genres"] = source_games.at[source, "genres"]
            games.at[target, "tags"] = source_games.at[source, "tags"]

    # Remove games marked for deletion
    games = games.drop(index=items_to_delete, errors="ignore")
    df_games = games.reset_index()

    # Combine genres and tags into one feature list, removing duplicates.
    df_games["tags"] = df_games.apply(
        lambda row: list(set(
            (row["genres"] if isinstance(row["genres"], list) else [])
            + (row["tags"] if isinstance(row["tags"], list) else [])
        )),
        axis=1
    )
    df_games = df_games.drop(columns=["genres"])

    # Remove interactions for games marked for deletion.
    df_interaction = df_interaction[
        ~df_interaction["item_id"].isin(items_to_delete)
    ].copy()

    # Save the final game metadata and interaction tables.
    df_games.to_csv("steam_games_tag.csv", index=False)
    df_interaction.to_csv("steam_interactions.csv", index=False)

    # =============================================================================
    # 9. Finalize and summarize the datasets
    # =============================================================================
    print("\n" + "=" * 70)
    print("9. FINALIZE AND SUMMARIZE THE DATASETS")
    print("=" * 70)

    # Summarize the final interaction data after metadata recovery and game removal.
    print("\n[Final dataset]")
    print("Interaction rows:", len(df_interaction))
    print("Unique users:", df_interaction["steam_id"].nunique())
    print("Unique games:", df_games["item_id"].nunique())
    print(
        "Duplicate user-game rows:",
        df_interaction.duplicated(subset=["steam_id", "item_id"]).sum()
    )

    # Calculate games per user and convert playtime from minutes to hours.
    games_per_user = df_interaction.groupby("steam_id")["item_id"].nunique()
    playtime_hours = df_interaction["playtime_forever"] / 60
    positive_playtime_hours = playtime_hours[playtime_hours > 0]

    # Summarize the number of games owned by each user.
    print("\nGames per user:")
    print(games_per_user.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))

    # Summarize lifetime playtime and the share of records with zero playtime.
    print("\nLifetime playtime per game (hours):")
    print(playtime_hours.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))
    print("Zero-playtime records:", (playtime_hours == 0).sum())
    print("Zero-playtime share:", f"{(playtime_hours == 0).mean():.2%}")

    # Find games with feature.
    has_features = df_games["tags"].apply(
        lambda features: isinstance(features, list) and len(features) > 0
    )
    games_with_features = set(df_games.loc[has_features, "item_id"])

    # Find interactions linked to games with features.
    interactions_with_features = df_interaction["item_id"].isin(games_with_features)

    # Show metadata coverage by game count and interaction count.
    print("\nFeature coverage:")
    print(
        f"Games with features: {has_features.sum()} / {len(df_games)} "
        f"({has_features.mean():.2%})"
    )
    print(
        "Interactions linked to games with features:",
        f"{interactions_with_features.sum()} / {len(df_interaction)} "
        f"({interactions_with_features.mean():.2%})"
    )

    # Check for missing values in the final interaction table.
    print("\nMissing values in final interaction table:")
    print(df_interaction.isna().sum())

    # Final plots

    # Plot the distribution of owned games per user.
    sns.histplot(x=games_per_user, bins=50)
    plt.yscale("log")
    plt.xlabel("Games per user")
    plt.ylabel("Number of users (log scale)")
    plt.title("Games per user")
    plt.show()

    # Plot positive lifetime playtime per game on a logarithmic scale.
    sns.histplot(x=positive_playtime_hours, bins=50, log_scale=True)
    plt.xlabel("Lifetime playtime (hours, log scale)")
    plt.ylabel("Number of user-game records")
    plt.title("Lifetime playtime per game")
    plt.show()

    # Plot the number of features available for each game.
    feature_counts = df_games.loc[has_features, "tags"].map(len)

    sns.histplot(x=feature_counts, bins=30)
    plt.xlabel("Number of genres and tags")
    plt.ylabel("Number of games")
    plt.title("Metadata features per game")
    plt.show()