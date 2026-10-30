"""
Generate 32,000 reproducible synthetic promotion interactions.

All interactions are simulated, not actual customer behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TARGET_INTERACTIONS = 32_000
SEED = 2026
# End of the simulated observation window (inclusive).
OBSERVATION_END = pd.Timestamp("2026-10-30 23:59:59")

ACTIVITY_WEIGHTS = {"light": 0.55, "moderate": 1.0, "frequent": 1.7, "very_frequent": 2.6}


def read_data():
    users = pd.read_csv(BASE_DIR / "users.csv", parse_dates=["created_at"])
    prefs = pd.read_csv(BASE_DIR / "synthetic_user_preferences.csv")
    stores = pd.read_csv(BASE_DIR / "stores.csv")
    promos = pd.read_csv(BASE_DIR / "promotions_with_categories.csv", parse_dates=["start_date", "end_date"])

    for name, df, key in [("users", users, "user_id"), ("preferences", prefs, "user_id"),
                          ("stores", stores, "store_id"), ("promotions", promos, "promotion_id")]:
        if df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f"{name}: {key} must be present and unique")
    if set(users.user_id) != set(prefs.user_id):
        raise ValueError("User IDs differ between users and preferences")
    if not set(promos.store_id).issubset(set(stores.store_id)):
        raise ValueError("Some promotions reference missing stores")

    # The store category is authoritative in stores.csv.
    promos = promos.drop(columns=["store_category"], errors="ignore").merge(
        stores[["store_id", "category"]].rename(columns={"category": "store_category"}),
        on="store_id", validate="many_to_one")
    promos["eligible_end"] = promos["end_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    promos["eligible_end"] = promos["eligible_end"].clip(upper=OBSERVATION_END)
    return users.merge(prefs, on="user_id", validate="one_to_one"), promos


def parse_ids(value):
    return set(int(x) for x in json.loads(value)) if pd.notna(value) else set()


def parse_categories(value):
    return set(str(x) for x in json.loads(value)) if pd.notna(value) else set()


def promotion_scores(user, eligible, rng):
    """Preference-weighted selection, with a separate exploration path."""
    favorite_stores = parse_ids(user.preferred_store_ids)
    favorite_products = parse_categories(user.preferred_product_categories)
    strength_category = float(user.category_preference_strength)
    strength_store = float(user.store_preference_strength)

    scores = np.ones(len(eligible), dtype=float)
    scores += (eligible.store_category.to_numpy() == user.primary_store_category) * (2.5 * strength_category)
    scores += (eligible.store_category.to_numpy() == user.secondary_store_category) * (1.2 * strength_category)
    scores += eligible.store_id.isin(favorite_stores).to_numpy() * (3.0 * strength_store)
    scores += eligible.promotion_category.isin(favorite_products).to_numpy() * 1.7
    scores += (eligible.promotion_type.to_numpy() == user.preferred_promotion_type) * 0.9

    # Discount strength is intentionally modest: different units cannot be compared directly.
    # Only percent-based discounts get a small, capped numeric bonus.
    percent = eligible.discount_unit.astype(str).str.lower().eq("percent").to_numpy()
    discount = pd.to_numeric(eligible.discount_value, errors="coerce").fillna(0).to_numpy()
    sensitivity = {"low": 0.1, "medium": 0.35, "high": 0.7}.get(user.discount_sensitivity, 0.35)
    scores += percent * np.clip(discount / 50.0, 0, 1) * sensitivity

    if rng.random() < float(user.exploration_probability):
        # Explore outside known interests where possible; otherwise sample broadly.
        outside = (~eligible.store_id.isin(favorite_stores)
                   & ~eligible.store_category.isin([user.primary_store_category, user.secondary_store_category])
                   & ~eligible.promotion_category.isin(favorite_products)).to_numpy()
        if outside.any():
            scores = outside.astype(float)
        else:
            scores = np.ones(len(eligible), dtype=float)
    return scores / scores.sum()



def choose_event(user, promo, rng):
    """Choose an interaction type based on user preferences."""
    favorite_stores = parse_ids(user.preferred_store_ids)
    favorite_products = parse_categories(user.preferred_product_categories)

    affinity = 0.0

    if promo.store_category == user.primary_store_category:
        affinity += 0.8
    elif promo.store_category == user.secondary_store_category:
        affinity += 0.4

    if int(promo.store_id) in favorite_stores:
        affinity += 0.8

    if promo.promotion_category in favorite_products:
        affinity += 0.5

    if promo.promotion_type == user.preferred_promotion_type:
        affinity += 0.3

    # Stronger affinity increases the probability of engagement.
    # These are simulation assumptions, not real customer statistics.
    engagement = min(affinity / 2.4, 1.0)

    purchase_prob = 0.015 + 0.035 * engagement
    save_prob = 0.045 + 0.075 * engagement
    click_prob = 0.18 + 0.12 * engagement
    view_prob = 1 - purchase_prob - save_prob - click_prob

    return rng.choice(
        ["view", "click", "save", "purchase"],
        p=[view_prob, click_prob, save_prob, purchase_prob]
    )


def generate_interactions():
    """Generate synthetic interactions and save them to CSV."""

    rng = np.random.default_rng(SEED)

    users, promos = read_data()

    interactions = []

    # Assign more interactions to more active shoppers.
    activity = users["shopping_activity_level"].map(
        ACTIVITY_WEIGHTS
    ).fillna(1.0).to_numpy(dtype=float)

    activity = activity / activity.sum()

    # Distribute exactly 32,000 interactions among users.
    counts = rng.multinomial(TARGET_INTERACTIONS, activity)

    for user, count in zip(users.itertuples(index=False), counts):

        created_at = pd.Timestamp(user.created_at)

        # Only promotions with an eligible date after account creation.
        available_promos = promos[
            promos["eligible_end"] >= created_at
        ].copy()

        if available_promos.empty:
            if count > 0:
                raise ValueError(
                    f"No eligible promotions for user {user.user_id}"
                )
            continue

        for _ in range(int(count)):

            # Choose a promotion using user preferences.
            probabilities = promotion_scores(
                user, available_promos, rng
            )

            selected_index = rng.choice(
                len(available_promos),
                p=probabilities
            )

            promo = available_promos.iloc[selected_index]

            # An interaction cannot happen before account creation
            # or before the promotion starts.
            start_time = max(
                created_at,
                pd.Timestamp(promo.start_date)
            )

            end_time = pd.Timestamp(promo.eligible_end)

            if start_time > end_time:
                raise ValueError("Invalid interaction date range")

            # Random interaction timestamp within the valid window.
            seconds = int((end_time - start_time).total_seconds())

            random_seconds = int(
                rng.integers(0, seconds + 1)
            )

            interaction_time = (
                start_time + pd.Timedelta(seconds=random_seconds)
            )

            event_type = choose_event(user, promo, rng)

            interactions.append({
                "user_id": int(user.user_id),
                "promotion_id": int(promo.promotion_id),
                "interaction_type": event_type,
                "interaction_timestamp": interaction_time
            })

    # Convert generated records into a DataFrame.
    interactions_df = pd.DataFrame(interactions)

    # Sort interactions chronologically.
    interactions_df = interactions_df.sort_values(
        "interaction_timestamp"
    ).reset_index(drop=True)

    # Assign unique interaction IDs.
    interactions_df.insert(
        0,
        "interaction_id",
        np.arange(1, len(interactions_df) + 1)
    )

    # Validate record count.
    assert len(interactions_df) == TARGET_INTERACTIONS

    # Save CSV in the same folder as this Python script.
    output_path = BASE_DIR / "interactions.csv"

    interactions_df.to_csv(output_path, index=False)

    print("Interaction generation completed!")
    print(f"Total interactions: {len(interactions_df):,}")
    print(f"Unique users: {interactions_df['user_id'].nunique()}")
    print(f"Unique promotions: {interactions_df['promotion_id'].nunique()}")

    print("\nInteraction distribution:")
    print(interactions_df["interaction_type"].value_counts())

    print(f"\nFile saved to: {output_path}")


def main():
    generate_interactions()


if __name__ == "__main__":
    main()

