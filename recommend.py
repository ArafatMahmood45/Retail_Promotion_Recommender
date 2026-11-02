
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch

from train_recommender import HybridRecommender, FEATURE_COLS

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "model_artifacts"


class PromotionRecommender:
    def __init__(self):
        # Load the exact mappings saved with the trained model.
        with open(ARTIFACTS / "feature_mappings.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.maps = self.metadata["categorical_vocabularies"]

        checkpoint = torch.load(
            ARTIFACTS / "hybrid_recommender.pt",
            map_location="cpu",
            weights_only=True,
        )

        self.model = HybridRecommender(checkpoint["embedding_sizes"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        if checkpoint["feature_columns"] != FEATURE_COLS:
            raise ValueError("Model feature columns do not match training code.")

        self.users = pd.read_csv(BASE / "users.csv")
        self.stores = pd.read_csv(BASE / "stores.csv")
        self.promotions = pd.read_csv(
            BASE / "promotions_with_categories.csv",
            parse_dates=["start_date", "end_date"],
        )
        self.interactions = pd.read_csv(
            BASE / "interactions.csv",
            parse_dates=["interaction_timestamp"],
        )

        self.promotions = self.promotions.drop(
            columns=["store_category"], errors="ignore"
        ).merge(
            self.stores[["store_id", "store_name", "category"]].rename(
                columns={"category": "store_category"}
            ),
            on="store_id",
            validate="many_to_one",
        )

    def recommend(self, user_id, top_k=5, as_of=None):
        user_id = int(user_id)

        if user_id not in set(self.users["user_id"]):
            raise ValueError(f"User {user_id} does not exist.")

        if str(user_id) not in self.maps["user_id"]:
            raise ValueError("User was not included in the trained model.")

        date = pd.Timestamp(as_of).normalize() if as_of else pd.Timestamp.today().normalize()

        # Only consider promotions active on the requested date.
        eligible = self.promotions.loc[
            (self.promotions["start_date"] <= date)
            & (self.promotions["end_date"] >= date)
            ].copy()

        # Find promotions purchased by this user on or before the selected date.
        # Include the entire selected day, not just midnight.
        cutoff = date + pd.Timedelta(days=1)

        purchased_promotions = self.interactions.loc[
            (self.interactions["user_id"] == user_id)
            & (self.interactions["interaction_type"] == "purchase")
            & (self.interactions["interaction_timestamp"] < cutoff),
            "promotion_id",
        ].unique()

        # Remove purchased promotions before scoring and ranking.
        eligible = eligible.loc[
            ~eligible["promotion_id"].isin(purchased_promotions)
        ].copy()

        if eligible.empty:
            return eligible

        # Use exactly the same encodings as model training.
        keys = [
            "promotion_id",
            "store_id",
            "store_category",
            "promotion_category",
            "promotion_type",
            "discount_unit",
        ]

        for key in keys:
            eligible[key + "_idx"] = (
                eligible[key].astype(str)
                .map(self.maps[key])
                .fillna(0)
                .astype("int64")
            )

        mean = self.metadata["discount_value_mean_train"]
        std = self.metadata["discount_value_std_train"]

        eligible["discount_scaled"] = (
            pd.to_numeric(eligible["discount_value"]) - mean
        ) / std

        eligible["user_id_idx"] = self.maps["user_id"][str(user_id)]

        features = torch.tensor(
            eligible[FEATURE_COLS].to_numpy(dtype=np.float32)
        )

        with torch.no_grad():
            logits = self.model(features)
            eligible["recommendation_score"] = (
                torch.sigmoid(logits).numpy()
            )

        eligible = eligible.sort_values(
            "recommendation_score", ascending=False
        )

        return eligible[
            [
                "promotion_id",
                "store_name",
                "promotion_category",
                "promotion_type",
                "discount_unit",
                "discount_value",
                "start_date",
                "end_date",
                "recommendation_score",
            ]
        ].head(top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    recommender = PromotionRecommender()
    results = recommender.recommend(
        user_id=args.user,
        top_k=args.top,
        as_of=args.date,
    )

    if results.empty:
        print("No active promotions found for the selected date.")
    else:
        print(results.to_string(index=False))
