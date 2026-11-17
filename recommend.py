
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch

from train_history_recommender import (
    HistoryRecommender,
    ITEM_KEYS,
    MAX_HISTORY,
    history_arrays,
)

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "model_artifacts"
PREPARED = BASE / "prepared_data"

EVENT_TYPES = {"view", "click", "save", "purchase"}


class PromotionRecommender:
    def __init__(self):
        metadata = json.loads(
            (PREPARED / "feature_mappings.json").read_text(
                encoding="utf-8"
            )
        )
        self.maps = metadata["categorical_vocabularies"]

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
            columns=["store_category"],
            errors="ignore",
        ).merge(
            self.stores[
                ["store_id", "store_name", "category"]
            ].rename(
                columns={"category": "store_category"}
            ),
            on="store_id",
            validate="many_to_one",
        )

        self.promotions = self.promotions.sort_values(
            "promotion_id"
        ).reset_index(drop=True)

        self.promotions["item_idx"] = np.arange(
            1, len(self.promotions) + 1
        )

        self.lookup = dict(
            zip(
                self.promotions["promotion_id"].astype(int),
                self.promotions["item_idx"].astype(int),
            )
        )

        categorical = np.zeros(
            (len(self.promotions) + 1, len(ITEM_KEYS)),
            dtype=np.int64,
        )

        numeric = np.zeros(
            (len(self.promotions) + 1, 1),
            dtype=np.float32,
        )

        mean = metadata["discount_value_mean_train"]
        std = metadata["discount_value_std_train"]

        for record in self.promotions.itertuples(index=False):
            idx = int(record.item_idx)

            for j, key in enumerate(ITEM_KEYS):
                categorical[idx, j] = self.maps[key].get(
                    str(getattr(record, key)), 0
                )

            numeric[idx, 0] = (
                float(record.discount_value) - mean
            ) / std

        checkpoint = torch.load(
            ARTIFACTS / "history_recommender.pt",
            map_location="cpu",
            weights_only=True,
        )

        self.model = HistoryRecommender(
            checkpoint["embedding_sizes"],
            categorical,
            numeric,
        )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.max_history = checkpoint.get(
            "max_history", MAX_HISTORY
        )

        if self.max_history != MAX_HISTORY:
            raise ValueError(
                "MAX_HISTORY does not match training configuration."
            )

    def valid_user(self, user_id):
        return int(user_id) in set(self.users["user_id"])

    def valid_promotion(self, promotion_id, as_of):
        day = pd.Timestamp(as_of).normalize()

        matching = self.promotions.loc[
            (self.promotions["promotion_id"] == int(promotion_id))
            & (self.promotions["start_date"] <= day)
            & (self.promotions["end_date"] >= day)
        ]

        return not matching.empty

    def recommend(
        self,
        user_id,
        top_k=10,
        as_of=None,
        extra_interactions=None,
    ):
        user_id = int(user_id)

        if not self.valid_user(user_id):
            raise ValueError(f"User {user_id} does not exist.")

        day = (
            pd.Timestamp(as_of).normalize()
            if as_of
            else pd.Timestamp.today().normalize()
        )

        # Original synthetic history: strictly before demo date.
        previous = self.interactions.loc[
            (self.interactions["user_id"] == user_id)
            & (self.interactions["interaction_timestamp"] < day)
        ][
            [
                "promotion_id",
                "interaction_type",
                "interaction_timestamp",
            ]
        ].copy()

        # New demo activity: include events recorded on demo date.
        if extra_interactions is not None and len(extra_interactions):
            extra = extra_interactions.copy()

            extra["interaction_timestamp"] = pd.to_datetime(
                extra["interaction_timestamp"]
            )

            cutoff = day + pd.Timedelta(days=1)

            extra = extra.loc[
                (extra["user_id"] == user_id)
                & (extra["interaction_timestamp"] < cutoff)
                & (extra["interaction_timestamp"] >= day)
            ]

            extra = extra[
                [
                    "promotion_id",
                    "interaction_type",
                    "interaction_timestamp",
                ]
            ]

            previous = pd.concat(
                [previous, extra],
                ignore_index=True,
            )

        previous = previous.sort_values(
            "interaction_timestamp",
            kind="stable",
        )

        eligible = self.promotions.loc[
            (self.promotions["start_date"] <= day)
            & (self.promotions["end_date"] >= day)
        ].copy()

        purchased = previous.loc[
            previous["interaction_type"] == "purchase",
            "promotion_id",
        ].unique()

        eligible = eligible.loc[
            ~eligible["promotion_id"].isin(purchased)
        ].copy()

        if eligible.empty:
            return eligible

        events = [
            (int(promotion_id), interaction_type)
            for promotion_id, interaction_type in zip(
                previous["promotion_id"],
                previous["interaction_type"],
            )
            if int(promotion_id) in self.lookup
            and interaction_type in EVENT_TYPES
        ]

        hist_ids, hist_weights = history_arrays(
            events, self.lookup
        )

        n = len(eligible)

        candidate_ids = torch.tensor(
            eligible["item_idx"].to_numpy(),
            dtype=torch.long,
        )

        history_ids = torch.tensor(
            np.tile(hist_ids, (n, 1)),
            dtype=torch.long,
        )

        history_weights = torch.tensor(
            np.tile(hist_weights, (n, 1)),
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits = self.model(
                candidate_ids,
                history_ids,
                history_weights,
            )

            eligible["recommendation_score"] = (
                torch.sigmoid(logits).cpu().numpy()
            )

        eligible = eligible.sort_values(
            "recommendation_score",
            ascending=False,
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
    parser.add_argument("--top", type=int, default=10)

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
