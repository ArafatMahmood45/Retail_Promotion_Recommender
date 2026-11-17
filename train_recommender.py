
"""
Experiment 3: Weighted interaction training.

Original hybrid recommender with promotion-ID embeddings.

Click weight = 1
Save weight = 2
Purchase weight = 3
Sampled negative weight = 1

Evaluation remains based on clicks, saves and purchases
as equally relevant positive interactions.

The synthetic user preference file is NOT read.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE = Path(__file__).resolve().parent
PREPARED = BASE / "prepared_data"
OUT = BASE / "model_artifacts"

SEED = 2026
EPOCHS = 12
BATCH_SIZE = 512
NEGATIVES_PER_POSITIVE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INTERACTION_WEIGHTS = {
    "click": 1.0,
    "save": 2.0,
    "purchase": 3.0,
}

NEGATIVE_WEIGHT = 1.0

FEATURE_COLS = [
    "user_id_idx",
    "promotion_id_idx",
    "store_id_idx",
    "store_category_idx",
    "promotion_category_idx",
    "promotion_type_idx",
    "discount_unit_idx",
    "discount_scaled",
]


def eligible_ids(promos, timestamp):
    day = pd.Timestamp(timestamp).normalize()

    return promos.index[
        (promos.start_date <= day)
        & (promos.end_date >= day)
    ].to_numpy()


def make_examples(events, promos, rng, negatives=2):
    """
    Generate positive and sampled negative examples.

    Each positive receives a training weight determined by
    its interaction type. Negative examples receive weight 1.
    """

    positives = events[
        events.interaction_type.isin(
            ["click", "save", "purchase"]
        )
    ]

    observed = (
        events.groupby("user_id")
        .promotion_id.apply(set)
        .to_dict()
    )

    rows = []

    for event in positives.itertuples(index=False):
        user_id = int(event.user_id)
        promotion_id = int(event.promotion_id)

        weight = INTERACTION_WEIGHTS[
            event.interaction_type
        ]

        rows.append(
            (user_id, promotion_id, 1.0, weight)
        )

        choices = eligible_ids(
            promos, event.interaction_timestamp
        )

        choices = [
            int(p)
            for p in choices
            if int(p) not in observed.get(user_id, set())
        ]

        if choices:
            sample = rng.choice(
                choices,
                size=min(negatives, len(choices)),
                replace=False,
            )

            rows.extend(
                (
                    user_id,
                    int(p),
                    0.0,
                    NEGATIVE_WEIGHT,
                )
                for p in sample
            )

    if not rows:
        raise ValueError(
            "No positive engagement events in this split."
        )

    return pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "promotion_id",
            "label",
            "weight",
        ],
    )


class HybridRecommender(nn.Module):
    """
    Original architecture with promotion-ID embeddings.
    """

    def __init__(self, sizes, dim=16):
        super().__init__()

        self.user = nn.Embedding(
            sizes["user_id"], dim
        )

        self.promotion = nn.Embedding(
            sizes["promotion_id"], dim
        )

        self.store = nn.Embedding(
            sizes["store_id"], 8
        )

        self.store_category = nn.Embedding(
            sizes["store_category"], 8
        )

        self.product_category = nn.Embedding(
            sizes["promotion_category"], 8
        )

        self.promo_type = nn.Embedding(
            sizes["promotion_type"], 4
        )

        self.discount_unit = nn.Embedding(
            sizes["discount_unit"], 4
        )

        input_dim = dim * 2 + 8 * 3 + 4 * 2 + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = torch.cat(
            [
                self.user(x[:, 0].long()),
                self.promotion(x[:, 1].long()),
                self.store(x[:, 2].long()),
                self.store_category(x[:, 3].long()),
                self.product_category(x[:, 4].long()),
                self.promo_type(x[:, 5].long()),
                self.discount_unit(x[:, 6].long()),
                x[:, 7:8].float(),
            ],
            dim=1,
        )

        return self.net(z).squeeze(1)


def encode_examples(examples, features, maps):
    joined = examples.merge(
        features[FEATURE_COLS[1:]],
        left_on="promotion_id",
        right_index=True,
        validate="many_to_one",
    )

    joined["user_id_idx"] = (
        joined.user_id.astype(str)
        .map(maps["user_id"])
    )

    if joined[FEATURE_COLS].isna().any().any():
        raise ValueError(
            "Feature encoding produced missing values."
        )

    x = torch.tensor(
        joined[FEATURE_COLS].to_numpy(
            dtype=np.float32
        )
    )

    y = torch.tensor(
        joined.label.to_numpy(
            dtype=np.float32
        )
    )

    weights = torch.tensor(
        joined.weight.to_numpy(
            dtype=np.float32
        )
    )

    return x, y, weights


def weighted_loss(predictions, targets, weights, criterion):
    """
    Apply interaction weights to individual BCE losses.
    """

    individual_losses = criterion(
        predictions, targets
    )

    return (
        individual_losses * weights
    ).sum() / weights.sum()


def evaluate_loss(model, loader, criterion):
    """
    Use the original unweighted validation loss
    for model selection, making validation results
    comparable to the baseline.
    """

    model.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for x, y, weights in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            predictions = model(x)

            individual_losses = criterion(
                predictions, y
            )

            total += individual_losses.sum().item()
            count += len(y)

    return total / max(count, 1)


def ranking_metrics(
    model,
    history,
    holdout,
    promos,
    features,
    maps,
    ks=(5, 10),
):
    """
    Calculate Precision, Recall and Hit Rate at
    both Top 5 and Top 10.

    Evaluation rules are unchanged from the baseline.
    """

    history_seen = (
        history.groupby("user_id")
        .promotion_id.apply(set)
        .to_dict()
    )

    positives = holdout[
        holdout.interaction_type.isin(
            ["click", "save", "purchase"]
        )
    ]

    scores = {k: [] for k in ks}

    model.eval()

    with torch.no_grad():
        for user_id, group in positives.groupby("user_id"):
            at = group.interaction_timestamp.min()

            relevant = set(
                group.loc[
                    group.interaction_timestamp >= at,
                    "promotion_id",
                ].astype(int)
            )

            candidates = [
                int(p)
                for p in eligible_ids(promos, at)
                if int(p) not in history_seen.get(
                    user_id, set()
                )
            ]

            relevant &= set(candidates)

            if not relevant or not candidates:
                continue

            batch = pd.DataFrame(
                {
                    "user_id": [user_id] * len(candidates),
                    "promotion_id": candidates,
                    "label": 0.0,
                    "weight": 1.0,
                }
            )

            x, _, _ = encode_examples(
                batch, features, maps
            )

            predictions = (
                model(x.to(DEVICE))
                .cpu()
                .numpy()
            )

            ranked = np.array(candidates)[
                np.argsort(-predictions)
            ]

            for k in ks:
                top = set(ranked[:k].tolist())
                hits = len(top & relevant)

                scores[k].append(
                    (
                        hits / k,
                        hits / len(relevant),
                        float(hits > 0),
                    )
                )

    metrics = {}

    for k in ks:
        if not scores[k]:
            metrics[f"users_evaluated@{k}"] = 0
            metrics[f"precision@{k}"] = None
            metrics[f"recall@{k}"] = None
            metrics[f"hit_rate@{k}"] = None
            continue

        arr = np.array(scores[k])

        metrics[f"users_evaluated@{k}"] = len(scores[k])

        metrics[f"precision@{k}"] = round(
            float(arr[:, 0].mean()), 4
        )

        metrics[f"recall@{k}"] = round(
            float(arr[:, 1].mean()), 4
        )

        metrics[f"hit_rate@{k}"] = round(
            float(arr[:, 2].mean()), 4
        )

    metrics["users_evaluated"] = metrics.get(
        "users_evaluated@5", 0
    )

    return metrics


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rng = np.random.default_rng(SEED)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = [
        "train.csv",
        "validation.csv",
        "test.csv",
        "feature_mappings.json",
    ]

    for name in required:
        if not (PREPARED / name).exists():
            raise FileNotFoundError(
                f"Missing {PREPARED / name}; "
                "run prepare_training_data.py first"
            )

    metadata = json.loads(
        (PREPARED / "feature_mappings.json")
        .read_text(encoding="utf-8")
    )

    maps = metadata["categorical_vocabularies"]

    train = pd.read_csv(
        PREPARED / "train.csv",
        parse_dates=["interaction_timestamp"],
    )

    valid = pd.read_csv(
        PREPARED / "validation.csv",
        parse_dates=["interaction_timestamp"],
    )

    test = pd.read_csv(
        PREPARED / "test.csv",
        parse_dates=["interaction_timestamp"],
    )

    for name, df in [
        ("train", train),
        ("validation", valid),
        ("test", test),
    ]:
        required_features = (
            FEATURE_COLS[:-1]
            + ["discount_value_scaled"]
        )

        if (
            df.empty
            or df[required_features].isna().any().any()
        ):
            raise ValueError(
                f"{name}: empty data or missing "
                "encoded features"
            )

        print(
            f"{name}: {len(df):,} interactions"
        )

    stores = pd.read_csv(
        BASE / "stores.csv"
    )

    promos = pd.read_csv(
        BASE / "promotions_with_categories.csv",
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    promos = promos.drop(
        columns=["store_category"],
        errors="ignore",
    ).merge(
        stores[
            ["store_id", "category"]
        ].rename(
            columns={
                "category": "store_category"
            }
        ),
        on="store_id",
        validate="many_to_one",
    )

    for key in [
        "promotion_id",
        "store_id",
        "store_category",
        "promotion_category",
        "promotion_type",
        "discount_unit",
    ]:
        promos[key + "_idx"] = (
            promos[key]
            .astype(str)
            .map(maps[key])
            .fillna(0)
            .astype("int64")
        )

    promos["discount_scaled"] = (
        (
            pd.to_numeric(promos.discount_value)
            - metadata["discount_value_mean_train"]
        )
        / metadata["discount_value_std_train"]
    )

    features = promos.set_index(
        "promotion_id",
        drop=False,
    )

    # Generate weighted training examples.
    train_ex = make_examples(
        train,
        features,
        rng,
        NEGATIVES_PER_POSITIVE,
    )

    valid_ex = make_examples(
        valid,
        features,
        rng,
        NEGATIVES_PER_POSITIVE,
    )

    x_train, y_train, w_train = encode_examples(
        train_ex, features, maps
    )

    x_valid, y_valid, w_valid = encode_examples(
        valid_ex, features, maps
    )

    train_loader = DataLoader(
        TensorDataset(
            x_train, y_train, w_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TensorDataset(
            x_valid, y_valid, w_valid
        ),
        batch_size=BATCH_SIZE,
    )

    # Original architecture.
    model = HybridRecommender(
        {
            key: len(value)
            for key, value in maps.items()
        }
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    # Return one loss per example so it can be weighted.
    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    best_loss = float("inf")
    best_state = None

    print(f"\nDevice: {DEVICE}")
    print(f"Training pairs: {len(train_ex):,}")
    print(
        "Model: Original Hybrid NCF "
        "with weighted interactions"
    )

    print(
        "Interaction weights:",
        INTERACTION_WEIGHTS,
    )

    # Train the model.
    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_weighted_loss = 0.0
        total_weight = 0.0

        for x, y, weights in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            weights = weights.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(x)

            loss = weighted_loss(
                predictions,
                y,
                weights,
                criterion,
            )

            loss.backward()
            optimizer.step()

            total_weighted_loss += (
                loss.item() * weights.sum().item()
            )

            total_weight += weights.sum().item()

        train_loss = (
            total_weighted_loss / total_weight
        )

        val_loss = evaluate_loss(
            model,
            valid_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    history = pd.concat(
        [train, valid],
        ignore_index=True,
    )

    metrics = ranking_metrics(
        model=model,
        history=history,
        holdout=test,
        promos=features,
        features=features,
        maps=maps,
        ks=(5, 10),
    )

    # Save separately to preserve the original baseline.
    model_path = OUT / "hybrid_recommender_weighted.pt"
    metrics_path = OUT / "test_metrics_weighted.json"

    torch.save(
        {
            "state_dict": best_state,
            "embedding_sizes": {
                key: len(value)
                for key, value in maps.items()
            },
            "feature_columns": FEATURE_COLS,
            "architecture": "HybridRecommender",
            "best_validation_loss": best_loss,
            "interaction_weights": INTERACTION_WEIGHTS,
        },
        model_path,
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nTEST RANKING METRICS")
    print("-" * 45)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved weighted model to:", model_path)
    print("Saved weighted metrics to:", metrics_path)

    print(
        "Important: results describe synthetic behavior, "
        "not verified real-world accuracy."
    )


if __name__ == "__main__":
    main()
