
from pathlib import Path
import json

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
PREPARED = BASE / "prepared_data"
ARTIFACTS = BASE / "model_artifacts"

KS = [5, 10]
POSITIVE_EVENTS = {"click", "save", "purchase"}

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train = pd.read_csv(
    PREPARED / "train.csv",
    parse_dates=["interaction_timestamp"]
)

valid = pd.read_csv(
    PREPARED / "validation.csv",
    parse_dates=["interaction_timestamp"]
)

test = pd.read_csv(
    PREPARED / "test.csv",
    parse_dates=["interaction_timestamp"]
)

promos = pd.read_csv(
    BASE / "promotions_with_categories.csv",
    parse_dates=["start_date", "end_date"]
)

promos = promos.set_index("promotion_id", drop=False)

print(f"Training interactions: {len(train):,}")
print(f"Validation interactions: {len(valid):,}")
print(f"Testing interactions: {len(test):,}")
print(f"Catalog promotions: {len(promos):,}")

# --------------------------------------------------
# 3. CALCULATE POPULARITY FROM TRAINING ONLY
# --------------------------------------------------

positive_train = train[
    train["interaction_type"].isin(POSITIVE_EVENTS)
]

popularity = (
    positive_train.groupby("promotion_id")
    .size()
    .to_dict()
)

popularity_scores = {
    int(promotion_id): int(popularity.get(promotion_id, 0))
    for promotion_id in promos.index
}

print(
    "Promotions with positive training engagement:",
    sum(score > 0 for score in popularity_scores.values())
)

# --------------------------------------------------
# 4. BUILD USER HISTORY (TRAIN + VALIDATION)
# --------------------------------------------------

history = pd.concat(
    [train, valid],
    ignore_index=True
)

history_seen = (
    history.groupby("user_id")["promotion_id"]
    .apply(set)
    .to_dict()
)

# --------------------------------------------------
# 5. IDENTIFY POSITIVE TEST INTERACTIONS
# --------------------------------------------------

positive_test = test[
    test["interaction_type"].isin(POSITIVE_EVENTS)
]

# --------------------------------------------------
# 6. GET ACTIVE PROMOTIONS
# --------------------------------------------------

def eligible_ids(timestamp):
    day = pd.Timestamp(timestamp).normalize()

    active = promos[
        (promos["start_date"] <= day)
        & (promos["end_date"] >= day)
    ]

    return active.index.to_numpy()


# --------------------------------------------------
# 7. EVALUATE POPULARITY BASELINE AT K=5 AND K=10
# --------------------------------------------------

scores = {k: [] for k in KS}

for user_id, group in positive_test.groupby("user_id"):

    # Same evaluation timestamp as neural network.
    at = group["interaction_timestamp"].min()

    relevant = set(
        group.loc[
            group["interaction_timestamp"] >= at,
            "promotion_id"
        ].astype(int)
    )

    candidates = [
        int(p)
        for p in eligible_ids(at)
        if int(p) not in history_seen.get(user_id, set())
    ]

    relevant &= set(candidates)

    if not relevant or not candidates:
        continue

    # Rank by historical popularity.
    # Break ties using promotion ID.
    ranked = sorted(
        candidates,
        key=lambda p: (
            -popularity_scores[p],
            p
        )
    )

    for k in KS:

        top_k = set(ranked[:k])

        hits = len(top_k & relevant)

        precision = hits / k
        recall = hits / len(relevant)
        hit_rate = float(hits > 0)

        scores[k].append(
            (precision, recall, hit_rate)
        )

# --------------------------------------------------
# 8. CALCULATE BASELINE METRICS
# --------------------------------------------------

baseline_metrics = {}

for k in KS:

    if not scores[k]:
        raise ValueError(
            f"No eligible users found for K={k}."
        )

    arr = np.array(scores[k])

    baseline_metrics[f"users_evaluated@{k}"] = len(scores[k])

    baseline_metrics[f"precision@{k}"] = round(
        float(arr[:, 0].mean()), 4
    )

    baseline_metrics[f"recall@{k}"] = round(
        float(arr[:, 1].mean()), 4
    )

    baseline_metrics[f"hit_rate@{k}"] = round(
        float(arr[:, 2].mean()), 4
    )

baseline_metrics["users_evaluated"] = (
    baseline_metrics["users_evaluated@5"]
)

print("\nPOPULARITY BASELINE RESULTS")
print("-" * 45)

for metric, value in baseline_metrics.items():
    if metric.startswith("users_evaluated"):
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.4f} ({value:.2%})")

# --------------------------------------------------
# 9. LOAD WEIGHTED NEURAL NETWORK METRICS
# --------------------------------------------------

ncf_path = ARTIFACTS / "test_metrics_weighted.json"

if not ncf_path.exists():
    raise FileNotFoundError(
        f"Weighted model metrics not found: {ncf_path}"
    )

with open(ncf_path, "r", encoding="utf-8") as file:
    ncf_metrics = json.load(file)

# --------------------------------------------------
# 10. COMPARE BOTH APPROACHES
# --------------------------------------------------

metric_names = [
    ("Precision@5", "precision@5"),
    ("Recall@5", "recall@5"),
    ("Hit Rate@5", "hit_rate@5"),
    ("Precision@10", "precision@10"),
    ("Recall@10", "recall@10"),
    ("Hit Rate@10", "hit_rate@10"),
]

comparison = pd.DataFrame({
    "Metric": [
        name for name, key in metric_names
    ],
    "Weighted Hybrid NCF": [
        ncf_metrics[key]
        for name, key in metric_names
    ],
    "Popularity Baseline": [
        baseline_metrics[key]
        for name, key in metric_names
    ],
})

comparison["Difference"] = (
    comparison["Weighted Hybrid NCF"]
    - comparison["Popularity Baseline"]
)

print("\nMODEL COMPARISON")
print("-" * 75)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Weighted Hybrid NCF": "{:.2%}".format,
            "Popularity Baseline": "{:.2%}".format,
            "Difference": "{:+.2%}".format,
        }
    )
)

# --------------------------------------------------
# 11. CHECK USER COUNTS
# --------------------------------------------------

for k in KS:

    ncf_count = ncf_metrics.get(
        f"users_evaluated@{k}",
        ncf_metrics.get("users_evaluated")
    )

    baseline_count = baseline_metrics[
        f"users_evaluated@{k}"
    ]

    print(f"\nK={k}")
    print("NCF users evaluated:", ncf_count)
    print("Baseline users evaluated:", baseline_count)

    if ncf_count == baseline_count:
        print("Evaluation user counts match.")
    else:
        print("WARNING: Evaluation user counts differ.")

# --------------------------------------------------
# 12. SAVE BASELINE METRICS
# --------------------------------------------------

ARTIFACTS.mkdir(
    parents=True,
    exist_ok=True
)

output_path = ARTIFACTS / "baseline_metrics.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        baseline_metrics,
        file,
        indent=4
    )

print(f"\nBaseline metrics saved to: {output_path}")
