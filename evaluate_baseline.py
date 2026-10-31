
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

K = 5
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

# New promotions receive zero popularity.
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
# 6. GET PROMOTIONS ACTIVE AT A GIVEN TIME
# --------------------------------------------------

def eligible_ids(timestamp):
    day = pd.Timestamp(timestamp).normalize()

    active = promos[
        (promos["start_date"] <= day)
        & (promos["end_date"] >= day)
    ]

    return active.index.to_numpy()


# --------------------------------------------------
# 7. EVALUATE POPULARITY RECOMMENDATIONS
# --------------------------------------------------

scores = []
users_evaluated = []

for user_id, group in positive_test.groupby("user_id"):

    # Match the NCF evaluation timestamp.
    at = group["interaction_timestamp"].min()

    # Relevant test promotions.
    relevant = set(
        group.loc[
            group["interaction_timestamp"] >= at,
            "promotion_id"
        ].astype(int)
    )

    # Only active promotions not seen in history.
    candidates = [
        int(p)
        for p in eligible_ids(at)
        if int(p) not in history_seen.get(user_id, set())
    ]

    # Match NCF ground-truth filtering.
    relevant &= set(candidates)

    if not relevant or not candidates:
        continue

    # Rank eligible promotions by historical popularity.
    # Stable tie-break: promotion ID.
    ranked = sorted(
        candidates,
        key=lambda p: (
            -popularity_scores[p],
            p
        )
    )

    top_k = set(ranked[:K])

    hits = len(top_k & relevant)

    precision = hits / K
    recall = hits / len(relevant)
    hit_rate = float(hits > 0)

    scores.append((precision, recall, hit_rate))
    users_evaluated.append(user_id)

# --------------------------------------------------
# 8. CALCULATE BASELINE METRICS
# --------------------------------------------------

if not scores:
    raise ValueError(
        "No eligible users found for baseline evaluation."
    )

arr = np.array(scores)

baseline_metrics = {
    "users_evaluated": len(scores),
    f"precision@{K}": round(float(arr[:, 0].mean()), 4),
    f"recall@{K}": round(float(arr[:, 1].mean()), 4),
    f"hit_rate@{K}": round(float(arr[:, 2].mean()), 4),
}

print("\nPOPULARITY BASELINE RESULTS")
print("-" * 45)

for metric, value in baseline_metrics.items():
    if metric == "users_evaluated":
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.4f} ({value:.2%})")

# --------------------------------------------------
# 9. LOAD SAVED HYBRID NCF METRICS
# --------------------------------------------------

ncf_path = ARTIFACTS / "test_metrics.json"

if not ncf_path.exists():
    raise FileNotFoundError(
        f"NCF metrics file not found: {ncf_path}"
    )

with open(ncf_path, "r", encoding="utf-8") as file:
    ncf_metrics = json.load(file)

# --------------------------------------------------
# 10. COMPARE THE TWO APPROACHES
# --------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Precision@5",
        "Recall@5",
        "Hit Rate@5",
    ],
    "Hybrid NCF": [
        ncf_metrics["precision@5"],
        ncf_metrics["recall@5"],
        ncf_metrics["hit_rate@5"],
    ],
    "Popularity Baseline": [
        baseline_metrics["precision@5"],
        baseline_metrics["recall@5"],
        baseline_metrics["hit_rate@5"],
    ],
})

comparison["Difference"] = (
    comparison["Hybrid NCF"]
    - comparison["Popularity Baseline"]
)

print("\nMODEL COMPARISON")
print("-" * 65)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Hybrid NCF": "{:.2%}".format,
            "Popularity Baseline": "{:.2%}".format,
            "Difference": "{:+.2%}".format,
        }
    )
)

print(
    "\nNCF users evaluated:",
    ncf_metrics["users_evaluated"]
)

print(
    "Baseline users evaluated:",
    baseline_metrics["users_evaluated"]
)

if (
    ncf_metrics["users_evaluated"]
    == baseline_metrics["users_evaluated"]
):
    print(
        "\nEvaluation user counts match."
    )
else:
    print(
        "\nWARNING: Evaluation user counts differ."
    )

# --------------------------------------------------
# 11. SAVE BASELINE METRICS
# --------------------------------------------------

ARTIFACTS.mkdir(parents=True, exist_ok=True)

output_path = ARTIFACTS / "baseline_metrics.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(baseline_metrics, file, indent=4)

print(f"\nBaseline metrics saved to: {output_path}")
