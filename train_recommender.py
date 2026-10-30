"""
Train a hybrid neural recommender on simulated promotion interactions.
The synthetic preference file is deliberately NOT read.
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
PREPARED = BASE / 'prepared_data'
OUT = BASE / 'model_artifacts'
SEED = 2026
EPOCHS = 12
BATCH_SIZE = 512
NEGATIVES_PER_POSITIVE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_and_validate():
    users = pd.read_csv(BASE / 'users.csv', parse_dates=['created_at'])
    stores = pd.read_csv(BASE / 'stores.csv')
    promos = pd.read_csv(BASE / 'promotions_with_categories.csv', parse_dates=['start_date', 'end_date'])
    events = pd.read_csv(BASE / 'interactions.csv', parse_dates=['interaction_timestamp'])
    for name, df, key in [('users', users, 'user_id'), ('stores', stores, 'store_id'),
                          ('promotions', promos, 'promotion_id'), ('interactions', events, 'interaction_id')]:
        if df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f'{name}: invalid or duplicate {key}')
    if not events.interaction_type.isin(['view', 'click', 'save', 'purchase']).all():
        raise ValueError('Unknown interaction type')
    if not set(promos.store_id).issubset(set(stores.store_id)):
        raise ValueError('Promotions reference unknown stores')
    if not set(events.user_id).issubset(set(users.user_id)):
        raise ValueError('Interactions reference unknown users')
    if not set(events.promotion_id).issubset(set(promos.promotion_id)):
        raise ValueError('Interactions reference unknown promotions')
    if promos[['store_id', 'promotion_category', 'promotion_type', 'discount_value', 'discount_unit']].isna().any().any():
        raise ValueError('Missing promotion features')
    promos = promos.drop(columns=['store_category'], errors='ignore').merge(
        stores[['store_id', 'category']].rename(columns={'category': 'store_category'}),
        on='store_id', validate='many_to_one')
    checked = events.merge(users, on='user_id', validate='many_to_one').merge(
        promos[['promotion_id', 'start_date', 'end_date']], on='promotion_id', validate='many_to_one')
    dates = checked.interaction_timestamp.dt.normalize()
    if ((checked.interaction_timestamp < checked.created_at) |
        (dates < checked.start_date) | (dates > checked.end_date)).any():
        raise ValueError('Interactions before signup or outside promotion dates')
    events = events.sort_values(['interaction_timestamp', 'interaction_id']).reset_index(drop=True)
    return users, promos, events


def make_features(users, promos):
    # IDs/categories are encoded deterministically; mapping is saved for inference.
    maps = {}
    for key, vals in [('user_id', users.user_id), ('promotion_id', promos.promotion_id),
                      ('store_id', promos.store_id), ('store_category', promos.store_category),
                      ('promotion_category', promos.promotion_category),
                      ('promotion_type', promos.promotion_type), ('discount_unit', promos.discount_unit)]:
        maps[key] = {str(v): i for i, v in enumerate(sorted(vals.dropna().unique(), key=str))}
    features = promos.copy()
    for key in ['promotion_id', 'store_id', 'store_category', 'promotion_category', 'promotion_type', 'discount_unit']:
        features[key + '_idx'] = features[key].astype(str).map(maps[key]).astype('int64')
    # Discount is normalized using only known catalog metadata, not future engagement outcomes.
    features['discount_scaled'] = np.log1p(pd.to_numeric(features.discount_value).clip(lower=0)) / np.log1p(100)
    features = features.set_index('promotion_id', drop=False)
    return maps, features


def eligible_ids(promos, timestamp):
    # A promotion must exist and be active at the sampled interaction time.
    day = pd.Timestamp(timestamp).normalize()
    return promos.index[(promos.start_date <= day) & (promos.end_date >= day)].to_numpy()


def make_examples(events, promos, rng, negatives=2):
    """Positive = click/save/purchase; unobserved eligible items are sampled as weak negatives.

    Unknown items are NOT verified dislikes; these are sampled ranking proxies.
    """
    positives = events[events.interaction_type.isin(['click', 'save', 'purchase'])]
    observed = events.groupby('user_id').promotion_id.apply(set).to_dict()
    rows = []
    for event in positives.itertuples(index=False):
        rows.append((int(event.user_id), int(event.promotion_id), 1.0))
        choices = eligible_ids(promos, event.interaction_timestamp)
        choices = [int(p) for p in choices if int(p) not in observed.get(event.user_id, set())]
        if choices:
            sample = rng.choice(choices, size=min(negatives, len(choices)), replace=False)
            rows.extend((int(event.user_id), int(p), 0.0) for p in sample)
    if not rows:
        raise ValueError('No strong engagement events in this split')
    return pd.DataFrame(rows, columns=['user_id', 'promotion_id', 'label'])


class HybridRecommender(nn.Module):
    def __init__(self, sizes, dim=16):
        super().__init__()
        self.user = nn.Embedding(sizes['user_id'], dim)
        self.promotion = nn.Embedding(sizes['promotion_id'], dim)
        self.store = nn.Embedding(sizes['store_id'], 8)
        self.store_category = nn.Embedding(sizes['store_category'], 8)
        self.product_category = nn.Embedding(sizes['promotion_category'], 8)
        self.promo_type = nn.Embedding(sizes['promotion_type'], 4)
        self.discount_unit = nn.Embedding(sizes['discount_unit'], 4)
        self.net = nn.Sequential(nn.Linear(dim * 2 + 8 * 3 + 4 * 2 + 1, 64),
                                 nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 32),
                                 nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = torch.cat([self.user(x[:, 0].long()), self.promotion(x[:, 1].long()),
                       self.store(x[:, 2].long()), self.store_category(x[:, 3].long()),
                       self.product_category(x[:, 4].long()), self.promo_type(x[:, 5].long()),
                       self.discount_unit(x[:, 6].long()), x[:, 7:8].float()], dim=1)
        return self.net(z).squeeze(1)


FEATURE_COLS = ['user_id_idx', 'promotion_id_idx', 'store_id_idx', 'store_category_idx',
                'promotion_category_idx', 'promotion_type_idx', 'discount_unit_idx', 'discount_scaled']


def encode_examples(examples, features, maps):
    joined = examples.merge(features[FEATURE_COLS[1:]], left_on='promotion_id',
                            right_index=True, validate='many_to_one')
    joined['user_id_idx'] = joined.user_id.astype(str).map(maps['user_id'])
    if joined[FEATURE_COLS].isna().any().any():
        raise ValueError('Feature encoding produced missing values')
    return torch.tensor(joined[FEATURE_COLS].to_numpy(dtype=np.float32)), torch.tensor(
        joined.label.to_numpy(dtype=np.float32))


def evaluate_loss(model, loader, criterion):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for x, y in loader:
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            total += float(loss) * len(y)
            count += len(y)
    return total / max(count, 1)


def ranking_metrics(model, history, holdout, promos, features, maps, k=5):
    """Recall@K and HitRate@K on future strong interactions and time-eligible items.

    For each user, evaluate at their first strong holdout event time. Exclude
    items previously observed in history; future items are never used as features.
    """
    history_seen = history.groupby('user_id').promotion_id.apply(set).to_dict()
    positives = holdout[holdout.interaction_type.isin(['click', 'save', 'purchase'])]
    scores = []
    model.eval()
    with torch.no_grad():
        for user_id, group in positives.groupby('user_id'):
            at = group.interaction_timestamp.min()
            relevant = set(group.loc[group.interaction_timestamp >= at, 'promotion_id'].astype(int))
            candidates = [int(p) for p in eligible_ids(promos, at)
                          if int(p) not in history_seen.get(user_id, set())]
            relevant &= set(candidates)
            if not relevant or not candidates:
                continue
            batch = pd.DataFrame({'user_id': [user_id] * len(candidates),
                                  'promotion_id': candidates, 'label': 0.})
            x, _ = encode_examples(batch, features, maps)
            pred = model(x.to(DEVICE)).cpu().numpy()
            top = set(np.array(candidates)[np.argsort(-pred)[:k]].tolist())
            hits = len(top & relevant)
            scores.append((hits / k, hits / len(relevant), float(hits > 0)))
    if not scores:
        return {'users_evaluated': 0, f'precision@{k}': None,
                f'recall@{k}': None, f'hit_rate@{k}': None}
    arr = np.array(scores)
    return {'users_evaluated': len(scores), f'precision@{k}': round(float(arr[:, 0].mean()), 4),
            f'recall@{k}': round(float(arr[:, 1].mean()), 4),
            f'hit_rate@{k}': round(float(arr[:, 2].mean()), 4)}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # Use the exact splits and encodings produced by prepare_training_data.py.
    required = ['train.csv', 'validation.csv', 'test.csv', 'feature_mappings.json']
    for name in required:
        if not (PREPARED / name).exists():
            raise FileNotFoundError(f'Missing {PREPARED / name}; run prepare_training_data.py first')
    metadata = json.loads((PREPARED / 'feature_mappings.json').read_text(encoding='utf-8'))
    maps = metadata['categorical_vocabularies']
    train = pd.read_csv(PREPARED / 'train.csv', parse_dates=['interaction_timestamp'])
    valid = pd.read_csv(PREPARED / 'validation.csv', parse_dates=['interaction_timestamp'])
    test = pd.read_csv(PREPARED / 'test.csv', parse_dates=['interaction_timestamp'])
    for name, df in [('train', train), ('validation', valid), ('test', test)]:
        if df.empty or df[FEATURE_COLS[:-1] + ['discount_value_scaled']].isna().any().any():
            raise ValueError(f'{name}: empty data or missing encoded features')
        print(f'{name}: {len(df):,} interactions')

    # Catalog metadata is public to the recommender; hidden user preferences are never loaded.
    stores = pd.read_csv(BASE / 'stores.csv')
    promos = pd.read_csv(BASE / 'promotions_with_categories.csv', parse_dates=['start_date', 'end_date'])
    promos = promos.drop(columns=['store_category'], errors='ignore').merge(
        stores[['store_id', 'category']].rename(columns={'category': 'store_category'}),
        on='store_id', validate='many_to_one')
    for key in ['promotion_id', 'store_id', 'store_category', 'promotion_category',
                'promotion_type', 'discount_unit']:
        promos[key + '_idx'] = promos[key].astype(str).map(maps[key]).fillna(0).astype('int64')
    promos['discount_scaled'] = ((pd.to_numeric(promos.discount_value) -
                                 metadata['discount_value_mean_train']) /
                                 metadata['discount_value_std_train'])
    features = promos.set_index('promotion_id', drop=False)

    # Observed click/save/purchase = positive. Unobserved active offers = sampled unknowns.
    # These unknowns are NOT confirmed dislikes. Views remain in history, not positive labels.
    train_ex = make_examples(train, features, rng, NEGATIVES_PER_POSITIVE)
    valid_ex = make_examples(valid, features, rng, NEGATIVES_PER_POSITIVE)
    x_train, y_train = encode_examples(train_ex, features, maps)
    x_valid, y_valid = encode_examples(valid_ex, features, maps)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=BATCH_SIZE)
    model = HybridRecommender({key: len(val) for key, val in maps.items()}).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_state = float('inf'), None
    print(f'Device: {DEVICE}; training pairs: {len(train_ex):,}')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val_loss = evaluate_loss(model, valid_loader, criterion)
        print(f'Epoch {epoch:02d}/{EPOCHS}: train_loss={total / len(train_ex):.4f} '
              f'val_loss={val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # Test remains untouched until model selection is complete.
    metrics = ranking_metrics(model, pd.concat([train, valid], ignore_index=True),
                              test, features, features, maps, k=5)
    torch.save({'state_dict': best_state,
                'embedding_sizes': {k: len(v) for k, v in maps.items()},
                'feature_columns': FEATURE_COLS,
                'architecture': 'HybridRecommender',
                'best_validation_loss': best_loss}, OUT / 'hybrid_recommender.pt')
    (OUT / 'feature_mappings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (OUT / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Test ranking metrics:', metrics)
    print('Saved model to:', OUT / 'hybrid_recommender.pt')
    print('Important: results describe synthetic behavior, not verified real-world accuracy.')


if __name__ == '__main__':
    main()
