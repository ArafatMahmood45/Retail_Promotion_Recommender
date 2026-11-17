from pathlib import Path
from collections import defaultdict
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
BATCH_SIZE = 256
MAX_HISTORY = 40
NEGATIVES_PER_POSITIVE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS = {'view': 0., 'click': 1., 'save': 2., 'purchase': 3.}
ITEM_KEYS = ['store_id', 'store_category', 'promotion_category', 'promotion_type', 'discount_unit']


def load_data():
    metadata = json.loads((PREPARED / 'feature_mappings.json').read_text(encoding='utf-8'))
    maps = metadata['categorical_vocabularies']
    splits = [pd.read_csv(PREPARED / f'{s}.csv', parse_dates=['interaction_timestamp'])
              for s in ('train', 'validation', 'test')]
    stores = pd.read_csv(BASE / 'stores.csv')
    promos = pd.read_csv(BASE / 'promotions_with_categories.csv', parse_dates=['start_date', 'end_date'])
    promos = promos.drop(columns=['store_category'], errors='ignore').merge(
        stores[['store_id', 'category']].rename(columns={'category': 'store_category'}),
        on='store_id', validate='many_to_one')
    promos = promos.sort_values('promotion_id').reset_index(drop=True)
    # Index zero is reserved for padding. The model does not use promotion-ID embeddings.
    promos['item_idx'] = np.arange(1, len(promos) + 1)
    item_lookup = dict(zip(promos.promotion_id.astype(int), promos.item_idx.astype(int)))
    categorical = np.zeros((len(promos) + 1, len(ITEM_KEYS)), dtype=np.int64)
    numeric = np.zeros((len(promos) + 1, 1), dtype=np.float32)
    for row in promos.itertuples(index=False):
        idx = int(row.item_idx)
        for j, key in enumerate(ITEM_KEYS):
            categorical[idx, j] = maps[key].get(str(getattr(row, key)), 0)
        numeric[idx, 0] = (float(row.discount_value) - metadata['discount_value_mean_train']) / metadata['discount_value_std_train']
    return splits, promos, item_lookup, categorical, numeric, maps


def history_arrays(events, lookup):
    # All events supplied here must precede the prediction time.
    weighted = [(lookup[int(p)], WEIGHTS[t]) for p, t in events if WEIGHTS[t] > 0]
    weighted = weighted[-MAX_HISTORY:]
    ids = np.zeros(MAX_HISTORY, dtype=np.int64)
    weights = np.zeros(MAX_HISTORY, dtype=np.float32)
    if weighted:
        ids[-len(weighted):] = [x[0] for x in weighted]
        weights[-len(weighted):] = [x[1] for x in weighted]
    return ids, weights


def eligible_promos(promos, when):
    day = pd.Timestamp(when).normalize()
    return promos.loc[(promos.start_date <= day) & (promos.end_date >= day), 'promotion_id'].astype(int).to_numpy()


def build_examples(split, preceding, promos, lookup, rng):
    # Preceding contains only earlier splits. Events in this split are processed in time order.
    past = defaultdict(list)
    seen = defaultdict(set)
    for event in preceding.sort_values(['interaction_timestamp', 'interaction_id']).itertuples(index=False):
        u, p = int(event.user_id), int(event.promotion_id)
        past[u].append((p, event.interaction_type))
        seen[u].add(p)

    rows = []
    ordered = split.sort_values(['interaction_timestamp', 'interaction_id'])
    # Process all events with the same timestamp together to avoid same-time leakage.
    for when, group in ordered.groupby('interaction_timestamp', sort=True):
        for event in group.itertuples(index=False):
            if event.interaction_type not in ('click', 'save', 'purchase'):
                continue
            u, p = int(event.user_id), int(event.promotion_id)
            h, w = history_arrays(past[u], lookup)
            rows.append((lookup[p], h, w, 1., WEIGHTS[event.interaction_type]))
            options = [q for q in eligible_promos(promos, when)
                       if q not in seen[u] and q != p]
            if options:
                chosen = rng.choice(options, size=min(NEGATIVES_PER_POSITIVE, len(options)), replace=False)
                for q in chosen:
                    rows.append((lookup[int(q)], h, w, 0., 1.))
        for event in group.itertuples(index=False):
            u, p = int(event.user_id), int(event.promotion_id)
            past[u].append((p, event.interaction_type))
            seen[u].add(p)
    if not rows:
        raise ValueError('No positive training examples found')
    return TensorDataset(
        torch.tensor([r[0] for r in rows], dtype=torch.long),
        torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.long),
        torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.float32),
        torch.tensor([r[3] for r in rows], dtype=torch.float32),
        torch.tensor([r[4] for r in rows], dtype=torch.float32),
    )


class HistoryRecommender(nn.Module):
    def __init__(self, sizes, categorical, numeric, dim=24):
        super().__init__()
        self.register_buffer('catalog_categorical', torch.tensor(categorical, dtype=torch.long))
        self.register_buffer('catalog_numeric', torch.tensor(numeric, dtype=torch.float32))
        self.embeddings = nn.ModuleList([nn.Embedding(sizes[k], 8, padding_idx=0) for k in ITEM_KEYS])
        self.item_net = nn.Sequential(nn.Linear(8 * len(ITEM_KEYS) + 1, 64), nn.ReLU(), nn.Linear(64, dim))
        self.scorer = nn.Sequential(nn.Linear(dim * 3, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def item_vectors(self, ids):
        cat = self.catalog_categorical[ids]
        num = self.catalog_numeric[ids]
        parts = [emb(cat[..., j]) for j, emb in enumerate(self.embeddings)]
        return self.item_net(torch.cat(parts + [num], dim=-1))

    def forward(self, item_ids, history_ids, history_weights):
        candidate = self.item_vectors(item_ids)
        history = self.item_vectors(history_ids)
        weights = history_weights.unsqueeze(-1)
        preference = (history * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.)
        # Direct compatibility term, plus a nonlinear learned comparison.
        dot = (candidate * preference).sum(dim=-1) / candidate.shape[-1] ** 0.5
        mlp = self.scorer(torch.cat([candidate, preference, candidate * preference], dim=-1)).squeeze(-1)
        return dot + mlp


def loss_on(model, loader, criterion):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for item, hist, weights, label, sample_weight in loader:
            logits = model(item.to(DEVICE), hist.to(DEVICE), weights.to(DEVICE))
            losses = criterion(logits, label.to(DEVICE))
            total += losses.sum().item()
            count += len(label)
    return total / max(count, 1)


def ranking_metrics(model, train, valid, test, promos, lookup, ks=(5, 10)):
    # Evaluation history is train+validation only; never read test interactions as input.
    history = pd.concat([train, valid], ignore_index=True)
    past = defaultdict(list)
    seen = defaultdict(set)
    for event in history.sort_values(['interaction_timestamp', 'interaction_id']).itertuples(index=False):
        u, p = int(event.user_id), int(event.promotion_id)
        past[u].append((p, event.interaction_type))
        seen[u].add(p)
    positives = test[test.interaction_type.isin(['click', 'save', 'purchase'])]
    results = {k: [] for k in ks}
    model.eval()
    with torch.no_grad():
        for user, group in positives.groupby('user_id'):
            u = int(user)
            when = group.interaction_timestamp.min()
            candidates = [p for p in eligible_promos(promos, when) if p not in seen[u]]
            relevant = set(group.promotion_id.astype(int)) & set(candidates)
            if not candidates or not relevant:
                continue
            h, w = history_arrays(past[u], lookup)
            items = torch.tensor([lookup[p] for p in candidates], dtype=torch.long, device=DEVICE)
            hs = torch.tensor(np.tile(h, (len(candidates), 1)), dtype=torch.long, device=DEVICE)
            ws = torch.tensor(np.tile(w, (len(candidates), 1)), dtype=torch.float32, device=DEVICE)
            scores = model(items, hs, ws).cpu().numpy()
            ranked = np.array(candidates)[np.argsort(-scores)]
            for k in ks:
                hits = len(set(ranked[:k]) & relevant)
                results[k].append((hits / k, hits / len(relevant), float(hits > 0)))
    metrics = {}
    for k in ks:
        arr = np.array(results[k])
        metrics[f'users_evaluated@{k}'] = len(arr)
        for j, name in enumerate(('precision', 'recall', 'hit_rate')):
            metrics[f'{name}@{k}'] = round(float(arr[:, j].mean()), 4) if len(arr) else None
    metrics['users_evaluated'] = metrics['users_evaluated@5']
    return metrics


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (train, valid, test), promos, lookup, categorical, numeric, maps = load_data()
    print(f'train: {len(train):,} interactions')
    print(f'validation: {len(valid):,} interactions')
    print(f'test: {len(test):,} interactions')
    train_ds = build_examples(train, train.iloc[:0], promos, lookup, rng)
    valid_ds = build_examples(valid, train, promos, lookup, rng)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    model = HistoryRecommender({k: len(v) for k, v in maps.items()}, categorical, numeric).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    best_loss, best_state = float('inf'), None
    print(f'\nDevice: {DEVICE}; training pairs: {len(train_ds):,}')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, weight_sum = 0., 0.
        for item, hist, weights, label, sample_weight in train_loader:
            item, hist, weights = item.to(DEVICE), hist.to(DEVICE), weights.to(DEVICE)
            label, sample_weight = label.to(DEVICE), sample_weight.to(DEVICE)
            optimizer.zero_grad()
            losses = criterion(model(item, hist, weights), label)
            loss = (losses * sample_weight).sum() / sample_weight.sum()
            loss.backward()
            optimizer.step()
            total += (losses * sample_weight).sum().item()
            weight_sum += sample_weight.sum().item()
        val_loss = loss_on(model, valid_loader, criterion)
        print(f'Epoch {epoch:02d}/{EPOCHS}: train_loss={total / weight_sum:.4f} val_loss={val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = ranking_metrics(model, train, valid, test, promos, lookup)
    path = OUT / 'history_recommender.pt'
    torch.save({'state_dict': best_state, 'architecture': 'HistoryRecommender',
                'embedding_sizes': {k: len(v) for k, v in maps.items()},
                'max_history': MAX_HISTORY, 'best_validation_loss': best_loss}, path)
    metrics_path = OUT / 'test_metrics_history.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('\nTEST RANKING METRICS')
    for key, value in metrics.items():
        print(f'{key}: {value}')
    print('\nSaved model:', path)
    print('Saved metrics:', metrics_path)
    print('Synthetic-data results are not real-world performance estimates.')


if __name__ == '__main__':
    main()
