"""
Validate and join synthetic retail data; create chronological ML splits.
Run: python prepare_training_data.py
No synthetic_user_preferences.csv is read or used.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'prepared_data'
SPLIT_RATIOS = (0.70, 0.15, 0.15)
EVENT_WEIGHTS = {'view': 0.0, 'click': 1.0, 'save': 2.0, 'purchase': 3.0}


def load_and_check(filename, key, dates=()):
    df = pd.read_csv(BASE_DIR / filename, parse_dates=list(dates))
    if df.empty or df[key].isna().any() or df[key].duplicated().any():
        raise ValueError(f'{filename}: {key} must be unique and non-null')
    return df


def prepare():
    users = load_and_check('users.csv', 'user_id', ['created_at'])
    stores = load_and_check('stores.csv', 'store_id')
    promos = load_and_check('promotions_with_categories.csv', 'promotion_id', ['start_date', 'end_date'])
    events = load_and_check('interactions.csv', 'interaction_id', ['interaction_timestamp'])

    # Prevent duplicated store category columns; stores.csv is authoritative.
    promos = promos.drop(columns=['store_category'], errors='ignore')
    if not promos['store_id'].isin(stores['store_id']).all():
        raise ValueError('Promotions contain unknown store_id values')
    if not events['user_id'].isin(users['user_id']).all():
        raise ValueError('Interactions contain unknown user_id values')
    if not events['promotion_id'].isin(promos['promotion_id']).all():
        raise ValueError('Interactions contain unknown promotion_id values')

    if (promos['end_date'] < promos['start_date']).any():
        raise ValueError('A promotion ends before it starts')
    if not events['interaction_type'].isin(EVENT_WEIGHTS).all():
        raise ValueError('Unknown interaction_type in interactions.csv')

    # Every interaction remains one row; validate all joins explicitly.
    merged = (events
              .merge(users[['user_id', 'created_at']], on='user_id', how='left', validate='many_to_one')
              .merge(promos, on='promotion_id', how='left', validate='many_to_one')
              .merge(stores[['store_id', 'store_name', 'category']], on='store_id', how='left', validate='many_to_one')
              .rename(columns={'category': 'store_category'}))
    if len(merged) != len(events):
        raise ValueError('Join changed the interaction row count')

    required = ['user_id', 'promotion_id', 'store_id', 'interaction_timestamp', 'created_at',
                'start_date', 'end_date', 'store_category', 'promotion_category',
                'promotion_type', 'discount_unit', 'discount_value']
    if merged[required].isna().any().any():
        raise ValueError('Missing required values:\n' + str(merged[required].isna().sum()[lambda x: x > 0]))

    t = merged['interaction_timestamp']
    valid_dates = ((t >= merged['created_at']) &
                   (t >= merged['start_date']) &
                   (t < merged['end_date'] + pd.Timedelta(days=1)))
    if not valid_dates.all():
        raise ValueError(f'{(~valid_dates).sum()} interactions outside account/promotion validity dates')
    if not np.isfinite(pd.to_numeric(merged['discount_value'], errors='coerce')).all():
        raise ValueError('Invalid discount_value found')

    # Target is constructed from the event; event type/target must NOT be input features.
    merged['engagement_target'] = merged['interaction_type'].map(EVENT_WEIGHTS).astype('float32')
    merged = merged.sort_values(['interaction_timestamp', 'interaction_id']).reset_index(drop=True)

    n = len(merged)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    splits = {
        'train': merged.iloc[:train_end].copy(),
        'validation': merged.iloc[train_end:val_end].copy(),
        'test': merged.iloc[val_end:].copy(),
    }

    # ID and catalog category vocabularies are identifiers, not learned target statistics.
    # UNK index 0 supports any unseen categorical values during inference.
    vocab_columns = ['user_id', 'promotion_id', 'store_id', 'store_category',
                     'promotion_category', 'promotion_type', 'discount_unit']
    vocab = {}
    for col in vocab_columns:
        values = sorted(merged[col].astype(str).unique().tolist())
        vocab[col] = {'<UNK>': 0, **{v: i + 1 for i, v in enumerate(values)}}
        for split in splits.values():
            split[col + '_idx'] = split[col].astype(str).map(vocab[col]).fillna(0).astype('int64')

    # Numeric normalization fitted on TRAIN only, then applied unchanged to other splits.
    mean = float(pd.to_numeric(splits['train']['discount_value']).mean())
    std = float(pd.to_numeric(splits['train']['discount_value']).std(ddof=0))
    std = std if std > 0 else 1.0
    for split in splits.values():
        split['discount_value_scaled'] = ((split['discount_value'].astype(float) - mean) / std).astype('float32')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / 'model_dataset.csv', index=False)
    for name, split in splits.items():
        split.to_csv(OUT_DIR / f'{name}.csv', index=False)

    metadata = {
        'categorical_vocabularies': vocab,
        'discount_value_mean_train': mean,
        'discount_value_std_train': std,
        'engagement_target_weights': EVENT_WEIGHTS,
        'feature_columns': [col + '_idx' for col in vocab_columns] + ['discount_value_scaled'],
        'target_column': 'engagement_target',
        'split': 'global chronological 70/15/15',
        'caution': 'Synthetic event-only records; no non-engagement impressions. Targets are observed event strength, not calibrated click probabilities.'
    }
    (OUT_DIR / 'feature_mappings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Validated {n:,} interactions from {merged.user_id.nunique()} users, '
          f'{merged.promotion_id.nunique()} promotions, {merged.store_id.nunique()} stores.')
    for name, split in splits.items():
        print(f'{name}: {len(split):,} rows | {split.interaction_timestamp.min()} to {split.interaction_timestamp.max()}')
    print(f'Saved model_dataset.csv, train.csv, validation.csv, test.csv, '
          f'feature_mappings.json in: {OUT_DIR}')


if __name__ == '__main__':
    prepare()
