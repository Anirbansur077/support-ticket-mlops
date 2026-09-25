import pandas as pd
import numpy as np

SEVERE_TAGS = {'Outage', 'Crash', 'Bug', 'Security', 'Disruption', 'Network', 'Server', 'Data Loss'}


def compute_escalation_risk(row):
    is_severe_type = row['type'] in ['Incident', 'Problem']
    is_high_priority = row['priority'] == 'high'
    tags = [str(row.get(f'tag_{i}', '')) for i in range(1, 9)]
    has_severe_tag = any(tag in SEVERE_TAGS for tag in tags)
    return int(is_severe_type and is_high_priority and has_severe_tag)


def main():
    df = pd.read_csv('data/raw/tickets.csv')

    df_en = df[df['language'] == 'en'].copy()
    df_en = df_en.dropna(subset=['body'])
    df_en = df_en[df_en['body'].str.strip() != '']
    df_en = df_en.drop_duplicates(subset=['subject', 'body'])

    df_en['escalated'] = df_en.apply(compute_escalation_risk, axis=1)

    assert df_en['body'].isnull().sum() == 0
    assert df_en['priority'].isnull().sum() == 0
    assert df_en['escalated'].isin([0, 1]).all()
    assert df_en.duplicated(subset=['subject', 'body']).sum() == 0

    np.random.seed(42)
    start_date = pd.Timestamp('2025-01-01')
    end_date = pd.Timestamp('2026-09-01')
    df_en['created_at'] = pd.to_datetime(
        np.random.randint(start_date.value // 10**9, end_date.value // 10**9, size=len(df_en)),
        unit='s'
    )
    df_en = df_en.sort_values('created_at').reset_index(drop=True)

    split_idx = int(len(df_en) * 0.8)
    train = df_en.iloc[:split_idx].copy()
    val = df_en.iloc[split_idx:].copy()

    train.to_parquet('data/processed/train.parquet')
    val.to_parquet('data/processed/val.parquet')

    print(f"Train: {train.shape}, Val: {val.shape}")
    print(f"Train escalated %: {train['escalated'].mean():.4f}")
    print(f"Val escalated %: {val['escalated'].mean():.4f}")


if __name__ == "__main__":
    main()