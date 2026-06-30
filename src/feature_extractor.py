# ============================================================
# MMADF Feature Extractor — Fixed (no self-joins)
# Uses window functions directly without N² join explosion
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import sys
sys.path.append('.')

import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from src.config import DB_CONFIG, FEATURES

def get_engine():
    """SQLAlchemy engine — avoids pandas DBAPI2 warning."""
    c = DB_CONFIG
    url = (f"postgresql+psycopg2://{c['user']}:{c['password']}"
           f"@{c['host']}:{c['port']}/{c['database']}")
    return create_engine(url)

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# ── Main feature SQL — no self-joins, pure window functions ───
FEATURE_SQL = """
SELECT
    d.dec_id,
    d.declarant_id,
    d.hs_code,
    d.declared_value_usd,
    d.origin_country,
    d.submission_time,
    d.is_fraud_confirmed,

    -- FEATURE 1: value deviation from 90-day commodity median
    COALESCE(
        d.declared_value_usd / NULLIF(b.median_value_usd, 0),
        1.0
    )::FLOAT                                            AS value_deviation_ratio,

    -- FEATURE 2: HS code risk category (1-5)
    COALESCE(c.risk_category, 3)::FLOAT                 AS hs_risk_category,

    -- FEATURE 3: declarant risk tier
    COALESCE(dec.risk_tier, 3)::FLOAT                   AS declarant_risk_tier,

    -- FEATURE 4: submission hour (0-23)
    EXTRACT(HOUR FROM d.submission_time)::FLOAT         AS submission_hour,

    -- FEATURE 5: declarations in last 30 days (pure window — no join)
    COUNT(*) OVER (
        PARTITION BY d.declarant_id
        ORDER BY d.submission_time
        RANGE BETWEEN INTERVAL '30 days' PRECEDING
                  AND CURRENT ROW
    )::FLOAT                                            AS freq_30d,

    -- FEATURE 6: declarations in last 7 days (burst detection)
    COUNT(*) OVER (
        PARTITION BY d.declarant_id
        ORDER BY d.submission_time
        RANGE BETWEEN INTERVAL '7 days' PRECEDING
                  AND CURRENT ROW
    )::FLOAT                                            AS freq_7d,

    -- FEATURE 7: days since last declaration by same declarant
    COALESCE(
        EXTRACT(EPOCH FROM (
            d.submission_time -
            LAG(d.submission_time) OVER (
                PARTITION BY d.declarant_id
                ORDER BY d.submission_time
            )
        )) / 86400.0,
        30.0
    )::FLOAT                                            AS days_since_last

FROM declarations d
LEFT JOIN value_baselines b
       ON b.hs_code        = d.hs_code
      AND b.origin_country = d.origin_country
LEFT JOIN commodities  c   ON c.hs_code      = d.hs_code
LEFT JOIN declarants   dec ON dec.declarant_id = d.declarant_id
{where_clause}
ORDER BY d.submission_time
{limit_clause}
"""

FILL_DEFAULTS = {
    "value_deviation_ratio": 1.0,
    "hs_risk_category":      3.0,
    "declarant_risk_tier":   3.0,
    "submission_hour":      12.0,
    "freq_30d":              1.0,
    "freq_7d":               1.0,
    "days_since_last":      30.0
}

def extract_features(limit=None, unscored_only=False):
    """Extract feature vectors from PostgreSQL."""
    where = []
    if unscored_only:
        where.append(
            "NOT EXISTS (SELECT 1 FROM anomaly_scores s "
            "WHERE s.dec_id = d.dec_id)"
        )
    where_clause  = ("WHERE " + " AND ".join(where)) if where else ""
    limit_clause  = f"LIMIT {limit}" if limit else ""

    sql = FEATURE_SQL.format(
        where_clause=where_clause,
        limit_clause=limit_clause
    )
    engine = get_engine()
    df     = pd.read_sql(sql, engine)
    engine.dispose()
    return df[["dec_id","declarant_id","hs_code",
               "declared_value_usd","origin_country",
               "submission_time","is_fraud_confirmed"]
              + FEATURES
             ].fillna(FILL_DEFAULTS)

def get_training_features(max_records=10000):
    """
    Get features for NORMAL declarations only — for model training.
    Hard limit prevents memory issues.
    """
    sql = FEATURE_SQL.format(
        where_clause="WHERE d.is_fraud_confirmed IS DISTINCT FROM TRUE",
        limit_clause=f"LIMIT {max_records}"
    )
    engine = get_engine()
    df     = pd.read_sql(sql, engine)
    engine.dispose()
    df = df[FEATURES + ["declarant_id","submission_time",
                         "is_fraud_confirmed"]].fillna(FILL_DEFAULTS)
    print(f"✓ Training features loaded: {len(df):,} normal records")
    return df

def get_declarant_sequences(min_history=8):
    """
    Build per-declarant sequences for LSTM.
    Returns dict: {declarant_id: np.array(n_steps, n_features)}
    """
    df = get_training_features(max_records=10000)
    sequences = {}
    for did, grp in df.groupby("declarant_id"):
        if len(grp) < min_history:
            continue
        grp = grp.sort_values("submission_time")
        sequences[did] = grp[FEATURES].values.astype(np.float32)
    print(f"✓ Sequences built for {len(sequences)} declarants "
          f"(min {min_history} records each)")
    return sequences

# ── Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing feature extractor...")
    df = extract_features(limit=5)
    print(f"Sample ({len(df)} rows):")
    print(df[["dec_id"] + FEATURES].to_string(index=False))
    print("\nFeature stats:")
    print(df[FEATURES].describe().round(3))
    seqs = get_declarant_sequences()
    if seqs:
        sid = list(seqs.keys())[0]
        print(f"\nSample sequence shape: {seqs[sid].shape}")
    print("✓ Feature extractor OK")