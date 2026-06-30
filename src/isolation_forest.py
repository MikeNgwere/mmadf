# ============================================================
# MMADF — Isolation Forest (Layer 3)
# Point anomaly detection on customs declaration features
# Reference: Liu, Ting and Zhou (2012) ACM TKDD 6(1)
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import psycopg2
import psycopg2.extras

from src.config import (DB_CONFIG, FEATURES, IF_CONTAMINATION,
                         IF_MODEL_PATH, IF_SCALER_PATH)
from src.feature_extractor import get_training_features, extract_features

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# ── Training ───────────────────────────────────────────────────
def train(force=False):
    """
    Train Isolation Forest on normal declaration features.
    Saves model and scaler to disk.
    """
    if os.path.exists(IF_MODEL_PATH) and not force:
        print("IF model already exists — loading...")
        return load()

    print("Training Isolation Forest...")
    df       = get_training_features()
    X_train  = df[FEATURES].values
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  IF_MODEL_PATH)
    joblib.dump(scaler, IF_SCALER_PATH)

    print(f"✓ Isolation Forest trained on {len(X_train):,} records")
    print(f"✓ Model saved → {IF_MODEL_PATH}")
    return model, scaler

def load():
    """Load persisted model and scaler."""
    if not os.path.exists(IF_MODEL_PATH):
        return train()
    model  = joblib.load(IF_MODEL_PATH)
    scaler = joblib.load(IF_SCALER_PATH)
    print("✓ Isolation Forest loaded from disk")
    return model, scaler

def normalise_score(raw):
    """
    Convert raw IF score (more negative = more anomalous)
    to [0,1] where 1 = most anomalous.
    """
    return float(np.clip((raw * -1 + 0.5), 0, 1))

# ── Scoring ────────────────────────────────────────────────────
def score_batch(df, model=None, scaler=None):
    """
    Score a DataFrame of declarations.
    Returns DataFrame with if_score column added.
    """
    if model is None or scaler is None:
        model, scaler = load()

    X        = df[FEATURES].values
    X_scaled = scaler.transform(X)
    raw      = model.score_samples(X_scaled)
    df       = df.copy()
    df["if_score"] = [normalise_score(r) for r in raw]
    return df

def score_and_save(batch_size=500):
    """
    Score unscored declarations and write IF scores
    to PostgreSQL anomaly_scores table.
    Returns scored DataFrame.
    """
    model, scaler = load()
    df = extract_features(limit=batch_size, unscored_only=True)

    if df.empty:
        print("No unscored declarations found.")
        return pd.DataFrame()

    print(f"Scoring {len(df):,} declarations with Isolation Forest...")
    df = score_batch(df, model, scaler)

    # Write scores to PostgreSQL
    conn = get_conn()
    cur  = conn.cursor()

    rows = [(
        int(row["dec_id"]),
        float(row["if_score"]),
        None,               # lstm_score — filled later
        float(row["if_score"]),  # ensemble_score (IF only for now)
        {f: float(row[f]) for f in FEATURES}
    ) for _, row in df.iterrows()]

    psycopg2.extras.execute_values(cur, """
        INSERT INTO anomaly_scores
            (dec_id, if_score, lstm_score,
             ensemble_score, feature_vector)
        VALUES %s
        ON CONFLICT (dec_id) DO UPDATE
            SET if_score      = EXCLUDED.if_score,
                feature_vector= EXCLUDED.feature_vector
    """, [(r[0], r[1], r[2], r[3],
           psycopg2.extras.Json(r[4])) for r in rows],
    page_size=500)

    conn.commit()
    conn.close()

    avg  = df["if_score"].mean()
    high = (df["if_score"] >= 0.65).sum()
    print(f"✓ {len(df):,} declarations scored")
    print(f"  Avg IF score:  {avg:.4f}")
    print(f"  High anomaly (≥0.65): {high}")
    return df

# ── Evaluation ─────────────────────────────────────────────────
def evaluate():
    """
    Evaluate IF model on full dataset with known fraud labels.
    Prints precision, recall, F1 at the 0.65 threshold.
    """
    model, scaler = load()
    df  = extract_features()
    df  = score_batch(df, model, scaler)

    y_true = df["is_fraud_confirmed"].fillna(False).astype(int)
    y_pred = (df["if_score"] >= 0.65).astype(int)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = ((y_pred==1) & (y_true==0)).sum() / (y_true==0).sum()

    print("\n" + "="*45)
    print("ISOLATION FOREST EVALUATION")
    print("="*45)
    print(f"  Precision:          {p:.4f}")
    print(f"  Recall:             {r:.4f}")
    print(f"  F1-Score:           {f1:.4f}")
    print(f"  False Positive Rate:{fpr:.4f}")
    print("="*45)
    return {"precision":p, "recall":r, "f1":f1, "fpr":fpr}

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("Isolation Forest — Train, Score, Evaluate")
    print("="*55)

    # 1. Train
    train(force=True)

    # 2. Score all declarations
    scored = score_and_save(batch_size=10000)

    # 3. Evaluate
    results = evaluate()

    # 4. Show top anomalies
    if not scored.empty:
        top = scored.nlargest(5, "if_score")[
            ["dec_id","if_score","value_deviation_ratio",
             "submission_hour","is_fraud_confirmed"]
        ]
        print(f"\nTop 5 Anomalies Detected:")
        print(top.to_string(index=False))