# ============================================================
# MMADF — Temporal Anomaly Detection (No TensorFlow)
# Statistical sequence reconstruction — WSL2 compatible
# Saves as numpy arrays — no pickle module path issues
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import psycopg2
import warnings
warnings.filterwarnings('ignore')

from src.config import (DB_CONFIG, FEATURES,
                         LSTM_SEQUENCE_LEN, LSTM_FEATURES,
                         LSTM_THRESHOLD_PATH, LSTM_NORM_PATH,
                         MODEL_DIR)

SEQ_LEN    = LSTM_SEQUENCE_LEN   # 15
N_FEATURES = LSTM_FEATURES       # 7

# Save path — numpy format, no pickle class dependency
STAT_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.npy")


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ════════════════════════════════════════════════════════════
# STATISTICAL TEMPORAL MODEL
# Learns mean normal behaviour profile per feature.
# Anomaly score = MSE between new sequence and mean profile.
# Equivalent to a linear autoencoder on temporal sequences.
# ════════════════════════════════════════════════════════════

class StatTemporalModel:

    def __init__(self, seq_len=SEQ_LEN, n_features=N_FEATURES):
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.mean_profile = None
        self.std_profile  = None
        self.threshold    = None

    def fit(self, X_windows):
        """
        X_windows: array (n_windows, seq_len, n_features)
        Computes mean reconstruction profile from training windows.
        """
        self.mean_profile = np.mean(X_windows, axis=0)
        self.std_profile  = np.std(X_windows,  axis=0) + 1e-8
        errors            = self._reconstruction_errors(X_windows)
        self.threshold    = float(np.percentile(errors, 95))
        return self

    def _reconstruction_errors(self, X_windows):
        """MSE between each window and the mean profile."""
        diff   = X_windows - self.mean_profile
        errors = np.mean(np.square(diff), axis=(1, 2))
        return errors

    def score(self, X_window):
        """
        X_window: (seq_len, n_features)
        Returns (normalised_score [0-1], raw_mse).
        Higher score = more anomalous temporal pattern.
        """
        mse   = float(np.mean(np.square(
            X_window - self.mean_profile
        )))
        score = float(np.clip(
            mse / (self.threshold * 2.0 + 1e-9), 0, 1
        ))
        return score, mse

    def save(self, path):
        """Save model parameters as numpy dict — no pickle."""
        np.save(path, {
            'mean_profile': self.mean_profile,
            'std_profile':  self.std_profile,
            'threshold':    self.threshold,
            'seq_len':      self.seq_len,
            'n_features':   self.n_features
        }, allow_pickle=True)

    @classmethod
    def from_file(cls, path):
        """Load model from saved numpy dict."""
        data  = np.load(path, allow_pickle=True).item()
        model = cls(data['seq_len'], data['n_features'])
        model.mean_profile = data['mean_profile']
        model.std_profile  = data['std_profile']
        model.threshold    = data['threshold']
        return model


# ════════════════════════════════════════════════════════════
# NORMALISATION
# ════════════════════════════════════════════════════════════

def fit_normaliser(X_3d):
    """Fit min-max normaliser on 3D array."""
    flat = X_3d.reshape(-1, N_FEATURES)
    mn   = flat.min(axis=0)
    mx   = flat.max(axis=0)
    rng  = np.where((mx - mn) == 0, 1.0, mx - mn)
    X_n  = (X_3d - mn) / rng
    return X_n.astype(np.float32), mn, rng


def normalise(X_3d, mn, rng):
    """Apply fitted normaliser."""
    return ((X_3d - mn) / rng).astype(np.float32)


def build_windows(arr, seq_len):
    """Build sliding windows from 2D array (n_steps, features)."""
    wins = []
    for i in range(len(arr) - seq_len + 1):
        wins.append(arr[i:i + seq_len])
    return np.array(wins, dtype=np.float32)


# ════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ════════════════════════════════════════════════════════════

FEATURE_QUERY = """
    SELECT
        COALESCE(d.declared_value_usd /
            NULLIF(b.median_value_usd, 0), 1.0) AS value_deviation_ratio,
        COALESCE(c.risk_category, 3)             AS hs_risk_category,
        COALESCE(dec.risk_tier, 3)               AS declarant_risk_tier,
        EXTRACT(HOUR FROM d.submission_time)     AS submission_hour,
        COUNT(*) OVER (
            PARTITION BY d.declarant_id
            ORDER BY d.submission_time
            RANGE BETWEEN INTERVAL '30 days' PRECEDING
                      AND CURRENT ROW
        )                                        AS freq_30d,
        COUNT(*) OVER (
            PARTITION BY d.declarant_id
            ORDER BY d.submission_time
            RANGE BETWEEN INTERVAL '7 days' PRECEDING
                      AND CURRENT ROW
        )                                        AS freq_7d,
        COALESCE(EXTRACT(EPOCH FROM (
            d.submission_time -
            LAG(d.submission_time) OVER (
                PARTITION BY d.declarant_id
                ORDER BY d.submission_time
            )
        )) / 86400.0, 30.0)                      AS days_since_last,
        d.submission_time
    FROM declarations d
    LEFT JOIN value_baselines b
           ON b.hs_code        = d.hs_code
          AND b.origin_country = d.origin_country
    LEFT JOIN commodities  c   ON c.hs_code       = d.hs_code
    LEFT JOIN declarants   dec ON dec.declarant_id = d.declarant_id
    WHERE d.declarant_id = {did}
    {extra_where}
    ORDER BY d.submission_time ASC
    {limit}
"""


def load_declarant_features(declarant_id,
                             normal_only=False,
                             limit=None):
    """Load feature history for one declarant."""
    extra = ("AND d.is_fraud_confirmed IS DISTINCT FROM TRUE"
             if normal_only else "")
    lim   = f"LIMIT {limit}" if limit else ""
    sql   = FEATURE_QUERY.format(
        did=declarant_id, extra_where=extra, limit=lim
    )
    conn = get_conn()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    return df


# ════════════════════════════════════════════════════════════
# TRAIN
# ════════════════════════════════════════════════════════════

def train(force=False):
    """
    Train the statistical temporal model on normal declarations.
    Saves model as numpy arrays to avoid pickle issues.
    """
    if os.path.exists(STAT_MODEL_PATH) and not force:
        print("Temporal model already exists — loading...")
        return load()

    print("Loading declarant sequences from PostgreSQL...")
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute(f"""
        SELECT declarant_id, COUNT(*) AS cnt
        FROM   declarations
        WHERE  is_fraud_confirmed IS DISTINCT FROM TRUE
        GROUP  BY declarant_id
        HAVING COUNT(*) >= {SEQ_LEN}
        ORDER  BY cnt DESC
        LIMIT  150
    """)
    declarants = cur.fetchall()
    conn.close()

    print(f"  {len(declarants)} declarants with "
          f">= {SEQ_LEN} normal records found")

    if len(declarants) == 0:
        print("⚠  Not enough sequence data for training.")
        print(f"   Need declarants with >= {SEQ_LEN} declarations.")
        return None, None

    # Build windows one declarant at a time
    all_windows = []
    processed   = 0

    for did, cnt in declarants:
        try:
            df = load_declarant_features(
                did, normal_only=True
            )
            if len(df) < SEQ_LEN:
                continue
            vals = df[FEATURES].fillna(0).values.astype(
                np.float32
            )
            wins = build_windows(vals, SEQ_LEN)
            all_windows.append(wins)
            processed += 1

            if processed % 25 == 0:
                total_w = sum(len(w) for w in all_windows)
                print(f"  {processed}/{len(declarants)} declarants "
                      f"| {total_w:,} windows so far...")

            # Cap total windows to prevent memory issues
            total_so_far = sum(len(w) for w in all_windows)
            if total_so_far >= 3000:
                print(f"  Window cap reached ({total_so_far}). "
                      f"Stopping early.")
                break

        except Exception as e:
            continue

    if not all_windows:
        print("⚠  Could not build any training windows.")
        return None, None

    X_raw = np.concatenate(all_windows, axis=0)
    print(f"✓ {len(X_raw):,} training windows "
          f"from {processed} declarants")

    # Normalise
    X_norm, mn, rng = fit_normaliser(X_raw)
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(LSTM_NORM_PATH, np.stack([mn, rng]))

    # Split train / validation
    split   = int(len(X_norm) * 0.8)
    X_train = X_norm[:split]
    X_val   = X_norm[split:]
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # Fit statistical model
    model = StatTemporalModel(SEQ_LEN, N_FEATURES)
    model.fit(X_train)

    # Validation metrics
    val_errors = model._reconstruction_errors(X_val)
    val_mse    = float(np.mean(val_errors))
    threshold  = model.threshold

    # Save as numpy dict — no pickle class dependency
    model.save(STAT_MODEL_PATH)
    open(LSTM_THRESHOLD_PATH, 'w').write(str(threshold))

    print(f"\n✓ Temporal model trained successfully")
    print(f"  Training windows:    {len(X_train):,}")
    print(f"  Validation MSE:      {val_mse:.6f}")
    print(f"  Anomaly threshold:   {threshold:.6f}")
    print(f"  Model saved →        {STAT_MODEL_PATH}")
    return model, threshold


# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════

def load():
    """Load temporal model from numpy file."""
    if not os.path.exists(STAT_MODEL_PATH):
        print("No temporal model found — training now...")
        return train(force=True)

    model     = StatTemporalModel.from_file(STAT_MODEL_PATH)
    threshold = model.threshold
    print(f"✓ Temporal model loaded  |  "
          f"threshold: {threshold:.6f}")
    return model, threshold


# ════════════════════════════════════════════════════════════
# SCORE ONE DECLARANT
# ════════════════════════════════════════════════════════════

def score_declarant(declarant_id, model=None, threshold=None):
    """
    Score a declarant's most recent SEQ_LEN declarations.
    Returns (lstm_score [0-1], raw_mse) or (None, None).
    """
    if model is None:
        if not os.path.exists(STAT_MODEL_PATH):
            return None, None
        model, threshold = load()

    if not os.path.exists(LSTM_NORM_PATH):
        return None, None

    norm    = np.load(LSTM_NORM_PATH)
    mn, rng = norm[0], norm[1]

    try:
        df = load_declarant_features(
            declarant_id,
            normal_only=False,
            limit=SEQ_LEN * 2
        )
    except Exception:
        return None, None

    if len(df) < SEQ_LEN:
        return None, None

    # Use most recent SEQ_LEN records in chronological order
    vals   = (df[FEATURES]
              .fillna(0)
              .tail(SEQ_LEN)
              .values
              .astype(np.float32))

    X_norm = normalise(
        vals.reshape(1, SEQ_LEN, N_FEATURES), mn, rng
    )[0]

    lstm_score, mse = model.score(X_norm)
    return lstm_score, mse


# ════════════════════════════════════════════════════════════
# UPDATE ALL LSTM SCORES IN POSTGRESQL
# ════════════════════════════════════════════════════════════

def score_all_and_update():
    """Score all declarants with IF scores but no LSTM score."""
    model, threshold = load()
    if model is None:
        print("No temporal model available.")
        return {}

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT DISTINCT d.declarant_id
        FROM   declarations   d
        JOIN   anomaly_scores s ON s.dec_id = d.dec_id
        WHERE  s.lstm_score IS NULL
        LIMIT  300
    """)
    dids    = [r[0] for r in cur.fetchall()]
    conn.close()

    print(f"Scoring temporal model for {len(dids)} declarants...")
    scores  = {}
    skipped = 0

    for did in dids:
        sc, _ = score_declarant(did, model, threshold)
        if sc is None:
            skipped += 1
            continue
        scores[did] = sc

    if scores:
        conn = get_conn()
        cur  = conn.cursor()
        for did, sc in scores.items():
            cur.execute("""
                UPDATE anomaly_scores s
                SET    lstm_score = %s
                FROM   declarations d
                WHERE  d.dec_id       = s.dec_id
                  AND  d.declarant_id = %s
            """, (sc, did))
        conn.commit()
        conn.close()

    print(f"✓ Temporal scores updated:  {len(scores)}")
    print(f"  Skipped (short history):  {skipped}")
    return scores


# ════════════════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════════════════

def evaluate():
    """Evaluate temporal model against known fraud labels."""
    model, threshold = load()
    if model is None:
        return {}

    conn = get_conn()

    # Normal declarants
    cur = conn.cursor()
    cur.execute(f"""
        SELECT declarant_id FROM declarations
        WHERE  is_fraud_confirmed IS DISTINCT FROM TRUE
        GROUP  BY declarant_id
        HAVING COUNT(*) >= {SEQ_LEN}
        LIMIT  100
    """)
    normal_dids = [r[0] for r in cur.fetchall()]

    # Fraud declarants
    cur.execute("""
        SELECT DISTINCT declarant_id FROM declarations
        WHERE  is_fraud_confirmed = TRUE
        LIMIT  50
    """)
    fraud_dids = [r[0] for r in cur.fetchall()]
    conn.close()

    scores, labels = [], []

    for did in normal_dids:
        sc, _ = score_declarant(did, model, threshold)
        if sc is not None:
            scores.append(sc)
            labels.append(0)

    for did in fraud_dids:
        sc, _ = score_declarant(did, model, threshold)
        if sc is not None:
            scores.append(sc)
            labels.append(1)

    if not scores:
        print("No scores available for evaluation.")
        return {}

    from sklearn.metrics import (precision_score, recall_score,
                                  f1_score)
    y_true = np.array(labels)
    y_pred = (np.array(scores) >= 0.65).astype(int)

    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true,   y_pred, zero_division=0)
    f1  = f1_score(y_true,       y_pred, zero_division=0)
    fpr = (((y_pred == 1) & (y_true == 0)).sum()
           / max((y_true == 0).sum(), 1))

    print(f"\n{'='*45}")
    print("TEMPORAL MODEL EVALUATION")
    print(f"{'='*45}")
    print(f"  Precision:           {p:.4f}")
    print(f"  Recall:              {r:.4f}")
    print(f"  F1-Score:            {f1:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"{'='*45}")
    return {"precision": p, "recall": r, "f1": f1, "fpr": fpr}


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("MMADF Temporal Anomaly Detection — Layer 4")
    print("(Statistical Sequence Model — no TensorFlow)")
    print("=" * 55)

    # Clean up old broken pkl if it exists
    old_pkl = os.path.join(MODEL_DIR, "lstm_model.pkl")
    if os.path.exists(old_pkl):
        os.remove(old_pkl)
        print("✓ Removed old lstm_model.pkl")

    train(force=True)
    score_all_and_update()
    evaluate()
    print("\n✓ Temporal model complete")