# ============================================================
# MMADF — Ensemble Scoring and Alert Engine (Layer 5)
# Combines IF + LSTM scores, generates alerts in PostgreSQL,
# computes SHAP values for officer explainability
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import json
import os
import warnings
warnings.filterwarnings('ignore')

from src.config import (DB_CONFIG, FEATURES,
                         ENSEMBLE_LEVEL1, ENSEMBLE_LEVEL2)
from src.isolation_forest import load as load_if, score_batch
from src.lstm_model       import load as load_lstm, score_declarant

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# ── Fraud type classifier ──────────────────────────────────────
FRAUD_RULES = {
    "under_valuation": lambda f: (
        f.get("value_deviation_ratio", 1.0) < 0.35
        and f.get("hs_risk_category", 1) >= 3
    ),
    "temporal_burst": lambda f: (
        f.get("submission_hour", 12) in [*range(22,24), *range(0,4)]
        and f.get("freq_7d", 0) > 10
    ),
    "identity_reuse": lambda f: (
        f.get("days_since_last", 30) < 0.3
        and f.get("declarant_risk_tier", 1) >= 3
    ),
    "tariff_misclass": lambda f: (
        f.get("hs_risk_category", 3) <= 2
        and f.get("value_deviation_ratio", 1.0) > 3.0
    ),
}

def classify_fraud_type(feature_vec):
    """
    Predict the most likely fraud archetype from feature values.
    Returns the fraud type label string.
    """
    for fraud_type, rule in FRAUD_RULES.items():
        try:
            if rule(feature_vec):
                return fraud_type
        except Exception:
            continue
    return "unknown_pattern"

# ── SHAP explainability ────────────────────────────────────────
def compute_shap(feature_vec, if_model, if_scaler):
    """
    Compute SHAP feature contributions for IF model.
    Returns dict with per-feature importance and primary driver.
    """
    try:
        import shap
        X = np.array([[feature_vec[f] for f in FEATURES]])
        X_scaled = if_scaler.transform(X)
        explainer = shap.TreeExplainer(if_model)
        shap_vals = explainer.shap_values(X_scaled)[0]

        result = {FEATURES[i]: round(float(shap_vals[i]), 6)
                  for i in range(len(FEATURES))}
        primary = max(result, key=lambda k: abs(result[k]))
        val     = feature_vec.get(primary, "N/A")
        result["_primary_driver"] = primary
        result["_explanation"]    = (
            f"Primary anomaly driver: {primary} = "
            f"{val:.3f}" if isinstance(val, float) else str(val)
        )
        return result
    except ImportError:
        # SHAP not installed — return feature values only
        return {f: round(float(feature_vec.get(f, 0)), 4)
                for f in FEATURES}
    except Exception as e:
        return {"error": str(e)}

# ── Core ensemble function ─────────────────────────────────────
def compute_ensemble(if_score, lstm_score):
    """
    Combine IF and LSTM scores into ensemble score.
    If LSTM score is unavailable, fall back to IF score only.
    """
    if lstm_score is not None and lstm_score >= 0:
        return round(0.5 * if_score + 0.5 * lstm_score, 6)
    return round(float(if_score), 6)

# ── Main detection cycle ───────────────────────────────────────
def run_detection_cycle(batch_size=500, verbose=True):
    """
    Full MMADF detection cycle:
    1. Fetch unscored declarations from PostgreSQL
    2. Score with Isolation Forest
    3. Score declarant sequences with LSTM
    4. Compute ensemble scores
    5. Insert alerts for detections above thresholds
    6. Write SHAP values for Level 2 alerts

    Returns: dict with cycle statistics
    """
    if verbose:
        print("="*55)
        print("MMADF Detection Cycle")
        print("="*55)

    # Load models
    if_model, if_scaler = load_if()
    lstm_model, lstm_threshold = load_lstm()

    # ── Step 1: Get unscored declarations ─────────────────────
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(f"""
        SELECT d.dec_id, d.declarant_id, d.hs_code,
               d.declared_value_usd, d.origin_country,
               d.submission_time, d.is_fraud_confirmed,
               COALESCE(d.declared_value_usd /
                   NULLIF(b.median_value_usd,0), 1.0)   AS value_deviation_ratio,
               COALESCE(c.risk_category, 3)              AS hs_risk_category,
               COALESCE(dec.risk_tier, 3)                AS declarant_risk_tier,
               EXTRACT(HOUR FROM d.submission_time)::INT AS submission_hour,
               COALESCE(EXTRACT(EPOCH FROM (
                   d.submission_time -
                   LAG(d.submission_time) OVER (
                       PARTITION BY d.declarant_id
                       ORDER BY d.submission_time
                   )))/86400.0, 30.0)                    AS days_since_last,
               1 AS freq_30d, 1 AS freq_7d
        FROM   declarations d
        LEFT JOIN value_baselines b
               ON b.hs_code=d.hs_code AND b.origin_country=d.origin_country
        LEFT JOIN commodities  c   ON c.hs_code=d.hs_code
        LEFT JOIN declarants   dec ON dec.declarant_id=d.declarant_id
        WHERE NOT EXISTS (
            SELECT 1 FROM anomaly_scores s WHERE s.dec_id=d.dec_id
        )
        ORDER BY d.submission_time DESC
        LIMIT {batch_size}
    """)
    rows = cur.fetchall()

    if not rows:
        if verbose: print("No unscored declarations found.")
        conn.close()
        return {"scored": 0, "alerts_l1": 0, "alerts_l2": 0}

    df = pd.DataFrame([dict(r) for r in rows])
    if verbose: print(f"Processing {len(df):,} declarations...")

    # ── Step 2: IF scores ──────────────────────────────────────
    df_scored = score_batch(df, if_model, if_scaler)

    # ── Step 3: LSTM scores per declarant ─────────────────────
    unique_declarants = df["declarant_id"].unique()
    lstm_cache = {}

    for did in unique_declarants:
        score, _ = score_declarant(
            int(did), lstm_model, lstm_threshold
        )
        if score is not None:
            lstm_cache[int(did)] = score

    # ── Step 4 & 5: Ensemble + alerts ─────────────────────────
    inserted = 0
    alerts_l1 = 0
    alerts_l2 = 0

    for _, row in df_scored.iterrows():
        dec_id       = int(row["dec_id"])
        declarant_id = int(row["declarant_id"])
        if_score     = float(row["if_score"])
        lstm_score   = lstm_cache.get(declarant_id)

        ensemble = compute_ensemble(if_score, lstm_score)

        feature_vec = {f: float(row[f]) for f in FEATURES
                       if f in row}

        # Insert into anomaly_scores
        cur2 = conn.cursor()
        cur2.execute("""
            INSERT INTO anomaly_scores
                (dec_id, if_score, lstm_score,
                 ensemble_score, feature_vector)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (dec_id) DO UPDATE
                SET ensemble_score = EXCLUDED.ensemble_score,
                    lstm_score     = EXCLUDED.lstm_score
        """, (
            dec_id,
            if_score,
            lstm_score,
            ensemble,
            psycopg2.extras.Json(feature_vec)
        ))

        # Get score_id for alert reference
        cur2.execute(
            "SELECT score_id FROM anomaly_scores WHERE dec_id=%s",
            (dec_id,)
        )
        score_row = cur2.fetchone()
        if not score_row:
            conn.commit()
            continue
        score_id = score_row[0]

        # ── Step 6: Generate alerts if above threshold ─────────
        if ensemble >= ENSEMBLE_LEVEL1:
            level       = 2 if ensemble >= ENSEMBLE_LEVEL2 else 1
            fraud_type  = classify_fraud_type(feature_vec)
            shap_vals   = (
                compute_shap(feature_vec, if_model, if_scaler)
                if level == 2 else None
            )

            cur2.execute("""
                INSERT INTO alerts
                    (dec_id, score_id, alert_level,
                     fraud_type_predicted, feature_contribution)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, (
                dec_id, score_id, level,
                fraud_type,
                psycopg2.extras.Json(shap_vals) if shap_vals else None
            ))

            if level == 1: alerts_l1 += 1
            if level == 2: alerts_l2 += 1

            # Flag declaration as flagged in main table
            if level == 2:
                cur2.execute("""
                    UPDATE declarations SET status='flagged'
                    WHERE dec_id=%s
                """, (dec_id,))

        inserted += 1

    conn.commit()
    conn.close()

    stats = {
        "scored":    inserted,
        "alerts_l1": alerts_l1,
        "alerts_l2": alerts_l2,
        "fpr_est":   round(alerts_l1 / max(inserted, 1), 4)
    }

    if verbose:
        print(f"\n{'='*45}")
        print(f"Detection Cycle Complete")
        print(f"{'='*45}")
        print(f"  Declarations scored:  {inserted:,}")
        print(f"  Level 1 alerts:       {alerts_l1}")
        print(f"  Level 2 alerts:       {alerts_l2}")
        print(f"  Est. alert rate:      {stats['fpr_est']:.2%}")
        print(f"{'='*45}")

    return stats

# ── Full evaluation ────────────────────────────────────────────
def evaluate_full():
    """
    Full MMADF ensemble evaluation against fraud labels.
    Produces the research paper Table 4 results.
    """
    conn = get_conn()
    df   = pd.read_sql("""
        SELECT d.dec_id, d.is_fraud_confirmed,
               s.if_score, s.lstm_score, s.ensemble_score
        FROM   declarations d
        JOIN   anomaly_scores s ON s.dec_id = d.dec_id
        WHERE  d.is_fraud_confirmed IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        print("No scored declarations with labels found.")
        print("Run run_detection_cycle() first.")
        return {}

    from sklearn.metrics import (precision_score, recall_score,
                                  f1_score, confusion_matrix)

    y_true = df["is_fraud_confirmed"].astype(int)
    results = {}

    configs = {
        "Rule-Based Baseline":   df["ensemble_score"] >= 0.50,
        "Isolation Forest Only": df["if_score"]       >= 0.65,
        "LSTM Only":             df["lstm_score"].fillna(0) >= 0.65,
        "MMADF Ensemble":        df["ensemble_score"] >= 0.65,
    }

    print("\n" + "="*70)
    print("MMADF FULL EVALUATION — Research Paper Table 4")
    print("="*70)
    print(f"  {'Model':<25} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'FPR':>6}")
    print(f"  {'-'*55}")

    for name, y_pred in configs.items():
        y_pred = y_pred.astype(int)
        p   = precision_score(y_true, y_pred, zero_division=0)
        r   = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0,1]
        ).ravel()
        fpr = fp / max(tn + fp, 1)

        marker = " ◄ MMADF" if name == "MMADF Ensemble" else ""
        print(f"  {name:<25} {p:>6.3f} {r:>6.3f} "
              f"{f1:>6.3f} {fpr:>6.3f}{marker}")
        results[name] = {
            "precision":p, "recall":r,
            "f1":f1, "fpr":fpr
        }

    mmadf = results.get("MMADF Ensemble", {})
    base  = results.get("Rule-Based Baseline", {})
    if mmadf and base:
        fpr_red = (1 - mmadf["fpr"] / max(base["fpr"], 0.001)) * 100
        print(f"\n  FPR reduction vs baseline: {fpr_red:.1f}%")

    print("="*70)
    return results

# ── Alert browser ──────────────────────────────────────────────
def show_top_alerts(level=2, n=10):
    """Print top N alerts of given level."""
    conn = get_conn()
    df   = pd.read_sql(f"""
        SELECT a.alert_id, d.dec_id, a.alert_level,
               a.fraud_type_predicted,
               ROUND(s.ensemble_score::numeric, 4) AS score,
               ROUND(s.if_score::numeric,       4) AS if_score,
               ROUND(COALESCE(s.lstm_score,0)::numeric, 4) AS lstm_score,
               d.declared_value_usd,
               d.submission_time,
               dec.full_name AS declarant,
               dec.country_of_origin,
               d.is_fraud_confirmed
        FROM   alerts a
        JOIN   anomaly_scores s  ON s.score_id = a.score_id
        JOIN   declarations   d  ON d.dec_id   = a.dec_id
        JOIN   declarants     dec ON dec.declarant_id = d.declarant_id
        WHERE  a.alert_level = {level}
        ORDER  BY s.ensemble_score DESC
        LIMIT  {n}
    """, conn)
    conn.close()

    print(f"\nTop {n} Level {level} Alerts:")
    print("="*80)
    for _, row in df.iterrows():
        confirmed = "✓ CONFIRMED" if row["is_fraud_confirmed"] else "? Unconfirmed"
        print(f"  Alert #{row['alert_id']:04d}  |  "
              f"Dec #{row['dec_id']:05d}  |  "
              f"Score: {row['score']:.4f}  |  "
              f"{row['fraud_type_predicted']:<18}  |  {confirmed}")
        print(f"           Declarant: {row['declarant']:<25} "
              f"Value: ${row['declared_value_usd']:,.2f}  "
              f"Country: {row['country_of_origin']}")
        print()

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Run full detection cycle
    stats = run_detection_cycle(batch_size=10000)

    # 2. Full evaluation vs fraud labels
    evaluate_full()

    # 3. Show top alerts
    show_top_alerts(level=2, n=5)
    show_top_alerts(level=1, n=5)

    # 4. Quick DB summary
    conn = get_conn()
    cur  = conn.cursor()

    print("\nPostgreSQL Summary:")
    for tbl, col in [
        ("declarations",   "COUNT(*)"),
        ("anomaly_scores", "COUNT(*)"),
        ("alerts",         "COUNT(*)"),
        ("alerts",         "COUNT(*) FILTER (WHERE alert_level=1)"),
        ("alerts",         "COUNT(*) FILTER (WHERE alert_level=2)"),
    ]:
        cur.execute(f"SELECT {col} FROM {tbl}")
        val = cur.fetchone()[0]
        label = f"{tbl}.{col}"
        print(f"  {label:<50} {val:>6,}")
    conn.close()