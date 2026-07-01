# ============================================================
# MMADF Flask REST API
# Exposes all detection data from PostgreSQL via REST endpoints
# Used by Streamlit dashboard and external consumers
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import sys
sys.path.append('.')

from flask import Flask, jsonify, request, abort
import psycopg2
import psycopg2.extras
from datetime import datetime
import json

from src.config import DB_CONFIG, ENSEMBLE_LEVEL1, ENSEMBLE_LEVEL2

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Database helper ────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def db_query(sql, params=None, one=False):
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params or ())
    result = cur.fetchone() if one else cur.fetchall()
    conn.close()
    if result is None:
        return {} if one else []
    if one:
        return dict(result)
    return [dict(r) for r in result]

def db_execute(sql, params=None):
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(sql, params or ())
        conn.commit()
        return cur.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# ── Health check ───────────────────────────────────────────────
@app.route("/health")
def health():
    try:
        result = db_query(
            "SELECT COUNT(*) AS n FROM declarations", one=True
        )
        return jsonify({
            "status":      "ok",
            "service":     "MMADF Flask API",
            "database":    "mmadf_db",
            "declarations": result.get("n", 0),
            "timestamp":   datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ════════════════════════════════════════════════════════════
# ENDPOINT 1: Dashboard KPI Statistics
# GET /api/stats
# ════════════════════════════════════════════════════════════
@app.route("/api/stats")
def stats():
    """
    Returns all KPI metrics for the Streamlit dashboard.
    Single call — no multiple round trips needed.
    """
    s = {}

    # Declaration counts
    s["total_declarations"] = db_query(
        "SELECT COUNT(*) AS n FROM declarations", one=True
    )["n"]

    s["flagged_declarations"] = db_query(
        "SELECT COUNT(*) AS n FROM declarations WHERE status='flagged'",
        one=True
    )["n"]

    # Score statistics
    score_stats = db_query("""
        SELECT
            ROUND(AVG(ensemble_score)::numeric,  4) AS avg_score,
            ROUND(MAX(ensemble_score)::numeric,  4) AS max_score,
            ROUND(MIN(ensemble_score)::numeric,  4) AS min_score,
            COUNT(*) AS total_scored
        FROM anomaly_scores
    """, one=True)
    s.update(score_stats)

    # Alert counts
    s["alerts_level1"] = db_query(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_level=1",
        one=True
    )["n"]

    s["alerts_level2"] = db_query(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_level=2",
        one=True
    )["n"]

    s["unresolved_alerts"] = db_query(
        "SELECT COUNT(*) AS n FROM alerts WHERE resolved=FALSE",
        one=True
    )["n"]

    s["resolved_alerts"] = db_query(
        "SELECT COUNT(*) AS n FROM alerts WHERE resolved=TRUE",
        one=True
    )["n"]

    # True positive rate
    tp_result = db_query("""
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN was_true_positive THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*),0)
            , 1) AS tp_rate
        FROM alerts
        WHERE resolved = TRUE
    """, one=True)
    s["true_positive_rate"] = tp_result.get("tp_rate")

    # Fraud type breakdown
    s["fraud_by_type"] = db_query("""
        SELECT fraud_type_predicted AS fraud_type,
               COUNT(*) AS count
        FROM   alerts
        GROUP  BY fraud_type_predicted
        ORDER  BY count DESC
    """)

    return jsonify(s)

# ════════════════════════════════════════════════════════════
# ENDPOINT 2: Alerts List
# GET /api/alerts?level=2&limit=25&resolved=false
# ════════════════════════════════════════════════════════════
@app.route("/api/alerts")
def get_alerts():
    """
    Returns paginated list of alerts with full declaration
    and declarant details.
    """
    level    = request.args.get("level",    2,     type=int)
    limit    = request.args.get("limit",    25,    type=int)
    resolved = request.args.get("resolved", "false").lower() == "true"

    rows = db_query("""
        SELECT
            a.alert_id,
            a.alert_level,
            a.fraud_type_predicted,
            a.triggered_at,
            a.resolved,
            a.was_true_positive,
            ROUND(s.ensemble_score::numeric, 4) AS ensemble_score,
            ROUND(s.if_score::numeric,       4) AS if_score,
            ROUND(COALESCE(s.lstm_score,0)::numeric, 4) AS lstm_score,
            d.dec_id,
            d.declared_value_usd,
            d.hs_code,
            d.origin_country,
            d.submission_time,
            d.border_post,
            d.status,
            d.is_fraud_confirmed,
            dec.full_name            AS declarant_name,
            dec.reg_number           AS declarant_reg,
            dec.country_of_origin    AS declarant_country,
            dec.risk_tier            AS declarant_risk_tier,
            a.feature_contribution
        FROM   alerts a
        JOIN   anomaly_scores s   ON s.score_id    = a.score_id
        JOIN   declarations   d   ON d.dec_id      = a.dec_id
        JOIN   declarants     dec ON dec.declarant_id = d.declarant_id
        WHERE  a.alert_level = %s
          AND  a.resolved    = %s
        ORDER  BY s.ensemble_score DESC
        LIMIT  %s
    """, (level, resolved, limit))

    return jsonify({
        "count":  len(rows),
        "level":  level,
        "alerts": rows
    })

# ════════════════════════════════════════════════════════════
# ENDPOINT 3: Single Alert Detail
# GET /api/alerts/<alert_id>
# ════════════════════════════════════════════════════════════
@app.route("/api/alerts/<int:alert_id>")
def get_alert_detail(alert_id):
    """Returns full detail for a single alert including SHAP."""
    row = db_query("""
        SELECT a.*, s.*, d.*,
               dec.full_name, dec.reg_number,
               dec.country_of_origin, dec.risk_tier
        FROM   alerts a
        JOIN   anomaly_scores s   ON s.score_id      = a.score_id
        JOIN   declarations   d   ON d.dec_id         = a.dec_id
        JOIN   declarants     dec ON dec.declarant_id = d.declarant_id
        WHERE  a.alert_id = %s
    """, (alert_id,), one=True)

    if not row:
        abort(404, description=f"Alert {alert_id} not found")
    return jsonify(row)

# ════════════════════════════════════════════════════════════
# ENDPOINT 4: Resolve Alert
# POST /api/alerts/<alert_id>/resolve
# Body: {"true_positive": true/false, "notes": "..."}
# ════════════════════════════════════════════════════════════
@app.route("/api/alerts/<int:alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    """Officer resolves an alert after investigation."""
    data    = request.get_json() or {}
    is_tp   = data.get("true_positive", False)
    notes   = data.get("notes", "")

    rows = db_execute("""
        UPDATE alerts
        SET    resolved          = TRUE,
               was_true_positive = %s
        WHERE  alert_id = %s
          AND  resolved  = FALSE
    """, (is_tp, alert_id))

    if rows == 0:
        abort(404, description=f"Alert {alert_id} not found or already resolved")

    # Log resolution to audit_log
    db_execute("""
        INSERT INTO audit_log(table_name, operation, details)
        VALUES ('alerts','RESOLVE', %s)
    """, (psycopg2.extras.Json({
        "alert_id":       alert_id,
        "true_positive":  is_tp,
        "notes":          notes,
        "resolved_at":    datetime.now().isoformat()
    }),))

    return jsonify({
        "status":        "resolved",
        "alert_id":      alert_id,
        "true_positive": is_tp
    })

# ════════════════════════════════════════════════════════════
# ENDPOINT 5: Score Distribution (for histogram chart)
# GET /api/scores/distribution?sample=2000
# ════════════════════════════════════════════════════════════
@app.route("/api/scores/distribution")
def score_distribution():
    """Returns score samples for histogram visualisation."""
    sample = request.args.get("sample", 2000, type=int)

    rows = db_query("""
        SELECT s.ensemble_score,
               s.if_score,
               COALESCE(s.lstm_score, 0)  AS lstm_score,
               CASE WHEN d.is_fraud_confirmed
                    THEN 'fraud' ELSE 'normal' END AS label
        FROM   anomaly_scores s
        JOIN   declarations   d ON d.dec_id = s.dec_id
        ORDER  BY RANDOM()
        LIMIT  %s
    """, (sample,))

    return jsonify({
        "count": len(rows),
        "data":  rows
    })

# ════════════════════════════════════════════════════════════
# ENDPOINT 6: Recent Declarations with Scores
# GET /api/declarations?limit=50&flagged_only=false
# ════════════════════════════════════════════════════════════
@app.route("/api/declarations")
def get_declarations():
    """Returns recent declarations with their anomaly scores."""
    limit       = request.args.get("limit",       50,    type=int)
    flagged     = request.args.get("flagged_only", "false").lower()
    flagged_sql = "AND d.status = 'flagged'" if flagged == "true" else ""

    rows = db_query(f"""
        SELECT
            d.dec_id,
            d.declarant_id,
            d.hs_code,
            d.declared_value_usd,
            d.origin_country,
            d.submission_time,
            d.border_post,
            d.status,
            d.is_fraud_confirmed,
            dec.full_name            AS declarant_name,
            dec.risk_tier,
            ROUND(COALESCE(s.ensemble_score,0)::numeric, 4) AS ensemble_score,
            ROUND(COALESCE(s.if_score,      0)::numeric, 4) AS if_score,
            c.risk_category          AS hs_risk_category,
            c.description            AS commodity_description
        FROM   declarations  d
        JOIN   declarants    dec ON dec.declarant_id = d.declarant_id
        LEFT JOIN anomaly_scores s ON s.dec_id       = d.dec_id
        LEFT JOIN commodities    c ON c.hs_code       = d.hs_code
        WHERE  1=1 {flagged_sql}
        ORDER  BY d.submission_time DESC
        LIMIT  %s
    """, (limit,))

    return jsonify({"count": len(rows), "declarations": rows})

# ════════════════════════════════════════════════════════════
# ENDPOINT 7: Model Performance Summary
# GET /api/performance
# ════════════════════════════════════════════════════════════
@app.route("/api/performance")
def performance():
    """Returns model performance metrics from PostgreSQL data."""
    from sklearn.metrics import (precision_score, recall_score,
                                  f1_score)

    rows = db_query("""
        SELECT d.is_fraud_confirmed,
               s.if_score,
               COALESCE(s.lstm_score, 0) AS lstm_score,
               s.ensemble_score
        FROM   declarations   d
        JOIN   anomaly_scores s ON s.dec_id = d.dec_id
        WHERE  d.is_fraud_confirmed IS NOT NULL
    """)

    if not rows:
        return jsonify({"error": "No labelled data available"})

    import numpy as np
    y_true = np.array([int(r["is_fraud_confirmed"]) for r in rows])
    configs = {
        "rule_based":      np.array([int(r["ensemble_score"] >= 0.50) for r in rows]),
        "isolation_forest":np.array([int(r["if_score"]       >= 0.65) for r in rows]),
        "lstm_only":       np.array([int(r["lstm_score"]      >= 0.65) for r in rows]),
        "mmadf_ensemble":  np.array([int(r["ensemble_score"] >= 0.65) for r in rows]),
    }

    results = {}
    for name, y_pred in configs.items():
        p   = precision_score(y_true, y_pred, zero_division=0)
        r   = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        fpr = ((y_pred==1) & (y_true==0)).sum() / max((y_true==0).sum(), 1)
        results[name] = {
            "precision": round(float(p),  4),
            "recall":    round(float(r),  4),
            "f1_score":  round(float(f1), 4),
            "fpr":       round(float(fpr),4)
        }

    return jsonify({
        "total_labelled": len(y_true),
        "fraud_count":    int(y_true.sum()),
        "models":         results
    })

# ════════════════════════════════════════════════════════════
# ENDPOINT 8: Declarant Risk Profile
# GET /api/declarants/<declarant_id>
# ════════════════════════════════════════════════════════════
@app.route("/api/declarants/<int:declarant_id>")
def declarant_profile(declarant_id):
    """Returns full risk profile for a declarant."""
    profile = db_query("""
        SELECT dec.*,
               COUNT(d.dec_id)                          AS total_declarations,
               SUM(d.declared_value_usd)                AS total_value_usd,
               AVG(d.declared_value_usd)                AS avg_value_usd,
               COUNT(a.alert_id)                        AS total_alerts,
               COUNT(a.alert_id) FILTER
                   (WHERE a.alert_level = 2)             AS level2_alerts,
               MAX(s.ensemble_score)                    AS max_anomaly_score,
               AVG(s.ensemble_score)                    AS avg_anomaly_score
        FROM   declarants     dec
        LEFT JOIN declarations   d   ON d.declarant_id  = dec.declarant_id
        LEFT JOIN anomaly_scores s   ON s.dec_id         = d.dec_id
        LEFT JOIN alerts         a   ON a.dec_id         = d.dec_id
        WHERE  dec.declarant_id = %s
        GROUP  BY dec.declarant_id
    """, (declarant_id,), one=True)

    if not profile:
        abort(404, description=f"Declarant {declarant_id} not found")

    # Recent declarations for this declarant
    recent = db_query("""
        SELECT d.dec_id, d.hs_code, d.declared_value_usd,
               d.origin_country, d.submission_time, d.status,
               ROUND(COALESCE(s.ensemble_score,0)::numeric,4) AS score
        FROM   declarations   d
        LEFT JOIN anomaly_scores s ON s.dec_id = d.dec_id
        WHERE  d.declarant_id = %s
        ORDER  BY d.submission_time DESC
        LIMIT  10
    """, (declarant_id,))

    return jsonify({
        "profile":              profile,
        "recent_declarations":  recent
    })

# ── Error handlers ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("MMADF Flask REST API")
    print("="*55)
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /api/stats")
    print("  GET  /api/alerts?level=2&limit=25")
    print("  GET  /api/alerts/<id>")
    print("  POST /api/alerts/<id>/resolve")
    print("  GET  /api/scores/distribution")
    print("  GET  /api/declarations")
    print("  GET  /api/performance")
    print("  GET  /api/declarants/<id>")
    print("="*55)
    print("Running on http://localhost:5001")
    print("="*55)
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
    