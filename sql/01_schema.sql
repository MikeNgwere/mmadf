-- ============================================================
-- MMADF PostgreSQL Schema
-- MCS 504 Database Engineering | Mike T. Ngwere | R186209Q
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Reference tables first
CREATE TABLE IF NOT EXISTS commodities (
    hs_code          VARCHAR(10)   PRIMARY KEY,
    description      VARCHAR(255)  NOT NULL,
    risk_category    INT           NOT NULL CHECK (risk_category BETWEEN 1 AND 5),
    avg_value_per_kg NUMERIC(10,2) NOT NULL DEFAULT 0,
    last_updated     TIMESTAMP     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS declarants (
    declarant_id    SERIAL        PRIMARY KEY,
    reg_number      VARCHAR(20)   UNIQUE NOT NULL,
    full_name       VARCHAR(150)  NOT NULL,
    country_of_origin CHAR(3)    NOT NULL,
    risk_tier       INT           NOT NULL DEFAULT 1
                    CHECK (risk_tier BETWEEN 1 AND 5),
    registered_date DATE          DEFAULT CURRENT_DATE,
    is_active       BOOLEAN       DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS officers (
    officer_id   SERIAL       PRIMARY KEY,
    badge_number VARCHAR(20)  UNIQUE NOT NULL,
    full_name    VARCHAR(150) NOT NULL,
    role         VARCHAR(50)  NOT NULL,
    border_post  VARCHAR(50)  NOT NULL DEFAULT 'FORBES',
    is_active    BOOLEAN      DEFAULT TRUE
);

-- Core declarations table
CREATE TABLE IF NOT EXISTS declarations (
    dec_id              SERIAL        PRIMARY KEY,
    declarant_id        INT           NOT NULL REFERENCES declarants(declarant_id),
    hs_code             VARCHAR(10)   NOT NULL REFERENCES commodities(hs_code),
    declared_value_usd  NUMERIC(14,2) NOT NULL CHECK (declared_value_usd >= 0),
    declared_weight_kg  NUMERIC(12,3),
    origin_country      CHAR(3)       NOT NULL,
    submission_time     TIMESTAMP     NOT NULL DEFAULT NOW(),
    officer_id          INT           REFERENCES officers(officer_id),
    border_post         VARCHAR(50)   NOT NULL DEFAULT 'FORBES',
    status              VARCHAR(20)   NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','assessed','cleared','flagged','seized')),
    is_fraud_confirmed  BOOLEAN       DEFAULT NULL
);

-- ML output tables
CREATE TABLE IF NOT EXISTS anomaly_scores (
    score_id        SERIAL        PRIMARY KEY,
    dec_id          INT           NOT NULL UNIQUE REFERENCES declarations(dec_id),
    if_score        NUMERIC(8,6)  NOT NULL,
    lstm_score      NUMERIC(8,6),
    ensemble_score  NUMERIC(8,6)  NOT NULL,
    scored_at       TIMESTAMP     DEFAULT NOW(),
    model_version   VARCHAR(15)   DEFAULT 'v1.0',
    feature_vector  JSONB
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id              SERIAL       PRIMARY KEY,
    dec_id                INT          NOT NULL REFERENCES declarations(dec_id),
    score_id              INT          NOT NULL REFERENCES anomaly_scores(score_id),
    alert_level           INT          NOT NULL CHECK (alert_level IN (1,2)),
    fraud_type_predicted  VARCHAR(40),
    feature_contribution  JSONB,
    triggered_at          TIMESTAMP    DEFAULT NOW(),
    resolved              BOOLEAN      DEFAULT FALSE,
    was_true_positive     BOOLEAN      DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS declarant_sequences (
    seq_id       SERIAL    PRIMARY KEY,
    declarant_id INT       NOT NULL REFERENCES declarants(declarant_id),
    sequence_json JSONB    NOT NULL,
    window_size  INT       DEFAULT 15,
    computed_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id     SERIAL       PRIMARY KEY,
    table_name VARCHAR(50)  NOT NULL,
    operation  VARCHAR(10)  NOT NULL,
    changed_by VARCHAR(100) DEFAULT CURRENT_USER,
    changed_at TIMESTAMP    DEFAULT NOW(),
    details    JSONB
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_dec_declarant ON declarations(declarant_id);
CREATE INDEX IF NOT EXISTS idx_dec_submitted ON declarations(submission_time DESC);
CREATE INDEX IF NOT EXISTS idx_dec_hs_code   ON declarations(hs_code);
CREATE INDEX IF NOT EXISTS idx_dec_fraud     ON declarations(is_fraud_confirmed);
CREATE INDEX IF NOT EXISTS idx_scores_ens    ON anomaly_scores(ensemble_score DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_level  ON alerts(alert_level, resolved);

-- Materialised view for feature computation
CREATE MATERIALIZED VIEW IF NOT EXISTS value_baselines AS
SELECT
    hs_code,
    origin_country,
    PERCENTILE_CONT(0.5) WITHIN GROUP
        (ORDER BY declared_value_usd)      AS median_value_usd,
    AVG(declared_value_usd)                AS mean_value_usd,
    STDDEV(declared_value_usd)             AS stddev_value_usd,
    COUNT(*)                               AS sample_count
FROM declarations
WHERE is_fraud_confirmed IS DISTINCT FROM TRUE
GROUP BY hs_code, origin_country
HAVING COUNT(*) >= 3;

CREATE UNIQUE INDEX IF NOT EXISTS idx_baselines
    ON value_baselines(hs_code, origin_country);

SELECT 'Schema created successfully' AS status;