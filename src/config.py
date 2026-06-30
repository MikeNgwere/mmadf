# ============================================================
# MMADF Configuration — FIXED with absolute paths
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import os

# ── Get absolute path to project root ─────────────────────────
# This works regardless of which directory Python is called from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Database ───────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "database": "mmadf_db",
    "user":     "postgres",
    "password": "postgres123",
    "port":     5432
}

# ── Model settings ─────────────────────────────────────────────
IF_CONTAMINATION    = 0.05
LSTM_SEQUENCE_LEN   = 15
LSTM_FEATURES       = 7
BASELINE_DAYS       = 180
ENSEMBLE_LEVEL1     = 0.65
ENSEMBLE_LEVEL2     = 0.85

# ── Feature names ──────────────────────────────────────────────
FEATURES = [
    "value_deviation_ratio",
    "hs_risk_category",
    "declarant_risk_tier",
    "submission_hour",
    "freq_30d",
    "freq_7d",
    "days_since_last"
]

# ── Absolute model file paths ──────────────────────────────────
# Using absolute paths prevents "model not found" errors
# when Streamlit runs from a different working directory
MODEL_DIR           = os.path.join(PROJECT_ROOT, "models")
IF_MODEL_PATH       = os.path.join(MODEL_DIR, "isolation_forest.pkl")
IF_SCALER_PATH      = os.path.join(MODEL_DIR, "if_scaler.pkl")
LSTM_MODEL_PATH     = os.path.join(MODEL_DIR, "lstm_model.keras")
LSTM_THRESHOLD_PATH = os.path.join(MODEL_DIR, "lstm_threshold.txt")
LSTM_NORM_PATH      = os.path.join(MODEL_DIR, "lstm_normalisation.npy")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Debug: confirm paths on startup ────────────────────────────
if __name__ == "__main__":
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Models dir:    {MODEL_DIR}")
    print(f"IF model:      {IF_MODEL_PATH}")
    print(f"LSTM model:    {LSTM_MODEL_PATH}")
    print(f"IF exists:     {os.path.exists(IF_MODEL_PATH)}")
    print(f"LSTM exists:   {os.path.exists(LSTM_MODEL_PATH)}")