# MMADF — Multi-Model Anomaly Detection Framework

**AI-Driven Database Anomaly Detection for Fraud Prevention: A Hybrid Machine Learning Framework for Relational Transaction Systems**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791)](https://www.postgresql.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

---

## 📋 Overview

MMADF is a database-resident hybrid anomaly detection framework that embeds AI-driven fraud detection directly within a relational database, eliminating the external ETL pipelines that introduce latency and discard temporal context in conventional fraud detection systems.

The framework combines **Isolation Forest** point anomaly detection with **temporal sequence reconstruction**, computing all fraud-detection features natively using PostgreSQL window functions and materialised views — no external data warehousing required.

> **Database Engineering Gap Addressed:** Relational transaction databases lack native AI-driven anomaly detection capability. Detection is currently performed externally, losing the temporal relational context that is demonstrably the strongest predictor of fraudulent behaviour. MMADF closes this gap.

---

## 🎓 Academic Context

| Field | Detail |
|---|---|
| **Author** | Mike Ngwere |
| **Student Number** | R186209Q |
| **Course** | MCS 504 — Database Engineering |
| **Institution** | University of Zimbabwe, Department of Computer Science |
| **Semester** | Semester I, 2026 |

---

## 📊 Key Results

| Metric | MMADF Ensemble | Rule-Based Baseline | Improvement |
|---|---|---|---|
| F1-Score | **0.883** (±0.011) | 0.675 | +30.8% |
| Precision | **0.891** | 0.711 | +25.3% |
| Recall | **0.876** | 0.643 | +36.2% |
| False Positive Rate | **5.3%** | 14.2% | −62.7% |

Evaluated against 7 algorithmic baselines (One-Class SVM, LOF, Isolation Forest, Temporal Model, Autoencoder, Random Forest, XGBoost) with 5-fold cross-validation and McNemar significance testing — **p < 0.001 against all comparators**.

---

## 🏗 System Architecture

MMADF implements a **five-layer detection pipeline** across a **four-tier deployment**:

### Detection Layers
1. **Data Ingestion & Schema** — 8-table normalised PostgreSQL schema
2. **Feature Engineering** — 7 features computed via SQL window functions
3. **Isolation Forest** — Point anomaly detection (`n_estimators=200`, `contamination=0.05`)
4. **Temporal Sequence Detection** — 15-step per-entity reconstruction model
5. **Ensemble + SHAP Alert Engine** — Combined scoring with explainability

### Deployment Tiers
```
┌─────────────────────────────────────────┐
│   Streamlit Dashboard (Tier 4)           │
│   Monitoring · Alerts · Research Tab     │
└──────────────────┬────────────────────────┘
                    │
┌──────────────────┴────────────────────────┐
│   Flask REST API (Tier 3)                │
│   8 endpoints · Authentication            │
└──────────────────┬────────────────────────┘
                    │
┌──────────────────┴────────────────────────┐
│   Python ML Pipeline (Tier 2)            │
│   scikit-learn · Isolation Forest         │
│   Temporal reconstruction · SHAP          │
└──────────────────┬────────────────────────┘
                    │
┌──────────────────┴────────────────────────┐
│   PostgreSQL 15 (Tier 1)                 │
│   8 tables · RLS · pgaudit · pgcrypto    │
│   Window functions · Materialised views   │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mmadf_prototype/
├── api/
│   └── app.py                 # Flask REST API (8 endpoints)
├── dashboard/
│   └── dashboard.py            # Streamlit monitoring dashboard
├── src/
│   ├── config.py                # DB configuration, thresholds
│   ├── data_generator.py        # Synthetic transaction generator
│   ├── feature_extractor.py     # SQL window-function features
│   ├── isolation_forest.py      # IF training/scoring
│   ├── lstm_model.py            # Temporal sequence model
│   └── ensemble.py              # Ensemble scoring + alert engine
├── sql/
│   └── 01_schema.sql            # Database schema (8 tables)
├── models/                      # Trained model artefacts (not tracked)
├── eval_pipeline.py             # Reproducible evaluation script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- pip / virtualenv

### Setup

```bash
# Clone repository
git clone https://github.com/MikeNgwere/mmadf.git
cd mmadf

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
sudo service postgresql start
createdb mmadf_db
psql -d mmadf_db -f sql/01_schema.sql
```

### Configure Database Connection

Edit `src/config.py` with your PostgreSQL credentials:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mmadf_db",
    "user": "postgres",
    "password": "your_password"
}
```

---

## 🚀 Running the System

The system requires **three components** running concurrently.

### Terminal 1 — Flask API
```bash
source venv/bin/activate
python3 api/app.py
```
Runs on `http://localhost:5001`

### Terminal 2 — Streamlit Dashboard
```bash
source venv/bin/activate
streamlit run dashboard/dashboard.py --server.port 8502 --server.address 0.0.0.0
```
Runs on `http://localhost:8502`

### Terminal 3 — Database / Data Generation
```bash
sudo service postgresql start

# First-time setup only — generate synthetic data and train models
python3 src/data_generator.py
python3 src/isolation_forest.py
python3 src/lstm_model.py
python3 src/ensemble.py
```

---

## 🔑 Default Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `Admin@ZIMRA2026!` |
| Officer | `officer1` | `Officer@ZIMRA2026!` |
| Viewer | `viewer1` | `Viewer@ZIMRA2026!` |

> ⚠️ Change these before any production or public deployment.

---

## 🧪 Reproducing Evaluation Results

```bash
source venv/bin/activate
pip install scipy -q
python3 eval_pipeline.py
```

This reproduces the 5-fold cross-validation results, McNemar significance tests, and benchmark comparisons reported in the accompanying research paper (Tables 1–2).

**Environment for reported results:**
- Python 3.10.12, Ubuntu 22.04 LTS (WSL2)
- scikit-learn 1.3.2, numpy 1.24.4
- `random_state=42` throughout

---

## 📐 Database Schema Highlights

| Table | Purpose |
|---|---|
| `declarations` | Core transaction records |
| `declarants` | Entity / customer registry |
| `commodities` | Product / HS code risk classification |
| `anomaly_scores` | IF, temporal, and ensemble scores |
| `alerts` | Level 1/2 alerts with SHAP explanations |
| `audit_log` | Tamper-evident audit trail (pgaudit) |
| `value_baselines` *(materialised view)* | 90-day rolling commodity value medians |

### Example Feature Engineering (SQL Window Function)

```sql
-- 30-day transaction frequency per entity, computed natively
COUNT(*) OVER (
    PARTITION BY declarant_id
    ORDER BY submission_time
    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
) AS freq_30d
```

---

## 🔍 Fraud Archetypes Detected

Derived from STRIDE threat modelling applied to relational transaction systems:

| Archetype | STRIDE Category | Primary Detector |
|---|---|---|
| Value Under-Declaration | Tampering | Isolation Forest |
| Temporal Burst Activity | Spoofing | Temporal Model |
| Entity Identity Reuse | Spoofing + Elevation of Privilege | Ensemble |
| Product Code Misclassification | Tampering | Isolation Forest |

---

## 📈 Model Performance Summary

| Algorithm | F1-Score | FPR | Type |
|---|---|---|---|
| Rule-Based Baseline | 0.675 | 14.2% | Unsupervised |
| One-Class SVM | 0.710 | 11.8% | Unsupervised |
| Local Outlier Factor | 0.737 | 10.3% | Unsupervised |
| Isolation Forest Only | 0.818 | 8.4% | Unsupervised |
| Temporal Model Only | 0.848 | 7.1% | Unsupervised |
| Autoencoder | 0.810 | 8.8% | Unsupervised |
| Random Forest † | 0.787 | 9.1% | Supervised |
| XGBoost † | 0.825 | 8.2% | Supervised |
| **MMADF Ensemble** | **0.883** | **5.3%** | **Unsupervised** |

† Supervised methods require confirmed fraud labels — included as upper-bound reference only. MMADF, operating without labels, outperforms both supervised comparators.

---

## 🌍 Deployment

For cloud deployment guidance (Streamlit Community Cloud, Render, Supabase), see [`DEPLOYMENT.md`](DEPLOYMENT.md).

**Recommended free-tier stack:**
- Frontend: Streamlit Community Cloud
- Backend API: Render
- Database: Supabase (PostgreSQL)

---

## 📚 Citation

```
Ngwere, M.T. (2026) AI-Driven Database Anomaly Detection for Fraud 
Prevention: A Hybrid Machine Learning Framework for Relational 
Transaction Systems. MCS 504 Database Engineering, University of 
Zimbabwe, Harare.
```

---

## ⚠️ Limitations

- Evaluation conducted entirely on **synthetic data** due to confidentiality restrictions on real fraud-labelled transaction datasets
- Single-domain parameterisation (customs declarations); cross-domain validation (banking, insurance) is future work
- No live operational deployment validation has been conducted

See Chapter 4 (Results and Discussion) and Chapter 5 (Recommendations) of the accompanying research paper for full discussion.

---

## 📄 License

This project was developed for academic purposes as part of MCS 504 — Database Engineering at the University of Zimbabwe. Not licensed for commercial use without permission.

---

## 👤 Contact

**Mike Ngwere**
R186209Q | University of Zimbabwe
Department of Computer Science
+263 782 568 399
