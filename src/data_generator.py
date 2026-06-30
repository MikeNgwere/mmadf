# ============================================================
# MMADF Data Generator — Synthetic ZIMRA Customs Declarations
# Generates 10,000 ZIMRA-parameterised records and inserts
# into PostgreSQL. Fixed: uses real officer IDs from DB.
# MCS 504 | Mike T. Ngwere | R186209Q
# ============================================================

import sys
sys.path.append('.')

import psycopg2
import psycopg2.extras
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from src.config import DB_CONFIG

fake = Faker()
rng  = np.random.default_rng(42)
random.seed(42)

# ── Reference data ─────────────────────────────────────────────
COMMODITIES = [
    ("8471.30", "Laptop computers",           3, 850.00),
    ("6110.20", "Cotton jerseys/pullovers",    2,  12.50),
    ("8703.23", "Motor vehicles <1500cc",      3, 8500.00),
    ("1001.11", "Durum wheat",                 1, 380.00),
    ("3004.90", "Medicaments mixed",           2,  95.00),
    ("8517.12", "Smartphones",                 4, 220.00),
    ("2204.21", "Wine in containers ≤2L",      2,  18.00),
    ("4011.10", "New pneumatic tyres",         2,  65.00),
    ("8528.72", "Television receivers",        3, 320.00),
    ("2709.00", "Crude petroleum oils",        4, 620.00),
]

COUNTRIES     = ["MOZ","ZAF","CHN","IND","GBR",
                  "ARE","DEU","USA","KEN","ZMB"]
BORDER_POSTS  = ["FORBES","BEITBRIDGE","CHIRUNDU","KARIBA"]

OFFICERS_DATA = [
    ("ZIM001", "Tendai Moyo",    "Assessment Officer",  "FORBES"),
    ("ZIM002", "Rudo Chigwanda", "Enforcement Officer", "FORBES"),
    ("ZIM003", "Joseph Dube",    "Senior Assessor",     "BEITBRIDGE"),
    ("ZIM004", "Farai Nyoni",    "Assessment Officer",  "CHIRUNDU"),
    ("ZIM005", "Chipo Mutasa",   "Enforcement Officer", "KARIBA"),
]

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# ── Insert reference data ──────────────────────────────────────
def seed_reference_data(conn):
    cur = conn.cursor()

    # Insert all 5 officers first
    for badge, name, role, post in OFFICERS_DATA:
        cur.execute("""
            INSERT INTO officers(badge_number, full_name, role, border_post)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (badge_number) DO NOTHING
        """, (badge, name, role, post))

    # Insert commodities
    for hs, desc, risk, avg in COMMODITIES:
        cur.execute("""
            INSERT INTO commodities
                (hs_code, description, risk_category, avg_value_per_kg)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (hs_code) DO NOTHING
        """, (hs, desc, risk, avg))

    conn.commit()

    # Fetch actual officer IDs from database
    cur.execute("SELECT officer_id FROM officers ORDER BY officer_id")
    officer_ids = [r[0] for r in cur.fetchall()]

    print(f"✓ Reference data seeded")
    print(f"  Officers in DB: {officer_ids}")
    return officer_ids

# ── Generate declarants ────────────────────────────────────────
def generate_declarants(conn, n=500):
    cur = conn.cursor()

    # Check how many already exist
    cur.execute("SELECT COUNT(*) FROM declarants")
    existing = cur.fetchone()[0]
    if existing >= n:
        cur.execute("SELECT declarant_id FROM declarants")
        ids = [r[0] for r in cur.fetchall()]
        print(f"✓ Using {len(ids)} existing declarants")
        return ids

    records = []
    for i in range(existing, n):
        records.append((
            f"REG{i+1:05d}",
            fake.company() if random.random() > 0.3 else fake.name(),
            random.choice(COUNTRIES),
            random.choices([1,2,3,4,5], weights=[40,30,15,10,5])[0]
        ))

    psycopg2.extras.execute_values(cur, """
        INSERT INTO declarants
            (reg_number, full_name, country_of_origin, risk_tier)
        VALUES %s
        ON CONFLICT (reg_number) DO NOTHING
    """, records)
    conn.commit()

    cur.execute("SELECT declarant_id FROM declarants")
    ids = [r[0] for r in cur.fetchall()]
    print(f"✓ {len(ids)} declarants ready")
    return ids

# ── Generate one normal declaration ───────────────────────────
def make_normal(declarant_id, hs_data, base_time, officer_ids):
    hs_code, _, _, avg_val = random.choice(hs_data)
    value  = avg_val * rng.uniform(0.4, 1.8)
    weight = value   / avg_val * rng.uniform(0.8, 1.2)
    hour   = int(np.clip(rng.normal(12, 3), 7, 19))
    submit = base_time + timedelta(
        days=random.randint(0, 89),
        hours=hour,
        minutes=random.randint(0, 59)
    )
    return (
        declarant_id,
        hs_code,
        round(float(value),  2),
        round(float(weight), 2),
        random.choice(COUNTRIES),
        submit,
        random.choice(officer_ids),   # ← uses real officer IDs from DB
        random.choice(BORDER_POSTS),
        "assessed",
        False   # is_fraud_confirmed
    )

# ── Inject fraud records ───────────────────────────────────────
def inject_fraud(records, n_fraud=800):
    """Seed 4 fraud archetypes into the declarations list."""
    fraud_idx = random.sample(range(len(records)), n_fraud)
    archetypes = {
        "under_val": 350,   # 43.75% — value suppressed
        "burst":     180,   # 22.50% — after-hours submission
        "identity":  170,   # 21.25% — short days since last
        "misclass":  100,   # 12.50% — low-risk HS, high value
    }
    idx = 0
    for atype, count in archetypes.items():
        for _ in range(count):
            if idx >= len(fraud_idx):
                break
            fi  = fraud_idx[idx]
            idx += 1
            rec = list(records[fi])

            if atype == "under_val":
                # Declare 10-28% of actual average value
                high_risk = [x for x in COMMODITIES if x[2] >= 3]
                hs, _, _, avg = random.choice(high_risk)
                rec[1] = hs
                rec[2] = round(float(avg * rng.uniform(0.10, 0.28)), 2)

            elif atype == "burst":
                # After-hours submission
                hour = random.choice([22, 23, 0, 1, 2, 3])
                rec[5] = rec[5].replace(hour=hour,
                                         minute=random.randint(0, 59))

            elif atype == "identity":
                # Very short interval since last declaration
                rec[5] = rec[5] - timedelta(
                    hours=random.randint(1, 6)
                )
                rec[4] = random.choice(
                    [c for c in COUNTRIES if c != rec[4]]
                )

            elif atype == "misclass":
                # Low-risk HS code but very high declared value
                low_risk = [x for x in COMMODITIES if x[2] <= 2]
                hs, _, _, _ = random.choice(low_risk)
                rec[1] = hs
                rec[2] = round(float(rng.uniform(5000, 15000)), 2)

            rec[9] = True   # mark as fraud
            records[fi] = tuple(rec)

    print(f"✓ {n_fraud} fraud records injected across 4 archetypes")
    return records

# ── Main ───────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("MMADF Data Generator — ZIMRA Parameterised Dataset")
    print("=" * 55)

    conn = get_conn()

    # Step 1: Seed reference data and get real officer IDs
    officer_ids = seed_reference_data(conn)

    # Step 2: Generate/load declarants
    declarant_ids = generate_declarants(conn, 500)

    # Step 3: Check if declarations already exist
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM declarations")
    existing_decl = cur.fetchone()[0]
    if existing_decl >= 10000:
        print(f"✓ Database already has {existing_decl:,} declarations")
        print("  Skipping generation. Delete and re-run to regenerate.")
        conn.close()
        return

    # Step 4: Generate 10,000 normal declarations
    base_time = datetime.now() - timedelta(days=90)
    print(f"Generating 10,000 declarations...")
    declarations = [
        make_normal(
            random.choice(declarant_ids),
            COMMODITIES,
            base_time,
            officer_ids
        )
        for _ in range(10000)
    ]

    # Step 5: Inject fraud
    declarations = inject_fraud(declarations, 800)

    # Step 6: Bulk insert into PostgreSQL
    print("Inserting into PostgreSQL...")
    psycopg2.extras.execute_values(cur, """
        INSERT INTO declarations (
            declarant_id,
            hs_code,
            declared_value_usd,
            declared_weight_kg,
            origin_country,
            submission_time,
            officer_id,
            border_post,
            status,
            is_fraud_confirmed
        )
        VALUES %s
    """, declarations, page_size=500)
    conn.commit()

    # Step 7: Refresh materialised view
    print("Refreshing value_baselines...")
    cur.execute("REFRESH MATERIALIZED VIEW value_baselines")
    conn.commit()

    # Step 8: Verify
    cur.execute("SELECT COUNT(*) FROM declarations")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM declarations "
        "WHERE is_fraud_confirmed = TRUE"
    )
    fraud = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM value_baselines")
    baselines = cur.fetchone()[0]

    conn.close()

    print(f"\n{'=' * 55}")
    print(f"✓ Total declarations:   {total:,}")
    print(f"✓ Fraud records:        {fraud:,}  "
          f"({fraud / total * 100:.1f}%)")
    print(f"✓ Normal records:       {total - fraud:,}")
    print(f"✓ Baseline pairs:       {baselines}")
    print(f"✓ Data generator DONE")
    print(f"{'=' * 55}")

if __name__ == "__main__":
    main()