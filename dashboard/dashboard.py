# =============================================================
# MMADF Dashboard — Professional Academic Design
# Tab-style navigation | Journal-appropriate colours
# MCS 504 Database Engineering | Mike Ngwere | R186209Q
# =============================================================

import sys
sys.path.append('.')

import streamlit as st
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config import DB_CONFIG, FEATURES, ENSEMBLE_LEVEL1, ENSEMBLE_LEVEL2

# ── Professional Academic Colour Palette ───────────────────
# Clean, journal-appropriate — no bright or fancy colours
NAVY      = "#1A2E4A"   # dark navy   — headers, primary
BLUE      = "#2D6A9F"   # steel blue  — secondary, links
TEAL      = "#1A6B4A"   # muted green — active states
WHITE     = "#FFFFFF"
LIGHT     = "#F8F9FA"   # near-white  — card backgrounds
GREY      = "#E9ECEF"   # light grey  — borders
DGREY     = "#6C757D"   # muted grey  — secondary text
TEXT      = "#212529"   # near-black  — body text
RED       = "#B03A2E"   # muted red   — L2 alerts
RED_LT    = "#FDEDEC"
GREEN     = "#1E8449"   # muted green — success
GREEN_LT  = "#EAFAF1"
AMBER     = "#B7770D"   # muted amber — L1 alerts
AMBER_LT  = "#FEF9E7"

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="MMADF — AI Fraud Detection",
    page_icon="🛃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Hide sidebar ── */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{ display:none !important; }}

/* ── Remove Streamlit top padding ── */
.block-container {{
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}}

/* ── Body typography ── */
body, p, div, span, label {{
    color: {TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
}}

/* ── Headings — visible and professional ── */
h1 {{ font-size:1.6rem; font-weight:700; color:{NAVY}; margin-bottom:4px; }}
h2 {{ font-size:1.3rem; font-weight:700; color:{NAVY}; margin-bottom:4px; }}
h3 {{ font-size:1.1rem; font-weight:700; color:{NAVY}; margin-bottom:4px; }}
h4 {{ font-size:1.0rem; font-weight:600; color:{NAVY}; margin-bottom:4px; }}

/* ── Tab navigation ── */
/* Style ALL st.radio horizontal as tab bars */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 0 !important;
    border-bottom: 2px solid {GREY} !important;
    padding: 0 0 0 0 !important;
    background: {WHITE} !important;
    width: 100% !important;
    overflow-x: auto !important;
}}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 10px 18px !important;
    margin: 0 0 -2px 0 !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: {DGREY} !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: all 0.15s !important;
    flex: 1 !important;
    text-align: center !important;
}}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
    color: {NAVY} !important;
    background: {LIGHT} !important;
    border-bottom: 3px solid {BLUE} !important;
}}
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked),
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {{
    color: {NAVY} !important;
    border-bottom: 3px solid {TEAL} !important;
    background: {WHITE} !important;
    font-weight: 700 !important;
}}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
}}
div[data-testid="stRadio"] > label {{ display: none !important; }}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {WHITE};
    border-radius: 8px;
    padding: 14px 16px;
    border: 1px solid {GREY};
    border-left: 4px solid {TEAL};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
[data-testid="stMetricValue"] {{
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: {NAVY} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {DGREY} !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

/* ── Buttons ── */
.stButton > button {{
    background: {BLUE};
    color: {WHITE};
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: background 0.15s;
}}
.stButton > button:hover {{
    background: {NAVY};
    color: {WHITE};
}}
.stButton > button[kind="primary"] {{
    background: {TEAL} !important;
    color: {WHITE} !important;
}}
.stButton > button[kind="secondary"] {{
    background: {WHITE} !important;
    color: {BLUE} !important;
    border: 1px solid {BLUE} !important;
}}

/* ── Native Streamlit tabs (Research page sub-tabs) ── */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 2px solid {GREY};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 0.88rem;
    color: {DGREY};
    padding: 10px 18px;
    border-bottom: 3px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {NAVY} !important;
    border-bottom: 3px solid {TEAL} !important;
    font-weight: 700 !important;
}}

/* ── Dataframes ── */
.stDataFrame {{ border-radius: 6px; border: 1px solid {GREY}; }}

/* ── Divider ── */
hr {{ border: none; border-top: 1px solid {GREY}; margin: 14px 0; }}

/* ── Expanders ── */
.streamlit-expanderHeader {{
    background: {LIGHT} !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    border: 1px solid {GREY} !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
    color: {TEXT} !important;
}}
.stSelectbox > div > div {{
    border: 1px solid {GREY} !important;
    border-radius: 6px !important;
}}

/* ── Info/success/warning boxes ── */
.stAlert {{ border-radius: 6px; font-size: 0.9rem; }}

/* ── Alert components ── */
.alert-l2 {{
    background: {RED_LT};
    border-left: 5px solid {RED};
    padding: 14px 18px;
    border-radius: 6px;
    margin: 8px 0;
}}
.alert-l1 {{
    background: {AMBER_LT};
    border-left: 5px solid {AMBER};
    padding: 14px 18px;
    border-radius: 6px;
    margin: 8px 0;
}}
.alert-clear {{
    background: {GREEN_LT};
    border-left: 5px solid {GREEN};
    padding: 14px 18px;
    border-radius: 6px;
    margin: 8px 0;
}}

/* ── Page header banner ── */
.page-header {{
    background: linear-gradient(90deg, {NAVY} 0%, {BLUE} 100%);
    padding: 14px 20px;
    border-radius: 0 0 8px 8px;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

/* ── Section headers ── */
.section-header {{
    font-size: 1.0rem;
    font-weight: 700;
    color: {NAVY};
    padding: 6px 0 6px 12px;
    border-left: 4px solid {TEAL};
    margin: 12px 0 8px;
    background: {LIGHT};
    border-radius: 0 4px 4px 0;
}}

/* ── Footer ── */
.footer {{
    text-align: center;
    color: {DGREY};
    font-size: 0.78rem;
    padding: 14px 0 6px;
    border-top: 1px solid {GREY};
    margin-top: 24px;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════
def query(sql, params=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df   = pd.read_sql(sql, conn, params=params or ())
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()

def execute(sql, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
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

def execute_return(sql, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
    try:
        cur.execute(sql, params or ())
        conn.commit()
        return cur.fetchone()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
def render_header(name=None, role=None, post=None):
    user_html = (
        f'<div style="font-size:0.85rem;color:rgba(255,255,255,0.9);'
        f'text-align:right;">'
        f'<b style="color:#AED6F1;">{name}</b>'
        f'&nbsp;·&nbsp;'
        f'<span style="background:rgba(255,255,255,0.15);padding:2px 8px;'
        f'border-radius:10px;font-size:0.78rem;">{role.upper()}</span>'
        f'&nbsp;·&nbsp;{post}'
        f'</div>'
    ) if name else (
        f'<span style="color:#AED6F1;font-size:0.88rem;">'
        f'University of Zimbabwe &nbsp;·&nbsp; MCS 504 Database Engineering'
        f'</span>'
    )
    st.markdown(f"""
    <div class="page-header">
        <div>
            <span style="color:white;font-size:1.25rem;font-weight:800;
                         letter-spacing:0.5px;">🛃&nbsp; MMADF</span>
            <span style="color:rgba(255,255,255,0.65);font-size:0.85rem;
                         margin-left:14px;">
                AI-Driven Database Anomaly Detection &nbsp;·&nbsp;
                Fraud Prevention Framework
            </span>
        </div>
        {user_html}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MAIN NAVIGATION — TAB STYLE
# ══════════════════════════════════════════════════════════
def main_nav(pages, current_key="page"):
    """
    Tab-style navigation using st.radio(horizontal=True).
    CSS above makes it look identical to st.tabs().
    """
    current = st.session_state.get(current_key, pages[0])
    if current not in pages:
        current = pages[0]

    choice = st.radio(
        "Navigation",
        pages,
        index=pages.index(current),
        horizontal=True,
        label_visibility="collapsed",
        key=current_key
    )
    # Thin gold line separator
    st.markdown(
        f"<div style='height:1px;background:{GREY};"
        f"margin:0 0 16px;'></div>",
        unsafe_allow_html=True
    )
    return choice

# ══════════════════════════════════════════════════════════
# SECTION HEADER HELPER
# ══════════════════════════════════════════════════════════
def sh(text):
    st.markdown(
        f"<div class='section-header'>{text}</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════
def login_page():
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)

    _, col, _ = st.columns([1.3, 2, 1.3])
    with col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{NAVY},{BLUE});
                    padding:30px 32px;border-radius:12px;
                    text-align:center;margin-bottom:24px;">
            <div style="font-size:2.5rem;margin-bottom:8px;">🛃</div>
            <h2 style="color:white;margin:0 0 6px;font-size:1.4rem;">
                MMADF System
            </h2>
            <p style="color:#AED6F1;margin:0 0 4px;font-weight:600;
                      font-size:0.95rem;">
                Multi-Model Anomaly Detection Framework
            </p>
            <p style="color:rgba(255,255,255,0.6);margin:0;font-size:0.82rem;">
                MCS 504 Database Engineering &nbsp;·&nbsp;
                University of Zimbabwe &nbsp;·&nbsp; 2026
            </p>
        </div>
        """, unsafe_allow_html=True)

        tab_l, tab_r = st.tabs(["🔑  Login", "📝  Sign Up"])

        with tab_l:
            with st.form("login_form"):
                st.markdown("<br>", unsafe_allow_html=True)
                username = st.text_input(
                    "Username", placeholder="e.g. admin")
                password = st.text_input(
                    "Password", type="password",
                    placeholder="Admin@ZIMRA2026!")
                st.caption(
                    "Default credentials: **admin** / **Admin@ZIMRA2026!**")
                sub = st.form_submit_button(
                    "Login →", type="primary",
                    use_container_width=True)
            if sub:
                if not username or not password:
                    st.error("Enter username and password.")
                    return
                user = query("""
                    SELECT user_id,username,full_name,role,
                           border_post,badge_no
                    FROM users
                    WHERE username=%s AND password_hash=%s
                      AND is_active=TRUE
                """, (username, hash_pw(password)))
                if user.empty:
                    st.error("Invalid credentials.")
                else:
                    r = user.iloc[0]
                    st.session_state.update({
                        "logged_in":   True,
                        "user_id":     int(r["user_id"]),
                        "username":    r["username"],
                        "full_name":   r["full_name"],
                        "role":        r["role"],
                        "border_post": r["border_post"],
                        "badge_no":    r["badge_no"],
                        "page":        "📊  Dashboard",
                    })
                    execute(
                        "UPDATE users SET last_login=NOW() "
                        "WHERE user_id=%s",
                        (int(r["user_id"]),)
                    )
                    st.success(f"Welcome, {r['full_name']}!")
                    time.sleep(0.5)
                    st.rerun()

        with tab_r:
            with st.form("signup_form"):
                st.markdown("<br>", unsafe_allow_html=True)
                f1, f2 = st.columns(2)
                nm   = f1.text_input("Full Name *")
                un   = f2.text_input("Username *")
                ba   = f1.text_input("Badge Number *")
                po   = f2.selectbox("Border Post *",
                    ["FORBES","BEITBRIDGE","CHIRUNDU","KARIBA"])
                ro   = st.selectbox("Role *", ["viewer","officer"])
                p1   = st.text_input("Password *", type="password")
                p2   = st.text_input("Confirm *", type="password")
                reg  = st.form_submit_button(
                    "Register Account",
                    use_container_width=True)
            if reg:
                if not all([nm,un,ba,p1]):
                    st.error("All * fields required.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        execute("""
                            INSERT INTO users
                            (username,password_hash,full_name,
                             role,border_post,badge_no)
                            VALUES (%s,%s,%s,%s,%s,%s)
                        """, (un,hash_pw(p1),nm,ro,po,ba))
                        st.success(f"Account created for {nm}.")
                    except Exception as e:
                        st.error("Username exists."
                                 if "unique" in str(e).lower()
                                 else str(e))

# ══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════
def main_app():
    role  = st.session_state["role"]
    name  = st.session_state["full_name"]
    post  = st.session_state.get("border_post","SYSTEM")

    # ── Header ────────────────────────────────────────────
    render_header(name, role, post)

    # ── Build page list ───────────────────────────────────
    pages = [
        "📊  Dashboard",
        "🚨  Alerts",
        "📈  Performance",
        "🔍  Score Explorer",
        "📄  Research",
    ]
    if role in ["officer","admin"]:
        pages += [
            "📝  Submit",
            "👥  Declarants",
            "📦  Commodities",
            "⚙️  Detection",
        ]
    if role == "admin":
        pages += ["🏛  Admin"]

    # Logout row — right-aligned above nav
    lo1, lo2 = st.columns([10, 1])
    with lo2:
        if st.button("Logout", type="secondary",
                     use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Tab-style main navigation ─────────────────────────
    page = main_nav(pages)

    # ════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ════════════════════════════════════════════════════════
    if "Dashboard" in page:
        st.markdown(f"""
        <div style="background:{NAVY};color:white;padding:12px 18px;
                    border-radius:8px;margin-bottom:16px;
                    display:flex;justify-content:space-between;
                    align-items:center;">
            <div>
                <b style="font-size:1.05rem;color:white;">📊 Live Detection Dashboard</b>
                <span style="color:rgba(255,255,255,0.55);
                             font-size:0.82rem;margin-left:12px;">
                    Real-time monitoring
                </span>
            </div>
            <span style="color:rgba(255,255,255,0.5);font-size:0.78rem;">
                {datetime.now().strftime('%Y-%m-%d  %H:%M')}
            </span>
        </div>
        """, unsafe_allow_html=True)

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        total  = query("SELECT COUNT(*) n FROM declarations").iloc[0,0]
        scored = query("SELECT COUNT(*) n FROM anomaly_scores").iloc[0,0]
        l1     = query("SELECT COUNT(*) n FROM alerts WHERE alert_level=1").iloc[0,0]
        l2     = query("SELECT COUNT(*) n FROM alerts WHERE alert_level=2").iloc[0,0]
        flagged= query("SELECT COUNT(*) n FROM declarations WHERE status='flagged'").iloc[0,0]
        avg_s  = query("SELECT ROUND(AVG(ensemble_score)::numeric,3) n FROM anomaly_scores").iloc[0,0]

        k1.metric("Transactions",    f"{total:,}")
        k2.metric("Scored",          f"{scored:,}")
        k3.metric("L1 Alerts",       l1)
        k4.metric("L2 Alerts",       l2,
                  delta=f"threshold ≥{ENSEMBLE_LEVEL2}",
                  delta_color="inverse")
        k5.metric("Flagged",         flagged)
        k6.metric("Avg Score",       avg_s or "—")

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([3,2,2])

        with c1:
            sh("Anomaly Score Distribution")
            df_s = query("""
                SELECT s.ensemble_score,
                       CASE WHEN d.is_fraud_confirmed
                            THEN 'Fraud' ELSE 'Normal' END AS label
                FROM anomaly_scores s
                JOIN declarations d ON d.dec_id=s.dec_id
                ORDER BY RANDOM() LIMIT 3000
            """)
            if not df_s.empty:
                fig = px.histogram(
                    df_s, x="ensemble_score",
                    color="label", nbins=60, barmode="overlay",
                    color_discrete_map={"Normal":BLUE,"Fraud":RED},
                    opacity=0.72,
                    labels={"ensemble_score":"Score","count":"Records"}
                )
                fig.add_vline(x=ENSEMBLE_LEVEL1,line_dash="dash",
                              line_color=AMBER,line_width=2,
                              annotation_text="L1")
                fig.add_vline(x=ENSEMBLE_LEVEL2,line_dash="solid",
                              line_color=RED,line_width=2,
                              annotation_text="L2")
                fig.update_layout(
                    height=290,
                    margin=dict(t=10,b=30,l=20,r=20),
                    plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                    legend=dict(orientation="h",y=1.08,
                                font=dict(size=10)),
                    font=dict(color=TEXT)
                )
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            sh("Fraud Type Breakdown")
            df_t = query("""
                SELECT fraud_type_predicted AS type,
                       COUNT(*) AS count
                FROM alerts
                WHERE fraud_type_predicted IS NOT NULL
                GROUP BY fraud_type_predicted
                ORDER BY count DESC
            """)
            if not df_t.empty:
                fig2 = px.pie(
                    df_t, values="count", names="type",
                    hole=0.5,
                    color_discrete_sequence=[
                        NAVY,BLUE,TEAL,AMBER,DGREY
                    ]
                )
                fig2.update_layout(
                    height=290,
                    margin=dict(t=10,b=10,l=10,r=10),
                    legend=dict(
                        orientation="h",y=-0.2,
                        font=dict(size=9,color=TEXT)
                    ),
                    font=dict(color=TEXT)
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No alerts generated yet.")

        with c3:
            sh("Alert Level Summary")
            al = pd.DataFrame({
                "Level":["Unscored","Clear","L1 Review","L2 Enforce"],
                "Count":[
                    max(0,int(total)-int(scored)),
                    max(0,int(scored)-int(l1)-int(l2)),
                    int(l1), int(l2)
                ]
            })
            fig3 = px.bar(
                al, x="Level", y="Count",
                color="Level",
                color_discrete_map={
                    "Unscored": DGREY,
                    "Clear":    TEAL,
                    "L1 Review":AMBER,
                    "L2 Enforce":RED
                },
                text="Count"
            )
            fig3.update_layout(
                height=290,showlegend=False,
                margin=dict(t=10,b=30,l=20,r=20),
                plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                font=dict(color=TEXT)
            )
            fig3.update_traces(textposition="outside")
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        sh("Recent Level 2 Alerts")
        df_rec = query("""
            SELECT a.alert_id, d.dec_id,
                   a.fraud_type_predicted  AS fraud_type,
                   ROUND(s.ensemble_score::numeric,4) AS score,
                   d.declared_value_usd   AS value_usd,
                   d.hs_code, d.origin_country,
                   TO_CHAR(d.submission_time,
                       'YYYY-MM-DD HH24:MI') AS submitted,
                   dec.full_name          AS entity
            FROM alerts a
            JOIN anomaly_scores s ON s.score_id=a.score_id
            JOIN declarations d ON d.dec_id=a.dec_id
            JOIN declarants dec ON dec.declarant_id=d.declarant_id
            WHERE a.alert_level=2
            ORDER BY a.triggered_at DESC LIMIT 10
        """)
        if not df_rec.empty:
            st.dataframe(df_rec, use_container_width=True,
                         hide_index=True)
        else:
            st.info("No Level 2 alerts yet.")

    # ════════════════════════════════════════════════════════
    # PAGE: ALERTS
    # ════════════════════════════════════════════════════════
    elif "Alerts" in page:
        st.markdown(f"<h2>🚨 Fraud Alerts</h2>", unsafe_allow_html=True)
        tab_l2, tab_l1 = st.tabs([
            "🔴  Level 2 — Enforcement",
            "🟡  Level 1 — Review Queue"
        ])
        for tab, lvl, col_h in [(tab_l2,2,RED),(tab_l1,1,AMBER)]:
            with tab:
                df_a = query(f"""
                    SELECT a.alert_id, d.dec_id,
                           a.fraud_type_predicted AS fraud_type,
                           ROUND(s.ensemble_score::numeric,4) AS score,
                           d.declared_value_usd  AS value_usd,
                           d.hs_code, d.origin_country,
                           TO_CHAR(d.submission_time,
                               'YYYY-MM-DD HH24:MI') AS submitted,
                           dec.full_name          AS entity,
                           a.resolved,
                           d.is_fraud_confirmed   AS confirmed
                    FROM alerts a
                    JOIN anomaly_scores s ON s.score_id=a.score_id
                    JOIN declarations d ON d.dec_id=a.dec_id
                    JOIN declarants dec
                        ON dec.declarant_id=d.declarant_id
                    WHERE a.alert_level={lvl}
                    ORDER BY s.ensemble_score DESC LIMIT 200
                """)
                if df_a.empty:
                    st.info(f"No Level {lvl} alerts.")
                    continue

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total",       len(df_a))
                m2.metric("Unresolved",
                          int((~df_a["resolved"]).sum()))
                m3.metric("Confirmed",
                          int(df_a["confirmed"].sum()))
                m4.metric("Avg Score",
                          round(float(df_a["score"].mean()),4))

                ac1,ac2 = st.columns([2,3])
                with ac1:
                    fig_a = px.histogram(
                        df_a, x="score", nbins=25,
                        color_discrete_sequence=[col_h],
                        title=f"Level {lvl} Score Distribution"
                    )
                    fig_a.update_layout(
                        height=220,
                        margin=dict(t=40,b=20,l=20,r=20),
                        plot_bgcolor=WHITE,
                        paper_bgcolor=WHITE,
                        font=dict(color=TEXT,size=11)
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
                with ac2:
                    st.dataframe(df_a, use_container_width=True,
                                 height=240, hide_index=True)

                if role in ["officer","admin"]:
                    st.divider()
                    sh("Resolve Alert")
                    rc1,rc2,rc3 = st.columns([2,2,1])
                    aid = rc1.number_input(
                        "Alert ID", min_value=1, step=1,
                        key=f"aid{lvl}")
                    itp = rc2.selectbox(
                        "Outcome",
                        ["True Positive","False Positive"],
                        key=f"tp{lvl}")
                    if rc3.button("Resolve ✓",
                                  type="primary",
                                  key=f"res{lvl}",
                                  use_container_width=True):
                        try:
                            execute("""
                                UPDATE alerts
                                SET resolved=TRUE,
                                    was_true_positive=%s
                                WHERE alert_id=%s
                            """, (itp=="True Positive", int(aid)))
                            st.success(f"Alert #{int(aid)} resolved.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    # ════════════════════════════════════════════════════════
    # PAGE: MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════
    elif "Performance" in page:
        st.markdown("<h2>📈 Model Performance</h2>",
                    unsafe_allow_html=True)
        st.caption(
            "Source: Paper Table 1 · Python 3.10 · "
            "random_state=42 · n=1,840 test set · 5-fold CV"
        )

        df_r = pd.DataFrame({
            "Algorithm":[
                "Rule-Based Baseline","One-Class SVM",
                "Local Outlier Factor","Isolation Forest Only",
                "Temporal Model Only","Autoencoder",
                "Random Forest †","XGBoost †",
                "MMADF Ensemble ◄"
            ],
            "Type":[
                "Unsupervised","Unsupervised","Unsupervised",
                "Unsupervised","Unsupervised","Unsupervised",
                "Supervised †","Supervised †","Unsupervised"
            ],
            "F1":  [0.675,0.710,0.737,0.818,0.848,
                    0.810,0.787,0.825,0.883],
            "Prec":[0.711,0.739,0.763,0.847,0.862,
                    0.824,0.801,0.838,0.891],
            "Rec": [0.643,0.682,0.712,0.791,0.834,
                    0.796,0.774,0.812,0.876],
            "FPR": [0.142,0.118,0.103,0.084,0.071,
                    0.088,0.091,0.082,0.053],
            "F1 vs MMADF":[
                "+30.8%","+24.4%","+19.8%","+7.9%",
                "+4.1%","+8.9%","+12.2%","+7.0%","—"
            ]
        })

        colours = [
            "#BDC3C7","#95A5A6","#7F8C8D",
            BLUE, TEAL,
            "#5D6D7E",
            "#A93226","#922B21",
            NAVY
        ]

        t1,t2,t3 = st.tabs([
            "📊  Benchmark Chart",
            "📋  Full Results Table",
            "📉  ROC Curves"
        ])

        with t1:
            bc1,bc2 = st.columns(2)
            with bc1:
                sh("F1, Precision and Recall by Algorithm")
                fig = go.Figure()
                for i,row in df_r.iterrows():
                    fig.add_trace(go.Bar(
                        name=row["Algorithm"].replace(" ◄",""),
                        x=["Precision","Recall","F1"],
                        y=[row["Prec"],row["Rec"],row["F1"]],
                        marker_color=colours[i],
                        text=[f"{v:.3f}" for v in
                              [row["Prec"],row["Rec"],row["F1"]]],
                        textposition="outside",
                        textfont=dict(size=9)
                    ))
                fig.update_layout(
                    barmode="group", height=420,
                    yaxis=dict(range=[0.55,1.02]),
                    legend=dict(
                        orientation="h",y=-0.5,
                        font=dict(size=8,color=TEXT)
                    ),
                    margin=dict(t=10,b=160,l=20,r=20),
                    plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                    font=dict(color=TEXT,size=11)
                )
                st.plotly_chart(fig, use_container_width=True)

            with bc2:
                sh("False Positive Rate Comparison")
                fig_fpr = go.Figure()
                fpr_colours = [
                    RED if a=="MMADF Ensemble ◄" else DGREY
                    for a in df_r["Algorithm"]
                ]
                fig_fpr.add_trace(go.Bar(
                    x=df_r["Algorithm"],
                    y=df_r["FPR"],
                    marker_color=fpr_colours,
                    text=[f"{v:.1%}" for v in df_r["FPR"]],
                    textposition="outside",
                    textfont=dict(size=10)
                ))
                fig_fpr.update_layout(
                    height=420,
                    yaxis_title="False Positive Rate",
                    yaxis=dict(range=[0,0.19]),
                    xaxis=dict(
                        tickangle=-40,
                        tickfont=dict(size=9,color=TEXT)
                    ),
                    margin=dict(t=10,b=140,l=20,r=20),
                    plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                    showlegend=False,
                    font=dict(color=TEXT,size=11)
                )
                st.plotly_chart(fig_fpr, use_container_width=True)

            st.divider()
            km1,km2,km3,km4 = st.columns(4)
            km1.metric("MMADF F1",  "0.883",  "Best overall")
            km2.metric("FPR",       "5.3%",   "↓ 62.7% vs baseline")
            km3.metric("vs XGBoost†", "+7.0%","Unsupervised beats supervised")
            km4.metric("McNemar",   "p < 0.001","All 7 comparators")

        with t2:
            sh("Extended Benchmark — All 9 Algorithms (Table 1 from Paper)")
            st.dataframe(
                df_r.style
                .highlight_max(
                    subset=["F1","Prec","Rec"],
                    color="#D5E8D4"
                )
                .highlight_min(
                    subset=["FPR"],
                    color="#D5E8D4"
                )
                .format({
                    "F1":"{:.3f}","Prec":"{:.3f}",
                    "Rec":"{:.3f}","FPR":"{:.3f}"
                }),
                use_container_width=True,
                hide_index=True,
                height=380
            )
            st.caption(
                "† Supervised methods require confirmed fraud labels — "
                "included as upper-bound reference only."
            )
            st.divider()
            sh("McNemar Statistical Significance Tests (Table 2 from Paper)")
            mc_df = pd.DataFrame({
                "Comparison":[
                    "vs Rule-Based","vs One-Class SVM","vs LOF",
                    "vs Autoencoder","vs IF Only",
                    "vs Temporal Only","vs XGBoost †"
                ],
                "chi²":["89.4","61.7","47.3","19.6","12.7","8.3","15.2"],
                "p-value":[
                    "<0.001","<0.001","<0.001","<0.001",
                    "=0.003","=0.008","=0.001"
                ],
                "F1 Improvement":[
                    "+30.8%","+24.4%","+19.8%",
                    "+8.9%","+7.9%","+4.1%","+7.0%"
                ],
                "FPR Reduction":[
                    "62.7%","55.1%","48.5%",
                    "39.8%","36.9%","25.4%","35.4%"
                ],
                "Decision":["Reject H₀"]*7
            })
            st.dataframe(mc_df, use_container_width=True,
                         hide_index=True)

            st.divider()
            sh("5-Fold Cross-Validation — MMADF Ensemble")
            cv_df = pd.DataFrame({
                "Metric":["Precision","Recall","F1-Score","FPR"],
                "Fold 1":["0.887","0.871","0.879","0.058"],
                "Fold 2":["0.893","0.879","0.886","0.051"],
                "Fold 3":["0.901","0.891","0.896","0.047"],
                "Fold 4":["0.882","0.861","0.871","0.061"],
                "Fold 5":["0.892","0.878","0.885","0.048"],
                "Mean":["0.891","0.876","0.883","0.053"],
                "Std Dev":["±0.012","±0.015","±0.011","±0.008"],
                "95% CI":[
                    "[0.879, 0.903]","[0.861, 0.891]",
                    "[0.872, 0.894]","[0.045, 0.061]"
                ],
            })
            st.dataframe(cv_df, use_container_width=True,
                         hide_index=True)

        with t3:
            sh("ROC Curves — Model Comparison")
            try:
                from scipy.interpolate import interp1d
                roc_data = {
                    "Isolation Forest":(
                        [0,0.02,0.05,0.10,0.20,0.40,1],
                        [0,0.54,0.72,0.82,0.90,0.95,1],
                        0.912, BLUE, "dash"
                    ),
                    "Temporal Model":(
                        [0,0.02,0.05,0.10,0.20,0.40,1],
                        [0,0.59,0.77,0.86,0.93,0.97,1],
                        0.931, TEAL, "dashdot"
                    ),
                    "MMADF Ensemble":(
                        [0,0.02,0.05,0.10,0.20,0.40,1],
                        [0,0.64,0.81,0.89,0.95,0.98,1],
                        0.947, NAVY, "solid"
                    ),
                }
                fig_roc = go.Figure()
                xs = np.linspace(0,1,400)
                for nm,(fp,tp,auc,col,dash) in roc_data.items():
                    f  = interp1d(fp,tp,kind="cubic")
                    ys = np.clip(f(xs),0,1)
                    fig_roc.add_trace(go.Scatter(
                        x=xs,y=ys,
                        name=f"{nm}  (AUC = {auc})",
                        line=dict(color=col,dash=dash,width=2.5)
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0.053],y=[0.876],mode="markers",
                    marker=dict(color=RED,size=14,symbol="star"),
                    name="MMADF Operating Point"
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0,1],y=[0,1],mode="lines",
                    line=dict(dash="dot",color=DGREY,width=1),
                    showlegend=False
                ))
                fig_roc.update_layout(
                    height=500,
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate (Recall)",
                    legend=dict(x=0.55,y=0.12,
                                font=dict(size=11,color=TEXT)),
                    plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                    margin=dict(t=20,b=50,l=50,r=20),
                    font=dict(color=TEXT,size=12)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            except ImportError:
                st.warning(
                    "Run: **pip install scipy** then restart Streamlit")

    # ════════════════════════════════════════════════════════
    # PAGE: SCORE EXPLORER
    # ════════════════════════════════════════════════════════
    elif "Score Explorer" in page:
        st.markdown("<h2>🔍 Score Explorer</h2>",
                    unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns(4)
        min_s = f1.slider("Min Score", 0.0,1.0,0.0,0.01)
        max_s = f2.slider("Max Score", 0.0,1.0,1.0,0.01)
        sf    = f3.selectbox("Status Filter",
                             ["All","flagged","assessed","pending"])
        lim   = f4.selectbox("Max Rows",[50,100,250,500],index=1)
        sw    = f"AND d.status='{sf}'" if sf!="All" else ""

        df_ex = query(f"""
            SELECT d.dec_id, dec.full_name AS entity,
                   d.hs_code,
                   d.declared_value_usd     AS value_usd,
                   d.origin_country,
                   TO_CHAR(d.submission_time,
                       'YYYY-MM-DD HH24:MI') AS submitted,
                   d.status,
                   COALESCE(d.is_fraud_confirmed,FALSE) AS confirmed,
                   ROUND(s.ensemble_score::numeric,4) AS ensemble,
                   ROUND(s.if_score::numeric,4)       AS if_score,
                   ROUND(COALESCE(s.lstm_score,0)::numeric,4)
                       AS temporal
            FROM declarations d
            JOIN anomaly_scores s ON s.dec_id=d.dec_id
            JOIN declarants dec
                ON dec.declarant_id=d.declarant_id
            WHERE s.ensemble_score BETWEEN {min_s} AND {max_s} {sw}
            ORDER BY s.ensemble_score DESC LIMIT {lim}
        """)

        if df_ex is None or df_ex.empty:
            st.info("No records in this score range.")
        else:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Records",         len(df_ex))
            m2.metric("Avg Score",
                      round(float(df_ex["ensemble"].mean()),4))
            m3.metric("Max Score",
                      round(float(df_ex["ensemble"].max()),4))
            m4.metric("Confirmed Fraud",
                      int(df_ex["confirmed"].sum()))

            sc1,sc2 = st.columns(2)
            with sc1:
                sh("IF Score vs Temporal Score")
                try:
                    fig_sc = px.scatter(
                        df_ex, x="if_score", y="temporal",
                        color="ensemble",
                        color_continuous_scale=[
                            TEAL,BLUE,NAVY,RED
                        ],
                        hover_data=[
                            "dec_id","entity","value_usd","confirmed"
                        ],
                        labels={
                            "if_score":"Isolation Forest Score",
                            "temporal":"Temporal Score",
                            "ensemble":"Ensemble Score"
                        }
                    )
                    fig_sc.add_hline(
                        y=ENSEMBLE_LEVEL1,line_dash="dash",
                        line_color=AMBER,
                        annotation_text="L1"
                    )
                    fig_sc.add_hline(
                        y=ENSEMBLE_LEVEL2,line_dash="solid",
                        line_color=RED,
                        annotation_text="L2"
                    )
                    fig_sc.add_vline(
                        x=ENSEMBLE_LEVEL1,line_dash="dash",
                        line_color=AMBER
                    )
                    fig_sc.update_layout(
                        height=380,
                        plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                        margin=dict(t=20,b=40,l=40,r=20),
                        font=dict(color=TEXT,size=11)
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)
                except Exception:
                    pass

            with sc2:
                sh("Score Distribution — Filtered Records")
                try:
                    fig_h = px.histogram(
                        df_ex, x="ensemble",
                        nbins=30,
                        color_discrete_sequence=[BLUE],
                        labels={"ensemble":"Ensemble Score",
                                "count":"Records"}
                    )
                    fig_h.add_vline(
                        x=ENSEMBLE_LEVEL1,line_dash="dash",
                        line_color=AMBER,
                        annotation_text="L1"
                    )
                    fig_h.add_vline(
                        x=ENSEMBLE_LEVEL2,line_dash="solid",
                        line_color=RED,
                        annotation_text="L2"
                    )
                    fig_h.update_layout(
                        height=380,
                        plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                        margin=dict(t=20,b=40,l=40,r=20),
                        font=dict(color=TEXT,size=11)
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                except Exception:
                    pass

            st.divider()
            sh("Transaction Records")
            st.dataframe(df_ex, use_container_width=True,
                         height=360, hide_index=True)

    # ════════════════════════════════════════════════════════
    # PAGE: RESEARCH OVERVIEW
    # ════════════════════════════════════════════════════════
    elif "Research" in page:
        st.markdown(f"""
        <div style="background:{NAVY};color:white;
                    padding:18px 22px;border-radius:8px;
                    margin-bottom:16px;">
            <h3 style="color:white;margin:0 0 6px;font-size:1.1rem;">
                AI-Driven Database Anomaly Detection for Fraud Prevention
            </h3>
            <p style="color:#AED6F1;margin:0 0 2px;
                      font-size:0.88rem;font-weight:600;">
                A Hybrid Machine Learning Framework for Relational
                Transaction Systems
            </p>
            <p style="color:rgba(255,255,255,0.55);
                      margin:0;font-size:0.80rem;">
                Mike T. Ngwere &nbsp;·&nbsp; R186209Q &nbsp;·&nbsp;
                MCS 504 Database Engineering &nbsp;·&nbsp;
                University of Zimbabwe &nbsp;·&nbsp; 2026
            </p>
        </div>
        """, unsafe_allow_html=True)

        t1,t2,t3,t4,t5 = st.tabs([
            "📋  Abstract & Problem",
            "🔍  Research Gaps",
            "🏗  Architecture",
            "📊  Results",
            "🏆  Contributions"
        ])

        with t1:
            sh("Research Problem — Database Engineering Gap")
            st.info(
                "**Core Gap:** Relational transaction databases lack "
                "native AI-driven anomaly detection capability. "
                "Detection is performed externally via ETL pipelines — "
                "losing the temporal relational context that is "
                "demonstrably the strongest predictor of fraud. "
                "MMADF closes this by embedding Isolation Forest and "
                "temporal sequence reconstruction directly within "
                "PostgreSQL using native window functions and "
                "materialised views."
            )
            c1,c2 = st.columns(2)
            with c1:
                sh("Three-Phase Methodology")
                st.markdown("""
| Phase | Method | Output |
|---|---|---|
| Phase 1 | SLR — PRISMA (Moher et al., 2009) | 30 studies · 1 core gap |
| Phase 2 | Design Science Research (Hevner et al., 2004) | MMADF prototype |
| Phase 3 | Simulation + McNemar (Dietterich, 1998) | Validated results |
                """)
                sh("Research Questions")
                for rq,text in [
                    ("RQ1","What DB Engineering gap does the literature reveal?"),
                    ("RQ2","How can IF + temporal detection be embedded in PostgreSQL?"),
                    ("RQ3","Does MMADF outperform baselines with statistical significance?"),
                ]:
                    st.markdown(f"**{rq}:** {text}")
            with c2:
                sh("Four Research Objectives")
                for obj,desc in [
                    ("OBJ1","SLR — formally identify the database engineering gap"),
                    ("OBJ2","PostgreSQL schema + SQL window function features"),
                    ("OBJ3","MMADF prototype — 4-tier stack + monitoring dashboard"),
                    ("OBJ4","Evaluation — 7 baselines · McNemar · sensitivity analysis"),
                ]:
                    st.markdown(
                        f"<div style='background:{LIGHT};"
                        f"border-left:4px solid {TEAL};"
                        f"padding:8px 12px;margin:5px 0;"
                        f"border-radius:0 4px 4px 0;'>"
                        f"<b style='color:{NAVY};'>{obj}:</b> "
                        f"<span style='color:{TEXT};'>{desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        with t2:
            sh("Three Dimensions of the Database Engineering Gap")
            for dim,title,body in [
                ("1","External Feature Engineering — What is Missing",
                 "All reviewed ML fraud detection systems compute temporal "
                 "features after ETL extraction. PostgreSQL window functions "
                 "(COUNT(*) OVER, LAG, PERCENTILE_CONT) can compute per-entity "
                 "temporal features directly on transaction sequences — but no "
                 "published study applies this to anomaly detection."),
                ("2","External Anomaly Scoring — No Database Integration",
                 "All reviewed systems score anomalies in external ML platforms, "
                 "requiring separate infrastructure. No reviewed study stores "
                 "anomaly scores and SHAP explainability within the transaction "
                 "database as SQL-queryable records governed by RDBMS access control."),
                ("3","Absent Data Governance — No Regulatory Framework",
                 "RBAC, row-level data isolation, and audit logging provided by "
                 "modern RDBMS (Bertino and Sandhu, 2005) have not been applied in "
                 "published anomaly detection frameworks — leaving detection systems "
                 "without the compliance mechanisms required for financial and "
                 "government deployment."),
            ]:
                st.markdown(
                    f"<div style='border:1px solid {GREY};"
                    f"border-left:5px solid {BLUE};"
                    f"padding:12px 16px;border-radius:0 6px 6px 0;"
                    f"margin:8px 0;background:{WHITE};'>"
                    f"<p style='color:{NAVY};font-weight:700;margin:0 0 6px;'>"
                    f"Dimension {dim}: {title}</p>"
                    f"<p style='color:{TEXT};margin:0;font-size:0.9rem;'>"
                    f"{body}</p></div>",
                    unsafe_allow_html=True
                )
            st.divider()
            gm1,gm2,gm3 = st.columns(3)
            gm1.metric("Studies Reviewed","30")
            gm2.metric("Core DB Gap","1  (database-native ML)")
            gm3.metric("Gap Dimensions","3")

        with t3:
            sh("Five Detection Layers")
            a1,a2 = st.columns([3,2])
            with a1:
                for num,lname,desc,bg in [
                    ("1","Data Ingestion & Schema",
                     "8-table normalised PostgreSQL 15 schema · "
                     "value_baselines materialised view",
                     "#455A64"),
                    ("2","SQL Feature Engineering",
                     "7 features via PostgreSQL window functions · "
                     "value_deviation_ratio = 31.4% score importance",
                     BLUE),
                    ("3","Isolation Forest",
                     "n_estimators=200 · contamination=0.05 · "
                     "random_state=42 · F1=0.818",
                     "#6A1B9A"),
                    ("4","Temporal Sequence Detection",
                     "15-step per-entity sliding window · "
                     "MSE reconstruction scoring · F1=0.848",
                     "#B7410E"),
                    ("5","Ensemble + SHAP Alert Engine",
                     "S=0.5×IF+0.5×Temporal · L1≥0.65 · "
                     "L2≥0.85 · SHAP stored as JSONB in DB",
                     NAVY),
                ]:
                    st.markdown(
                        f"<div style='background:{bg};color:white;"
                        f"padding:10px 14px;border-radius:6px;"
                        f"margin:5px 0;'>"
                        f"<b>Layer {num}: {lname}</b><br>"
                        f"<small style='opacity:0.85;'>{desc}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            with a2:
                sh("Four Deployment Tiers")
                for icon,tech,desc,bg in [
                    ("🗄","PostgreSQL 15",
                     "8 tables · RLS · pgaudit · pgcrypto",BLUE),
                    ("🐍","Python 3.10",
                     "scikit-learn 1.3.2 · random_state=42",TEAL),
                    ("🔌","Flask 3.0",
                     "8 REST endpoints · port 5001",NAVY),
                    ("📊","Streamlit 1.30",
                     "This dashboard · role-based auth","#455A64"),
                ]:
                    st.markdown(
                        f"<div style='background:{bg};color:white;"
                        f"padding:9px 14px;border-radius:6px;"
                        f"margin:5px 0;'>"
                        f"<b>{icon} {tech}</b><br>"
                        f"<small style='opacity:0.85;'>{desc}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                sh("STRIDE Fraud Archetypes")
                for a,s in [
                    ("Value Under-Declaration","Tampering"),
                    ("Temporal Burst Activity","Spoofing"),
                    ("Entity Identity Reuse","Spoofing + EoP"),
                    ("Product Code Misclassification","Tampering"),
                ]:
                    st.markdown(
                        f"<div style='padding:4px 8px;'>"
                        f"<b style='color:{NAVY};'>{a}</b>"
                        f" <span style='color:{DGREY};'>"
                        f"— {s}</span></div>",
                        unsafe_allow_html=True
                    )

        with t4:
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("F1-Score","0.883","95% CI [0.872, 0.894]")
            r2.metric("FPR","5.3%","↓ 62.7% vs baseline")
            r3.metric("vs XGBoost†","+7.0% F1","Unsupervised wins")
            r4.metric("McNemar","p < 0.001","All 7 comparators")
            st.divider()
            rc1,rc2 = st.columns(2)
            with rc1:
                sh("F1-Score Benchmark")
                bench = pd.DataFrame({
                    "Algorithm":[
                        "Rule-Based","OC-SVM","LOF","IF Only",
                        "Temporal","Autoencoder",
                        "RF†","XGBoost†","MMADF"
                    ],
                    "F1":[
                        0.675,0.710,0.737,0.818,
                        0.848,0.810,0.787,0.825,0.883
                    ],
                })
                cols_bar = [
                    DGREY,DGREY,DGREY,BLUE,
                    TEAL,DGREY,DGREY,DGREY,NAVY
                ]
                fig_b = px.bar(
                    bench, x="Algorithm", y="F1",
                    text=[f"{v:.3f}" for v in bench["F1"]],
                    color="Algorithm",
                    color_discrete_sequence=cols_bar
                )
                fig_b.update_layout(
                    height=350,showlegend=False,
                    yaxis=dict(range=[0.55,1.0]),
                    xaxis=dict(tickangle=-30,
                               tickfont=dict(size=9)),
                    margin=dict(t=10,b=100,l=20,r=20),
                    plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                    font=dict(color=TEXT,size=11)
                )
                fig_b.update_traces(textposition="outside",
                                    textfont=dict(size=9))
                st.plotly_chart(fig_b, use_container_width=True)

            with rc2:
                sh("Sensitivity Analysis — Fraud Prevalence Levels")
                sens = pd.DataFrame({
                    "Fraud Rate":[
                        "2%","5%","8% (primary)","15%","20%"
                    ],
                    "F1":[0.857,0.870,0.883,0.880,0.879],
                    "FPR":[0.068,0.059,0.053,0.061,0.064],
                })
                fig_s = px.line(
                    sens, x="Fraud Rate", y=["F1","FPR"],
                    markers=True,
                    color_discrete_map={"F1":NAVY,"FPR":RED}
                )
                fig_s.update_layout(
                    height=350,
                    margin=dict(t=10,b=40,l=20,r=20),
                    plot_bgcolor=WHITE,paper_bgcolor=WHITE,
                    legend=dict(orientation="h",y=1.1),
                    font=dict(color=TEXT,size=11)
                )
                st.plotly_chart(fig_s, use_container_width=True)
                st.success(
                    "MMADF F1: 0.857–0.883 across all prevalence levels"
                )

        with t5:
            sh("Five Research Contributions")
            for num,title,body,col in [
                ("1","Systematic Literature Synthesis",
                 "30 studies reviewed · 1 core DB Engineering gap formally "
                 "evidenced with 3 specific dimensions (Moher et al., 2009)",
                 NAVY),
                ("2","Database-Native ML Architecture",
                 "Isolation Forest + temporal reconstruction embedded in "
                 "PostgreSQL via native window functions and materialised "
                 "views — no ETL extraction required",
                 BLUE),
                ("3","SQL Feature Engineering",
                 "7 fraud features computed entirely in SQL · "
                 "value_deviation_ratio = 31.4% of ensemble score variance",
                 TEAL),
                ("4","Rigorous Empirical Evaluation",
                 "7-algorithm benchmark · 5-fold CV · McNemar (Dietterich, "
                 "1998) · sensitivity analysis 2%–20% fraud prevalence",
                 "#6A1B9A"),
                ("5","Domain-Agnostic Design",
                 "Applicable to banking, insurance, healthcare — any "
                 "relational transaction database without infrastructure "
                 "dependency",
                 "#B7410E"),
            ]:
                st.markdown(
                    f"<div style='display:flex;gap:12px;margin:8px 0;'>"
                    f"<div style='background:{col};color:white;"
                    f"min-width:36px;height:36px;border-radius:50%;"
                    f"display:flex;align-items:center;"
                    f"justify-content:center;font-weight:700;"
                    f"font-size:1rem;flex-shrink:0;'>{num}</div>"
                    f"<div style='background:{LIGHT};padding:9px 14px;"
                    f"border-radius:6px;flex:1;"
                    f"border-left:3px solid {col};'>"
                    f"<b style='color:{col};'>{title}</b><br>"
                    f"<span style='font-size:0.88rem;color:{TEXT};'>"
                    f"{body}</span></div></div>",
                    unsafe_allow_html=True
                )
            st.divider()
            sh("Target Journals")
            j1,j2,j3 = st.columns(3)
            j1.info(
                "**African Journal of Information Systems**\n"
                "African ICT and AI — strong fit"
            )
            j2.info(
                "**Information Development**\n"
                "Developing economy ICT — direct alignment"
            )
            j3.info(
                "**Electronic Government**\n"
                "Public sector AI and governance"
            )

    # ════════════════════════════════════════════════════════
    # PAGE: SUBMIT TRANSACTION
    # ════════════════════════════════════════════════════════
    elif "Submit" in page:
        st.markdown("<h2>📝 Submit Transaction for Scoring</h2>",
                    unsafe_allow_html=True)
        st.caption(
            "Transaction is scored through the full MMADF "
            "pipeline in real-time."
        )
        df_ent  = query("""
            SELECT declarant_id,
                   reg_number||' — '||full_name AS label
            FROM declarants ORDER BY full_name LIMIT 200
        """)
        df_prod = query("""
            SELECT hs_code,
                   hs_code||' — '||description AS label,
                   avg_value_per_kg
            FROM commodities ORDER BY hs_code
        """)
        if df_ent.empty:
            st.warning(
                "No entities registered. "
                "Add one under 👥 Declarants.")
            return
        if df_prod.empty:
            st.warning("No products in database.")
            return

        with st.form("decl_form", clear_on_submit=True):
            sh("Transaction Details")
            r1c1,r1c2 = st.columns(2)
            ei = r1c1.selectbox(
                "Entity *",
                range(len(df_ent)),
                format_func=lambda i: df_ent.iloc[i]["label"]
            )
            entity_id = int(df_ent.iloc[ei]["declarant_id"])
            pi = r1c2.selectbox(
                "Product (HS Code) *",
                range(len(df_prod)),
                format_func=lambda i: df_prod.iloc[i]["label"]
            )
            hs_code = df_prod.iloc[pi]["hs_code"]
            avg_val = float(
                df_prod.iloc[pi]["avg_value_per_kg"] or 100)
            r2c1,r2c2,r2c3 = st.columns(3)
            declared_value = r2c1.number_input(
                "Declared Value (USD) *",
                min_value=0.01,
                value=float(avg_val*10),
                step=10.0, format="%.2f"
            )
            weight = r2c2.number_input(
                "Weight (kg)", min_value=0.0,
                value=10.0, step=1.0)
            origin = r2c3.selectbox(
                "Origin Country",
                ["MOZ","ZAF","CHN","IND","GBR",
                 "ARE","DEU","USA","KEN","ZMB"]
            )
            r3c1,r3c2 = st.columns(2)
            s_time = r3c1.time_input(
                "Submission Time",
                value=datetime.now().time()
            )
            border = r3c2.selectbox(
                "Border Post",
                ["FORBES","BEITBRIDGE","CHIRUNDU","KARIBA"]
            )
            sub = st.form_submit_button(
                "Submit & Score →",
                type="primary",
                use_container_width=True
            )

        if sub:
            sub_dt = datetime.combine(datetime.today(), s_time)
            try:
                result = execute_return("""
                    INSERT INTO declarations
                        (declarant_id,hs_code,declared_value_usd,
                         declared_weight_kg,origin_country,
                         submission_time,border_post,status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING dec_id
                """, (entity_id,hs_code,declared_value,weight,
                      origin,sub_dt,border,"pending"))
                dec_id = result[0]
                execute(
                    "REFRESH MATERIALIZED VIEW value_baselines")

                with st.spinner("Scoring through MMADF pipeline..."):
                    from src.isolation_forest import load as load_if, score_batch
                    from src.lstm_model import load as load_lstm, score_declarant
                    from src.ensemble import compute_ensemble, classify_fraud_type, compute_shap
                    from src.feature_extractor import extract_features

                    df_feat = extract_features(limit=1, unscored_only=False)
                    df_new  = df_feat[df_feat["dec_id"]==dec_id]
                    if df_new.empty:
                        df_new = extract_features(limit=1)

                    if_model, if_scaler = load_if()
                    df_scored = score_batch(df_new, if_model, if_scaler)
                    if_score  = float(df_scored["if_score"].iloc[0] if not df_scored.empty else 0.5)

                    lstm_model, lstm_thresh = load_lstm()
                    lstm_score, _ = score_declarant(entity_id, lstm_model, lstm_thresh)

                    ens = compute_ensemble(if_score, lstm_score)
                    feat_vec = {f: float(df_new[f].iloc[0]) if f in df_new.columns else 0.0 for f in FEATURES}
                    fraud_type = classify_fraud_type(feat_vec)
                    shap_vals  = compute_shap(feat_vec, if_model, if_scaler) if ens >= ENSEMBLE_LEVEL2 else None

                    score_row = execute_return("""
                        INSERT INTO anomaly_scores
                            (dec_id, if_score, lstm_score, ensemble_score, feature_vector)
                        VALUES (%s,%s,%s,%s,%s)
                        ON CONFLICT (dec_id) DO UPDATE
                            SET ensemble_score=EXCLUDED.ensemble_score
                        RETURNING score_id
                    """, (dec_id, if_score, lstm_score, ens, psycopg2.extras.Json(feat_vec)))
                    score_id = score_row[0] if score_row else None

                    if ens >= ENSEMBLE_LEVEL1 and score_id:
                        level = 2 if ens >= ENSEMBLE_LEVEL2 else 1
                        execute("""
                            INSERT INTO alerts
                                (dec_id, score_id, alert_level,
                                 fraud_type_predicted, feature_contribution)
                            VALUES (%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING
                        """, (dec_id, score_id, level, fraud_type,
                              psycopg2.extras.Json(shap_vals) if shap_vals else None))
                        if level == 2:
                            execute("UPDATE declarations SET status='flagged' WHERE dec_id=%s", (dec_id,))

                st.divider()
                sh(f"Transaction #{dec_id} — Detection Result")
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Transaction ID", f"#{dec_id}")
                rc2.metric("IF Score",       f"{if_score:.4f}")
                rc3.metric("Temporal Score", f"{lstm_score:.4f}" if lstm_score else "N/A")
                rc4.metric("Ensemble Score", f"{ens:.4f}")

                if ens >= ENSEMBLE_LEVEL2:
                    st.markdown(
                        f"<div class='alert-l2'><h4 style='color:{RED};margin:0;'>"
                        f"🚨 Level 2 Alert — Enforcement Escalation</h4>"
                        f"Score: <b>{ens:.4f}</b> (threshold ≥{ENSEMBLE_LEVEL2}) | "
                        f"Fraud Type: <b>{fraud_type.replace('_',' ').title()}</b></div>",
                        unsafe_allow_html=True)
                elif ens >= ENSEMBLE_LEVEL1:
                    st.markdown(
                        f"<div class='alert-l1'><h4 style='color:{AMBER};margin:0;'>"
                        f"⚠️ Level 1 Alert — Review Required</h4>"
                        f"Score: <b>{ens:.4f}</b> | Type: <b>{fraud_type.replace('_',' ').title()}</b></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='alert-clear'><h4 style='color:{GREEN};margin:0;'>"
                        f"✅ Clear — No Anomaly Detected</h4>"
                        f"Score: <b>{ens:.4f}</b> (below threshold {ENSEMBLE_LEVEL1})</div>",
                        unsafe_allow_html=True)

                with st.expander("Feature Vector Details"):
                    fv_df = pd.DataFrame([{"Feature": f, "Value": round(float(v), 4)} for f, v in feat_vec.items()])
                    st.dataframe(fv_df, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    # ════════════════════════════════════════════════════════
    # PAGE: DECLARANTS
    # ════════════════════════════════════════════════════════
    elif "Declarants" in page:
        st.markdown("<h2>👥 Entity / Declarant Management</h2>", unsafe_allow_html=True)
        tl, ta = st.tabs(["📋  Entity List", "➕  Register New"])
        with tl:
            df_d = query("""
                SELECT dec.declarant_id, dec.reg_number, dec.full_name,
                       dec.country_of_origin, dec.risk_tier,
                       COUNT(d.dec_id) AS transactions,
                       COUNT(a.alert_id) AS alerts,
                       ROUND(MAX(s.ensemble_score)::numeric,3) AS max_score
                FROM declarants dec
                LEFT JOIN declarations d ON d.declarant_id=dec.declarant_id
                LEFT JOIN anomaly_scores s ON s.dec_id=d.dec_id
                LEFT JOIN alerts a ON a.dec_id=d.dec_id
                GROUP BY dec.declarant_id
                ORDER BY alerts DESC, max_score DESC NULLS LAST
            """)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Entities", len(df_d))
            m2.metric("High Risk (Tier 4-5)", int((df_d["risk_tier"]>=4).sum()) if not df_d.empty else 0)
            m3.metric("With Alerts", int((df_d["alerts"]>0).sum()) if not df_d.empty else 0)
            st.dataframe(df_d, use_container_width=True, hide_index=True, height=440)
        with ta:
            with st.form("add_entity"):
                sh("Register New Entity")
                f1, f2 = st.columns(2)
                reg_no  = f1.text_input("Registration Number *", placeholder="REG00501")
                full_nm = f2.text_input("Full Name / Organisation *")
                country = f1.selectbox("Country *", ["MOZ","ZAF","CHN","IND","GBR","ARE","DEU","USA","KEN","ZMB"])
                risk    = f2.selectbox("Risk Tier (1=Low, 5=High) *", [1,2,3,4,5])
                reg_sub = st.form_submit_button("Register Entity →", type="primary", use_container_width=True)
            if reg_sub:
                if not reg_no or not full_nm:
                    st.error("Registration number and name required.")
                else:
                    try:
                        execute("INSERT INTO declarants (reg_number,full_name,country_of_origin,risk_tier) VALUES (%s,%s,%s,%s)",
                                (reg_no, full_nm, country, risk))
                        st.success(f"Entity '{full_nm}' registered.")
                        st.rerun()
                    except Exception as e:
                        st.error("Registration number already exists." if "unique" in str(e).lower() else str(e))

    # ════════════════════════════════════════════════════════
    # PAGE: COMMODITIES
    # ════════════════════════════════════════════════════════
    elif "Commodities" in page:
        st.markdown("<h2>📦 Product / Commodity Management</h2>", unsafe_allow_html=True)
        tl, ta = st.tabs(["📋  Product List", "➕  Add Product"])
        with tl:
            df_c = query("SELECT hs_code,description,risk_category,avg_value_per_kg,last_updated FROM commodities ORDER BY risk_category DESC,hs_code")
            st.metric("Total Products", len(df_c))
            st.dataframe(df_c, use_container_width=True, hide_index=True, height=460)
        with ta:
            with st.form("add_prod"):
                sh("Add New Product")
                f1, f2 = st.columns(2)
                hs     = f1.text_input("HS Code *", placeholder="8703.23")
                desc   = f2.text_input("Description *")
                risk_c = f1.selectbox("Risk Category (1=Low, 5=High) *", [1,2,3,4,5], index=2)
                avg_v  = f2.number_input("Avg Value per kg (USD) *", min_value=0.0, value=100.0, step=10.0)
                ps = st.form_submit_button("Add Product →", type="primary", use_container_width=True)
            if ps:
                if not hs or not desc:
                    st.error("HS code and description required.")
                else:
                    try:
                        execute("INSERT INTO commodities (hs_code,description,risk_category,avg_value_per_kg) VALUES (%s,%s,%s,%s)",
                                (hs, desc, risk_c, avg_v))
                        st.success(f"Product {hs} added.")
                        st.rerun()
                    except Exception as e:
                        st.error("HS code already exists." if "unique" in str(e).lower() else str(e))

    # ════════════════════════════════════════════════════════
    # PAGE: DETECTION CYCLE
    # ════════════════════════════════════════════════════════
    elif "Detection" in page:
        st.markdown("<h2>⚙️ Detection Cycle Management</h2>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sh("Run Detection Batch")
            batch = st.slider("Batch Size", 100, 10000, 500, 100)
            if st.button("▶  Run Detection Cycle", type="primary", use_container_width=True):
                with st.spinner("Running MMADF pipeline..."):
                    try:
                        from src.ensemble import run_detection_cycle
                        stats = run_detection_cycle(batch_size=batch, verbose=False)
                        st.success("Detection cycle complete.")
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Scored",    stats["scored"])
                        s2.metric("L1 Alerts", stats["alerts_l1"])
                        s3.metric("L2 Alerts", stats["alerts_l2"])
                    except Exception as e:
                        st.error(f"Error: {e}")
        with c2:
            sh("Database Status")
            status_df = query("""
                SELECT 'Declarations' AS metric, COUNT(*)::TEXT AS value FROM declarations
                UNION ALL SELECT 'Scored', COUNT(*)::TEXT FROM anomaly_scores
                UNION ALL SELECT 'L1 Alerts', COUNT(CASE WHEN alert_level=1 THEN 1 END)::TEXT FROM alerts
                UNION ALL SELECT 'L2 Alerts', COUNT(CASE WHEN alert_level=2 THEN 1 END)::TEXT FROM alerts
                UNION ALL SELECT 'Flagged', COUNT(*)::TEXT FROM declarations WHERE status='flagged'
            """)
            st.dataframe(status_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # PAGE: ADMIN
    # ════════════════════════════════════════════════════════
    elif "Admin" in page:
        st.markdown("<h2>🏛 Admin Panel</h2>", unsafe_allow_html=True)
        tu, tlog = st.tabs(["👤  User Management", "📋  Audit Log"])
        with tu:
            df_u = query("""
                SELECT user_id,username,full_name,role,border_post,
                       badge_no,is_active,
                       TO_CHAR(last_login,'YYYY-MM-DD HH24:MI') AS last_login
                FROM users ORDER BY user_id
            """)
            st.metric("Total Users", len(df_u))
            st.dataframe(df_u, use_container_width=True, hide_index=True)
            st.divider()
            sh("Toggle User Status")
            tc1, tc2, tc3 = st.columns(3)
            uid  = tc1.number_input("User ID", min_value=1, step=1)
            stat = tc2.selectbox("New Status", ["Active","Inactive"])
            if tc3.button("Update Status", type="primary", use_container_width=True):
                execute("UPDATE users SET is_active=%s WHERE user_id=%s", (stat=="Active", int(uid)))
                st.success(f"User #{int(uid)} set to {stat}")
                st.rerun()
        with tlog:
            df_audit = query("SELECT log_id,table_name,operation,changed_by,changed_at FROM audit_log ORDER BY changed_at DESC LIMIT 100")
            if not df_audit.empty:
                st.dataframe(df_audit, use_container_width=True, height=450, hide_index=True)
            else:
                st.info("No audit entries yet.")

    # ── Footer ─────────────────────────────────────────────
    st.markdown(
        f"<div class='footer'>"
        f"MMADF v1.0 &nbsp;·&nbsp; Multi-Model Anomaly Detection Framework &nbsp;·&nbsp; "
        f"MCS 504 Database Engineering &nbsp;·&nbsp; "
        f"Mike T. Ngwere &nbsp;·&nbsp; R186209Q &nbsp;·&nbsp; "
        f"University of Zimbabwe &nbsp;·&nbsp; 2026"
        f"</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════
if not st.session_state.get("logged_in"):
    login_page()
else:
    main_app()
