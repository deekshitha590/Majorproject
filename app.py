"""
Entrepreneurial Intelligence Platform for Sustainable Innovation
A platform to help entrepreneurs evaluate and improve their startup ideas
using market analysis, feasibility assessment, and business consulting.
"""
import base64
import json
import os
import sys
import textwrap
from collections import Counter
from datetime import datetime
import pandas as pd

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from modules.database import (
    init_db, register_user, login_user,
    save_analysis, get_user_analyses, get_analysis_by_id,
    delete_analysis,
)
from modules.ai_engine import analyze_idea, generate_ideas, API_KEY

st.set_page_config(
    page_title="Startup Intelligence Platform",
    page_icon="🚀", layout="wide",
    initial_sidebar_state="collapsed",
)

CSS_PATH = os.path.join(os.path.dirname(__file__), "static", "style.css")
with open(CSS_PATH, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Animated gradient mesh background (injected as real HTML divs) ──────────
st.markdown("""
<div class="mesh-bg">
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="blob blob-3"></div>
  <div class="grid"></div>
  <div class="vignette"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.tab-nav{display:flex;gap:.5rem;margin-bottom:1.5rem;flex-wrap:wrap}
.hist-card{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;margin-bottom:.7rem;transition:border-color .2s;cursor:pointer}
.hist-card:hover{border-color:var(--accent)}

.leaderboard-row{display:flex;align-items:center;gap:1rem;background:var(--surface2);border-radius:10px;padding:.7rem 1rem;margin-bottom:.5rem;border:1px solid var(--border)}
.rank-num{font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:var(--accent);min-width:32px}
.comp-col{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1.2rem;height:100%}
.avatar-face{width:90px;height:90px;border-radius:50%;background:linear-gradient(135deg,#00E5A0,#3B82F6);display:flex;align-items:center;justify-content:center;font-size:2.5rem;margin:0 auto .8rem;animation:pulse-avatar 2s infinite}
@keyframes pulse-avatar{0%,100%{box-shadow:0 0 20px rgba(0,229,160,.3)}50%{box-shadow:0 0 40px rgba(0,229,160,.7)}}
.avatar-bubble-ai{background:linear-gradient(135deg,rgba(0,229,160,.1),rgba(59,130,246,.1));border:1px solid rgba(0,229,160,.3);border-radius:4px 16px 16px 16px;padding:.9rem 1.1rem;margin:.5rem 0;color:#E8EAF0;font-size:.93rem;line-height:1.7}
.avatar-bubble-user{background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);border-radius:16px 4px 16px 16px;padding:.9rem 1.1rem;margin:.5rem 0;color:#CBD5E1;font-size:.93rem;text-align:right}
</style>
""", unsafe_allow_html=True)

init_db()

DEFAULTS = {
    "page": "home", "user": None,
    "analysis_result": None, "analysis_meta": {},
    "generate_result": None,
    "generate_meta": {},
    "_analyze_mode": "analyze",
    "active_tab": "analyze",
    "avatar_history": [],
    "current_analysis_id": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

BUDGET_OPTIONS = [
    "Bootstrapped (< ₹1 Lakh)",
    "Low (₹1 Lakh – ₹5 Lakh)",
    "Small (₹5 Lakh – ₹25 Lakh)",
    "Medium (₹25 Lakh – ₹1 Crore)",
    "Large (₹1 Crore – ₹5 Crore)",
    "High (> ₹5 Crore)",
]

DOMAIN_TRENDS = {
    "Technology / SaaS": 92, "Finance / FinTech": 90,
    "Healthcare / MedTech": 88, "Education / EdTech": 85,
    "Energy / CleanTech": 83, "Retail / E-Commerce": 80,
    "Logistics / Supply Chain": 78, "Agriculture / AgriTech": 75,
    "Food & Beverage": 72, "Real Estate / PropTech": 70,
    "Media & Entertainment": 68, "Travel & Tourism": 65,
    "Manufacturing": 60, "Other": 55,
}

DOMAINS = list(DOMAIN_TRENDS.keys())

# ── Dataset loaders ──────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load startup intelligence dataset (500 records, 2024-2026, 25 fields)."""
    csv_path = os.path.join(os.path.dirname(__file__), "startup_intelligence_dataset.csv")
    try:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        # Support both old (Amount_USD) and new (Amount_INR) column names
        if "Amount_INR" not in df.columns and "Amount_USD" in df.columns:
            df["Amount_INR"] = (pd.to_numeric(df["Amount_USD"], errors="coerce").fillna(0) * 83.5).round().astype("int64")
            df.drop(columns=["Amount_USD"], inplace=True)
        else:
            df["Amount_INR"] = pd.to_numeric(df["Amount_INR"], errors="coerce").fillna(0)
        return df
    except Exception as e:
        return None


@st.cache_data
def load_old_dataset():
    """Load original 2015-2017 funding dataset (205 records)."""
    csv_path = os.path.join(os.path.dirname(__file__), "startup_funding_india.csv")
    try:
        df = pd.read_csv(csv_path)
        # Normalise to INR
        if "Amount_INR" not in df.columns and "Amount_USD" in df.columns:
            df["Amount_INR"] = (pd.to_numeric(df["Amount_USD"], errors="coerce").fillna(0) * 83.5).round().astype("int64")
        else:
            df["Amount_INR"] = pd.to_numeric(df.get("Amount_INR", 0), errors="coerce").fillna(0)
        return df
    except Exception:
        return None


def _fmt_inr(amount_inr: float) -> str:
    """Format a rupee amount into a human-readable string (Cr / L / K)."""
    if amount_inr >= 1e7:
        return f"₹{amount_inr/1e7:.1f} Cr"
    elif amount_inr >= 1e5:
        return f"₹{amount_inr/1e5:.1f} L"
    elif amount_inr >= 1e3:
        return f"₹{amount_inr/1e3:.1f} K"
    else:
        return f"₹{int(amount_inr):,}"


def get_domain_stats(df, domain_keyword):
    """
    Calculate data-driven scores and insights from the 500-row dataset.
    Returns sustainability score, market demand, feasibility, scalability,
    competition level, and similar startup ideas — all from real data.
    """
    if df is None or not domain_keyword:
        return {}

    mask   = df["Domain"].str.contains(domain_keyword, case=False, na=False)
    subset = df[mask]
    if subset.empty:
        kw2  = domain_keyword.split("/")[0].strip()
        mask = df["Domain"].str.contains(kw2, case=False, na=False)
        subset = df[mask]
    if subset.empty:
        return {}

    funded     = len(subset)
    sustain_sc = int(subset["Sustainability_Score"].mean())
    mkt_demand = int(subset["Market_Demand_Score"].mean())
    exec_avg       = subset["Execution_Difficulty"].mean()
    feasibility_sc = int(100 - exec_avg)
    late_stages    = ["Series C","Series D","Series E","Series F","Private Equity"]
    late_count     = subset[subset["Investment_Stage"].isin(late_stages)].shape[0]
    scalability_sc = int(min(100, (late_count / max(funded, 1)) * 100 * 2.5))
    competition    = subset["Competition_Level"].mode()[0] if funded > 0 else "Medium"
    growth_rate    = round(subset["Growth_Rate_Pct"].mean(), 1)
    success_rate   = int(subset["Success_Rate"].mean())
    risk_level     = subset["Risk_Level"].mode()[0] if funded > 0 else "Medium"
    social_impact  = int(subset["Social_Impact_Score"].mean())
    failure_reasons = (subset["Failure_Reasons"].value_counts().head(3).index.tolist())
    top_investors  = subset["Lead_Investor"].value_counts().head(3).index.tolist()
    top_city       = subset["City_Location"].value_counts().index[0] if funded > 0 else "Bangalore"
    avg_min_bud_cr = round(subset["Min_Budget_INR"].mean() / 1e7, 1)
    avg_max_bud_cr = round(subset["Max_Budget_INR"].mean() / 1e7, 1)
    avg_team       = int(subset["Ideal_Team_Size"].mean())

    similar_ideas = (
        subset[["Startup_Name","Idea_Type","City_Location","Investment_Stage",
                "Success_Rate","Sustainability_Score","Amount_INR",
                "Market_Demand_Score","Competition_Level","Growth_Rate_Pct",
                "Environmental_Impact","Social_Impact_Score","Technical_Complexity",
                "Execution_Difficulty","Risk_Level","Failure_Reasons",
                "Min_Budget_INR","Max_Budget_INR","Ideal_Team_Size",
                "Target_Market","Lead_Investor","Year","Domain"]]
        .sort_values("Success_Rate", ascending=False)
        .head(7)
        .to_dict("records")
    )

    idea_type_perf = (
        subset.groupby("Idea_Type")[["Success_Rate","Market_Demand_Score","Sustainability_Score"]]
        .mean()
        .sort_values("Success_Rate", ascending=False)
        .head(5)
        .reset_index()
        .to_dict("records")
    )

    return {
        "sustainability_score": sustain_sc,
        "market_demand_score":  mkt_demand,
        "feasibility_score":    feasibility_sc,
        "scalability_score":    scalability_sc,
        "competition_level":    competition,
        "growth_rate":          growth_rate,
        "success_rate":         success_rate,
        "risk_level":           risk_level,
        "social_impact_score":  social_impact,
        "funded_startups":      funded,
        "top_investors":        top_investors,
        "top_city":             top_city,
        "failure_reasons":      failure_reasons,
        "avg_min_budget_cr":    avg_min_bud_cr,
        "avg_max_budget_cr":    avg_max_bud_cr,
        "avg_team_size":        avg_team,
        "similar_startups":     similar_ideas,
        "top_idea_types":       idea_type_perf,
    }


DOMAIN_TO_KEYWORD = {
    "Technology / SaaS":        "Technology",
    "Finance / FinTech":        "FinTech",
    "Healthcare / MedTech":     "HealthTech",
    "Education / EdTech":       "EdTech",
    "Energy / CleanTech":       "CleanTech",
    "Retail / E-Commerce":      "E-Commerce",
    "Logistics / Supply Chain": "Logistics",
    "Agriculture / AgriTech":   "Agriculture",
    "Food & Beverage":          "Food",
    "Real Estate / PropTech":   "Real Estate",
    "Media & Entertainment":    "Media",
    "Travel & Tourism":         "Travel",
    "Manufacturing":            "Manufacturing",
    "Other":                    "",
}


def _nav(show_logout=False):
    user_html = ""
    if show_logout:
        user_html = f'<span style="color:var(--muted);font-size:.9rem;">👤 {st.session_state.user["name"]}</span>'
    st.markdown(
        f'<div class="navbar"><div class="logo">🚀 StartupIQ</div>{user_html}</div>',
        unsafe_allow_html=True,
    )
    if show_logout:
        col = st.columns([8, 1])[1]
        with col:
            if st.button("Logout", key="logout_btn"):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()


def _score_ring(value, label, color):
    return (f'<div class="score-ring" style="border-color:{color};">'
            f'<span class="value" style="color:{color};">{value}%</span>'
            f'<span class="label">{label}</span></div>')


def _badge(potential):
    cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(
        potential.lower(), "badge-medium")
    return f'<span class="potential-badge {cls}">{potential}</span>'


def _generate_report_text(r: dict, meta: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  Startup Idea Analysis Report")
    lines.append("=" * 70)
    lines.append(f"  Idea    : {meta.get('idea','')[:80]}")
    lines.append(f"  Domain  : {meta.get('domain','')}")
    lines.append(f"  Budget  : {meta.get('budget','')}")
    lines.append(f"  Market  : {meta.get('locality','')}")
    lines.append(f"  Team    : {meta.get('team_size','')} people")
    lines.append(f"  Date    : {datetime.now().strftime('%d %b %Y %H:%M')}")
    lines.append("=" * 70)
    lines.append(f"  SUCCESS PROBABILITY  : {r.get('success_probability', 0)}%")
    lines.append(f"  SUSTAINABILITY SCORE : {r.get('sustainability_score', 0)}%")
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    for ln in textwrap.wrap(r.get("summary", ""), 68):
        lines.append("  " + ln)
    lines.append("")
    lines.append("DETAILED ANALYSIS")
    lines.append("-" * 70)
    analysis = r.get("analysis", {})
    for k, label in [("market_demand", "Market Demand"), ("competition", "Competition"),
                     ("feasibility", "Feasibility"), ("scalability", "Scalability")]:
        lines.append(f"  [{label}]")
        for ln in textwrap.wrap(analysis.get(k, ""), 66):
            lines.append("    " + ln)
        lines.append("")
    lines.append("IMPROVEMENTS")
    lines.append("-" * 70)
    for i, imp in enumerate(r.get("improvements", []), 1):
        lines.append(f"  {i}. " + imp)
    lines.append("")
    lines.append("ALTERNATIVE BUSINESS IDEAS")
    lines.append("-" * 70)
    for idea in r.get("new_ideas", []):
        lines.append(f"  * {idea.get('title', '')} [{idea.get('market_potential', '')}]")
        for ln in textwrap.wrap(idea.get("description", ""), 64):
            lines.append("    " + ln)
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _download_report(r, meta):
    txt = _generate_report_text(r, meta)
    b64 = base64.b64encode(txt.encode()).decode()
    fname = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    href = (f'<a href="data:text/plain;base64,{b64}" download="{fname}" '
            f'style="display:inline-block;background:linear-gradient(135deg,#00E5A0,#3B82F6);'
            f'color:#000;font-weight:700;padding:.6rem 1.5rem;border-radius:50px;'
            f'text-decoration:none;font-family:\'Syne\',sans-serif;">📄 Download Report</a>')
    st.markdown(href, unsafe_allow_html=True)


def _chat_call(messages: list, api_key: str = "") -> str:
    import urllib.request as _ur

    key = api_key.strip() or API_KEY
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    payload = json.dumps({
        "model": "openrouter/auto",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 600,
    }).encode()

    req = _ur.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "StartupIQ Chat",
        },
        method="POST",
    )

    with _ur.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())

    return body["choices"][0]["message"]["content"].strip()

# ═══════════════════════════════════════════════════════════
#  HOME PAGE
# ═══════════════════════════════════════════════════════════

def page_home():
    _nav(False)

    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 1.5rem;">
      <div style="display:inline-block;background:rgba(0,229,160,.1);
           border:1px solid rgba(0,229,160,.3);border-radius:50px;
           padding:.4rem 1.2rem;font-size:.8rem;color:#00E5A0;
           letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.5rem;">
        India-Focused Startup Evaluation Platform
      </div>
      <h1 style="font-size:clamp(2rem,4.5vw,3.5rem);font-weight:800;line-height:1.15;margin-bottom:1rem;">
        Entrepreneurial Intelligence<br>
        <span style="background:linear-gradient(135deg,#00E5A0,#3B82F6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          for Sustainable Innovation
        </span>
      </h1>
      <p style="font-size:1.05rem;color:#9CA3AF;max-width:620px;margin:0 auto 2rem;line-height:1.8;">
        A decision-support platform that helps first-time entrepreneurs in India
        evaluate the viability of their business ideas before committing time and capital.
        Submit your idea, receive a structured evaluation with market data,
        competition analysis, and a sustainability assessment — all in seconds.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("🚀 Get Started", key="cta_reg", use_container_width=True):
            st.session_state.page = "register"; st.rerun()
    c4, c5, c6 = st.columns([2.4, 1, 2.4])
    with c5:
        if st.button("Login →", key="cta_login", use_container_width=True):
            st.session_state.page = "login"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;
         padding:2rem 2.5rem;margin-bottom:1.5rem;">
      <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.15rem;
           color:#00E5A0;margin-bottom:1rem;">About This Platform</div>
      <p style="color:#CBD5E1;line-height:1.9;font-size:.95rem;margin-bottom:1rem;">
        This platform was built to solve a real problem faced by beginner entrepreneurs in India —
        the lack of accessible, data-driven feedback before investing resources into a business idea.
        Most startup failures happen not because of poor execution, but because the idea was never
        validated against real market conditions.
      </p>
      <p style="color:#CBD5E1;line-height:1.9;font-size:.95rem;margin-bottom:1rem;">
        The platform takes five key inputs from the user — their target locality, budget range in
        Indian Rupees (₹), team size, business domain, and a description of their idea. It then sends
        these inputs to a large language model configured with the role of a venture capitalist,
        market analyst, and sustainability expert. The model returns a structured evaluation
        covering market demand, competition, feasibility, and long-term scalability.
      </p>
      <p style="color:#CBD5E1;line-height:1.9;font-size:.95rem;">
        Two primary scores are generated — a <strong style="color:#E8EAF0;">Success Probability</strong>
        (0–100%) reflecting the likelihood of the idea gaining traction in the market, and a
        <strong style="color:#E8EAF0;">Sustainability Score</strong> (0–100%) reflecting the long-term
        viability, environmental responsibility, and social impact of the business model.
        Alongside these scores, the platform provides three actionable improvements and
        seven alternative business ideas relevant to the same domain and budget range.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;
         padding:2rem 2.5rem;margin-bottom:1.5rem;">
      <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.15rem;
           color:#3B82F6;margin-bottom:1.2rem;">How It Works</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1.2rem;">
        <div style="text-align:center;">
          <div style="font-size:2rem;margin-bottom:.6rem;">📝</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;margin-bottom:.4rem;">Step 1 — Submit</div>
          <div style="color:#9CA3AF;font-size:.82rem;line-height:1.6;">
            Register and fill in your idea details — locality, budget, team size, domain, and description.
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:2rem;margin-bottom:.6rem;">⚙️</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;margin-bottom:.4rem;">Step 2 — Analyse</div>
          <div style="color:#9CA3AF;font-size:.82rem;line-height:1.6;">
            The platform sends your inputs to a language model acting as a business consultant
            specialised in the Indian market.
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:2rem;margin-bottom:.6rem;">📊</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;margin-bottom:.4rem;">Step 3 — Review</div>
          <div style="color:#9CA3AF;font-size:.82rem;line-height:1.6;">
            Receive a success score, sustainability score, market analysis, competitive landscape,
            and three improvement suggestions.
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:2rem;margin-bottom:.6rem;">💡</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;margin-bottom:.4rem;">Step 4 — Explore</div>
          <div style="color:#9CA3AF;font-size:.82rem;line-height:1.6;">
            Explore seven alternative business ideas in your domain, compare past analyses,
            and consult the built-in startup advisor.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-bottom:1.5rem;">
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem 2rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#F59E0B;margin-bottom:.9rem;">
          Scope of Evaluation
        </div>
        <div style="color:#CBD5E1;font-size:.88rem;line-height:2;">
          ✔ Market demand assessment for the Indian market<br>
          ✔ Competitive landscape analysis<br>
          ✔ Budget feasibility check in Indian Rupees (₹)<br>
          ✔ Team size vs. idea complexity assessment<br>
          ✔ Scalability potential across Indian states<br>
          ✔ Environmental and social sustainability rating<br>
          ✔ Three prioritised improvement recommendations<br>
          ✔ Seven ranked alternative startup ideas
        </div>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem 2rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#EF4444;margin-bottom:.9rem;">
          Who Should Use This
        </div>
        <div style="color:#CBD5E1;font-size:.88rem;line-height:2;">
          → First-time entrepreneurs exploring startup ideas<br>
          → Students working on business plan assignments<br>
          → Professionals considering a career shift to entrepreneurship<br>
          → Small business owners looking to expand into new domains<br>
          → Incubator mentors evaluating early-stage pitches<br>
          → Anyone in India wanting a quick market reality check<br>
          → Teams preparing for startup competitions<br>
          → Researchers studying Indian startup ecosystem trends
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;
         padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
           color:#00E5A0;margin-bottom:1rem;">Domains Covered</div>
      <div style="display:flex;flex-wrap:wrap;gap:.5rem;">
        <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#00E5A0;">Technology / SaaS</span>
        <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#3B82F6;">Healthcare / MedTech</span>
        <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#F59E0B;">Education / EdTech</span>
        <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#00E5A0;">Finance / FinTech</span>
        <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#3B82F6;">Agriculture / AgriTech</span>
        <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#F59E0B;">Energy / CleanTech</span>
        <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#00E5A0;">Retail / E-Commerce</span>
        <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#3B82F6;">Logistics / Supply Chain</span>
        <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#F59E0B;">Food & Beverage</span>
        <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#00E5A0;">Real Estate / PropTech</span>
        <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#3B82F6;">Media & Entertainment</span>
        <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#F59E0B;">Travel & Tourism</span>
        <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);border-radius:20px;padding:.3rem .9rem;font-size:.82rem;color:#00E5A0;">Manufacturing</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  REGISTER / LOGIN
# ═══════════════════════════════════════════════════════════

def page_register():
    _nav(False)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<h2 style="text-align:center;margin-bottom:1.5rem;">Create Account</h2>',
                    unsafe_allow_html=True)
        with st.form("reg_form"):
            name  = st.text_input("Full Name", placeholder="Rahul Sharma")
            email = st.text_input("Email", placeholder="rahul@example.com")
            pwd   = st.text_input("Password", type="password", placeholder="Min 6 characters")
            pwd2  = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            sub   = st.form_submit_button("Create Account", use_container_width=True)
        if sub:
            if not all([name, email, pwd, pwd2]):
                st.markdown('<div class="alert-error">⚠️ All fields are required.</div>', unsafe_allow_html=True)
            elif pwd != pwd2:
                st.markdown('<div class="alert-error">⚠️ Passwords do not match.</div>', unsafe_allow_html=True)
            elif len(pwd) < 6:
                st.markdown('<div class="alert-error">⚠️ Password must be at least 6 characters.</div>', unsafe_allow_html=True)
            else:
                r = register_user(name, email, pwd)
                if r["success"]:
                    st.markdown(f'<div class="alert-success">✅ {r["message"]}</div>', unsafe_allow_html=True)
                    st.session_state.page = "login"; st.rerun()
                else:
                    st.markdown(f'<div class="alert-error">⚠️ {r["message"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Already have an account? Login →", key="go_login"):
            st.session_state.page = "login"; st.rerun()


def page_login():
    _nav(False)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<h2 style="text-align:center;margin-bottom:1.5rem;">Welcome Back</h2>',
                    unsafe_allow_html=True)
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="rahul@example.com")
            pwd   = st.text_input("Password", type="password", placeholder="••••••••")
            sub   = st.form_submit_button("Login", use_container_width=True)
        if sub:
            if not all([email, pwd]):
                st.markdown('<div class="alert-error">⚠️ Enter email and password.</div>', unsafe_allow_html=True)
            else:
                r = login_user(email, pwd)
                if r["success"]:
                    st.session_state.user = r["user"]
                    st.session_state.page = "dashboard"; st.rerun()
                else:
                    st.markdown(f'<div class="alert-error">⚠️ {r["message"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("New here? Create an account →", key="go_register"):
            st.session_state.page = "register"; st.rerun()


# ═══════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════

def page_dashboard():
    _nav(True)
    tabs = {
        "analyze":   "⚡ Analyse",
        "history":   "🕐 History",
        "compare":   "⚔️ Compare",
        "leaderboard":"🏆 Leaderboard",
        "dataset":   "📂 Dataset Analysis",
        "avatar":    "🤖 Advisor",
    }
    cols = st.columns(len(tabs))
    for i, (key, label) in enumerate(tabs.items()):
        with cols[i]:
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                st.session_state.active_tab = key; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    tab = st.session_state.active_tab
    if tab == "analyze":    _tab_analyze()
    elif tab == "history":  _tab_history()
    elif tab == "compare":  _tab_compare()
    elif tab == "leaderboard": _tab_leaderboard()
    elif tab == "dataset":  _tab_dataset()
    elif tab == "avatar":   _tab_avatar()


# ─────────────────────────────────────────────────────────
#  TAB: ANALYSE
# ─────────────────────────────────────────────────────────

def _tab_analyze():
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
      <h3 style="margin-bottom:.4rem;">⚡ Startup Intelligence</h3>
      <p style="color:#9CA3AF;font-size:.88rem;margin:0;">
        No idea yet? Generate one. Already have an idea? Analyse it.
      </p>
    </div>""", unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("💡  Generate Ideas — I don't have an idea yet",
                     key="btn_mode_gen", use_container_width=True):
            st.session_state["_analyze_mode"] = "generate"
            st.session_state.analysis_result  = None
            st.session_state.generate_result  = None
            st.rerun()
    with mc2:
        if st.button("🔍  Analyse My Idea — I already have an idea",
                     key="btn_mode_ana", use_container_width=True):
            st.session_state["_analyze_mode"] = "analyze"
            st.session_state.analysis_result  = None
            st.session_state.generate_result  = None
            st.rerun()

    mode = st.session_state.get("_analyze_mode", "analyze")

    if mode == "generate":
        st.markdown("""
        <div style="background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.25);
             border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem;">
          <strong style="color:#00E5A0;">💡 Idea Generator</strong>
          <div style="color:#9CA3AF;font-size:.86rem;margin-top:.3rem;">
            Tell us your location, domain, and budget. The AI will generate
            3 startup ideas perfectly suited to your profile — each with a clear
            step-by-step plan to launch it.
          </div>
        </div>""", unsafe_allow_html=True)

        with st.form("generate_form"):
            gc1, gc2 = st.columns(2)
            with gc1:
                g_locality  = st.text_input("🌍 Your City / Target Market",
                                             placeholder="e.g. Hyderabad  or  Pan-India")
                g_budget    = st.selectbox("💰 Budget (₹ INR)", BUDGET_OPTIONS, key="g_budget")
                g_team      = st.number_input("👥 Team Size", min_value=1, max_value=500,
                                              value=1, key="g_team")
            with gc2:
                g_domain  = st.selectbox("🏭 Domain of Interest", DOMAINS, key="g_domain")
                g_context = st.text_area(
                    "📝 Any context about yourself (optional)",
                    height=150, key="g_context",
                    placeholder=(
                        "e.g. I am a fresh engineering graduate interested in agriculture. "
                        "I know Python. I want to solve problems for farmers in my district "
                        "and prefer not to raise external funding initially."
                    ),
                )
            gen_submitted = st.form_submit_button(
                "💡  Generate My Startup Ideas", use_container_width=True)

        if gen_submitted:
            if not g_locality.strip():
                st.markdown('<div class="alert-error">⚠️ Please enter your city or target market.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner("Generating startup ideas for you — please wait…"):
                    try:
                        _df  = load_dataset()
                        _kw  = DOMAIN_TO_KEYWORD.get(g_domain, "")
                        _ds  = get_domain_stats(_df, _kw) if _kw else {}
                        result = generate_ideas(
                            locality=g_locality, budget=g_budget,
                            team_size=int(g_team), domain=g_domain,
                            context=g_context, api_key="",
                            dataset_scores=_ds,
                        )
                        st.session_state.generate_result = result
                        st.session_state.generate_meta   = {
                            "locality": g_locality, "budget": g_budget,
                            "team_size": g_team, "domain": g_domain,
                        }
                        st.session_state["_gen_ds"] = _ds
                        st.rerun()
                    except Exception as exc:
                        st.markdown(f'<div class="alert-error">⚠️ Generation failed: {exc}</div>',
                                    unsafe_allow_html=True)

        if st.session_state.generate_result:
            _render_generate_results(
                st.session_state.generate_result,
                st.session_state.generate_meta,
                st.session_state.get("_gen_ds", {}),
            )

    else:
        st.markdown("""
        <div style="background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.25);
             border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem;">
          <strong style="color:#3B82F6;">🔍 Idea Analyser</strong>
          <div style="color:#9CA3AF;font-size:.86rem;margin-top:.3rem;">
            Describe your startup idea in detail. The AI will score it using real dataset
            benchmarks, identify weaknesses, suggest improvements, and give you a
            step-by-step launch plan.
          </div>
        </div>""", unsafe_allow_html=True)

        with st.form("idea_form"):
            col1, col2 = st.columns(2)
            with col1:
                locality  = st.text_input("🌍 Target Market / Locality",
                                          placeholder="e.g. Hyderabad, Telangana  or  Pan-India")
                budget    = st.selectbox("💰 Budget (₹ INR)", BUDGET_OPTIONS)
                team_size = st.number_input("👥 Team Size", min_value=1, max_value=500, value=3)
            with col2:
                domain    = st.selectbox("🏭 Domain", DOMAINS)
                idea      = st.text_area("💡 Describe Your Idea", height=150,
                                         placeholder="What problem does it solve? Who are the customers? How does it make money?")
                idea_title = st.text_input("📝 Short Title (for history)",
                                           placeholder="e.g. AgriTech Drone Delivery")
            submitted = st.form_submit_button("⚡  Analyse Idea", use_container_width=True)

        if submitted:
            if not locality.strip() or not idea.strip():
                st.markdown('<div class="alert-error">⚠️ Please fill in Locality and Idea Description.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner("Analysing your idea — please wait…"):
                    try:
                        _df_pre = load_dataset()
                        _kw_pre = DOMAIN_TO_KEYWORD.get(domain, "")
                        _ds_pre = get_domain_stats(_df_pre, _kw_pre) if _kw_pre else {}
                        result  = analyze_idea(locality, budget, int(team_size), domain, idea,
                                               api_key="", dataset_scores=_ds_pre)
                        title   = idea_title.strip() or idea[:40] + "..."
                        aid     = save_analysis(
                            user_id=st.session_state.user["id"],
                            locality=locality, budget=budget,
                            team_size=int(team_size), domain=domain,
                            idea=idea, result=json.dumps(result), title=title,
                        )
                        st.session_state.analysis_result = result
                        st.session_state.analysis_meta   = {
                            "locality": locality, "budget": budget,
                            "team_size": team_size, "domain": domain,
                            "idea": idea, "title": title,
                        }
                        st.session_state.current_analysis_id = aid
                        st.rerun()
                    except Exception as exc:
                        st.markdown(f'<div class="alert-error">⚠️ Analysis failed: {exc}</div>',
                                    unsafe_allow_html=True)

        if st.session_state.analysis_result:
            _render_results(st.session_state.analysis_result, st.session_state.analysis_meta)


# ─────────────────────────────────────────────────────────
#  IDEA DESCRIPTION HELPER
# ─────────────────────────────────────────────────────────

def _idea_description(idea_type, domain, target_market, competition, growth_rate,
                       success_rate, sustainability_score, market_demand_score,
                       env_impact, social_impact, tech_complexity, exec_difficulty,
                       risk_level, failure_reason, min_budget_inr, max_budget_inr,
                       team_size):
    if success_rate >= 35:
        sc_txt = f"a strong {success_rate}% success rate — meaning roughly 1 in 3 startups of this type succeed"
    elif success_rate >= 25:
        sc_txt = f"a moderate {success_rate}% success rate — roughly 1 in 4 startups of this type succeed"
    else:
        sc_txt = f"a below-average {success_rate}% success rate — only 1 in 5 startups of this type succeed"

    if sustainability_score >= 75:
        su_txt = f"excellent long-term sustainability ({sustainability_score}/100)"
    elif sustainability_score >= 60:
        su_txt = f"moderate long-term sustainability ({sustainability_score}/100)"
    else:
        su_txt = f"limited long-term sustainability ({sustainability_score}/100) — needs attention"

    if market_demand_score >= 75:
        md_txt = f"strong market demand ({market_demand_score}/100) — real investor interest exists"
    elif market_demand_score >= 55:
        md_txt = f"moderate market demand ({market_demand_score}/100) — growing but not yet mainstream"
    else:
        md_txt = f"low current market demand ({market_demand_score}/100) — early-stage opportunity"

    # ── Budget in ₹ INR ──────────────────────────────────────────────────────
    min_str = _fmt_inr(min_budget_inr)
    max_str = _fmt_inr(max_budget_inr)
    max_cr  = max_budget_inr / 1e7

    comp_txt = {
        "High":   "highly competitive — many established players exist, differentiation is critical",
        "Medium": "moderately competitive — room to enter with a clear niche",
        "Low":    "low competition — first-mover advantage is available",
    }.get(competition, "moderate competition")

    env_txt = {
        "Low":    "low environmental footprint — positive for sustainability scoring",
        "Medium": "moderate environmental impact — manageable with responsible practices",
        "High":   "high environmental impact — requires strong sustainability measures",
    }.get(env_impact, "moderate environmental impact")

    tech_txt = {
        "Low":    "low technical complexity — can be built without deep engineering expertise",
        "Medium": "medium technical complexity — requires a capable tech team",
        "High":   "high technical complexity — needs experienced engineers or AI/ML specialists",
    }.get(tech_complexity, "moderate technical complexity")

    risk_txt = {
        "Low":    "low risk — stable business model with predictable returns",
        "Medium": "medium risk — manageable with proper planning and execution",
        "High":   "high risk — requires strong execution, capital buffer, and contingency plans",
    }.get(risk_level, "medium risk")

    description = f"""
**What is this idea?**
{idea_type} is a business concept in the **{domain}** domain targeting the **{target_market}** market in India.
It involves building a product or service around {idea_type.lower()}, designed to serve customers in this segment.

**How successful is this idea?**
Based on real data from similar startups, this idea type has {sc_txt}.
The market shows {md_txt}, and the domain is growing at **{growth_rate}% YoY**.

**Is it sustainable long-term?**
This idea has {su_txt}.
Environmental impact is **{env_impact.lower()}** — {env_txt}.
Social impact scores at **{social_impact}/100**, indicating {"strong" if social_impact >= 75 else "moderate" if social_impact >= 55 else "limited"} positive community contribution.

**Is it feasible to build?**
This idea has {tech_txt}.
Execution difficulty is rated at **{exec_difficulty}/100** — {"straightforward to execute" if exec_difficulty <= 45 else "moderately challenging" if exec_difficulty <= 65 else "quite challenging to execute"}.
A team of **{team_size} people** is typically needed to run this type of startup.

**What does it cost? (₹ INR)**
Typical startup budget ranges from **{min_str}** (minimum) to **{max_str}** (to scale).
This puts it in the {"bootstrapped" if max_cr <= 0.5 else "seed-stage" if max_cr <= 3 else "growth-stage"} funding range.

**What is the competition like?**
The market is {comp_txt}.
Risk level is **{risk_txt}**.

**What causes startups like this to fail?**
The most common failure reason for this type of idea is: **{failure_reason}**.
Founders should proactively address this risk before scaling.
    """.strip()

    return description


# ─────────────────────────────────────────────────────────
#  GENERATE RESULTS RENDERER
# ─────────────────────────────────────────────────────────

_GEN_COLORS  = ["#00E5A0", "#3B82F6", "#F59E0B", "#A78BFA", "#F472B6"]
_STEP_COLORS = ["#00E5A0", "#3B82F6", "#F59E0B", "#A78BFA", "#F472B6", "#34D399"]


def _render_generate_results(r: dict, meta: dict, ds: dict):
    st.markdown("<hr style='border-color:var(--border);margin:2rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="alert-success">✅ Here are your personalised startup ideas!</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if r.get("summary"):
        st.markdown(
            f'<div class="card"><div class="section-title">💬 Why These Ideas Suit You</div>'
            f'<p style="color:#CBD5E1;line-height:1.7;">{r["summary"]}</p></div>',
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    ideas = r.get("ideas", [])
    if not ideas:
        st.warning("No ideas returned — please try again.")
        return

    if ds:
        st.markdown(
            f'<div style="background:rgba(0,229,160,.05);border:1px solid rgba(0,229,160,.2);'
            f'border-radius:10px;padding:.7rem 1.2rem;margin-bottom:1rem;font-size:.83rem;color:#9CA3AF;">'
            f'📂 Ideas benchmarked against <strong style="color:#CBD5E1;">{ds.get("funded_startups",0)}'
            f' real {meta.get("domain","")} startups</strong> from the dataset · '
            f'Domain growth rate: <strong style="color:#00E5A0;">{ds.get("growth_rate",0)}% YoY</strong> · '
            f'Competition: <strong style="color:#F59E0B;">{ds.get("competition_level","Medium")}</strong>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 Startup Ideas For You</div>', unsafe_allow_html=True)

    for idea in ideas:
        rank   = idea.get("rank", 1)
        color  = _GEN_COLORS[(rank - 1) % len(_GEN_COLORS)]
        pot    = idea.get("market_potential", "Medium")
        sp     = idea.get("success_probability", 0)
        ss     = idea.get("sustainability_score", 0)
        sp_col = "#00E5A0" if sp >= 60 else "#F59E0B" if sp >= 40 else "#EF4444"
        ss_col = "#00E5A0" if ss >= 60 else "#F59E0B" if ss >= 40 else "#EF4444"
        label  = "⭐ Best Pick" if rank == 1 else f"#{rank}"

        with st.expander(f"{label}  —  {idea.get('title','')}  ·  {pot} Potential",
                         expanded=(rank == 1)):
            ic1, ic2 = st.columns([3, 1])
            with ic1:
                st.markdown(f"""
                <div style="border-left:4px solid {color};padding-left:14px;margin-bottom:14px;">
                  <div style="font-size:1.1rem;font-weight:700;color:{color};margin-bottom:3px;">
                    {idea.get('title','')}
                  </div>
                  <div style="color:#9CA3AF;font-size:.9rem;font-style:italic;">
                    {idea.get('tagline','')}
                  </div>
                </div>""", unsafe_allow_html=True)

                for key, label_txt, border in [
                    ("problem",         "🔴 The Problem",      "#EF4444"),
                    ("solution",        "🟢 The Solution",     "#00E5A0"),
                    ("target_customer", "🔵 Who Is This For",  "#3B82F6"),
                    ("revenue_model",   "💰 How You Earn",     "#F59E0B"),
                    ("why_now",         "⚡ Why Now",          "#A78BFA"),
                ]:
                    val = idea.get(key, "")
                    if val:
                        st.markdown(f"""
                        <div style="border-left:3px solid {border};padding:8px 14px;
                                    background:rgba(255,255,255,.02);border-radius:0 8px 8px 0;
                                    margin-bottom:10px;">
                          <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                                      letter-spacing:.08em;color:#6B7280;margin-bottom:3px;">
                            {label_txt}
                          </div>
                          <div style="color:#CBD5E1;font-size:.9rem;line-height:1.65;">{val}</div>
                        </div>""", unsafe_allow_html=True)
            with ic2:
                st.markdown(
                    f'<div style="display:flex;flex-direction:column;align-items:center;'
                    f'gap:10px;padding-top:6px;">'
                    f'{_score_ring(sp,"Success",sp_col)}'
                    f'{_score_ring(ss,"Sustain",ss_col)}'
                    f'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    plan = r.get("best_idea_launch_plan", {})
    if plan:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">🗺️ Step-by-Step Launch Plan — {plan.get("title","")}</div>',
            unsafe_allow_html=True)

        info_parts = []
        if plan.get("team_needed"):
            info_parts.append(
                f'<div><strong style="color:#00E5A0;">👥 Team Needed:</strong> '
                f'<span style="color:#CBD5E1;">{plan["team_needed"]}</span></div>')
        if plan.get("biggest_risk"):
            info_parts.append(
                f'<div style="margin-top:.4rem;"><strong style="color:#EF4444;">⚠️ Biggest Risk:</strong> '
                f'<span style="color:#CBD5E1;">{plan["biggest_risk"]}</span></div>')
        if plan.get("first_milestone"):
            info_parts.append(
                f'<div style="margin-top:.4rem;"><strong style="color:#F59E0B;">🎯 90-Day Goal:</strong> '
                f'<span style="color:#CBD5E1;">{plan["first_milestone"]}</span></div>')
        if info_parts:
            st.markdown(
                f'<div class="card" style="margin-bottom:1rem;">{"".join(info_parts)}</div>',
                unsafe_allow_html=True)

        for i, step in enumerate(plan.get("steps", [])):
            sc = _STEP_COLORS[i % len(_STEP_COLORS)]
            st.markdown(f"""
            <div style="display:flex;gap:14px;background:var(--surface);
                        border:1px solid var(--border);border-radius:12px;
                        padding:14px 16px;margin-bottom:9px;">
              <div style="min-width:40px;height:40px;border-radius:50%;background:{sc}18;
                          border:2px solid {sc};display:flex;align-items:center;
                          justify-content:center;font-family:'Syne',sans-serif;
                          font-weight:800;font-size:1rem;color:{sc};flex-shrink:0;">
                {step.get('step', i+1)}
              </div>
              <div style="flex:1;">
                <div style="font-weight:700;color:{sc};margin-bottom:5px;font-size:.92rem;">
                  {step.get('title','')}
                </div>
                <div style="color:#CBD5E1;font-size:.87rem;line-height:1.6;margin-bottom:4px;">
                  <strong>Action:</strong> {step.get('action','')}
                </div>
                <div style="color:#9CA3AF;font-size:.83rem;margin-bottom:3px;">
                  <strong>Outcome:</strong> {step.get('outcome','')}
                </div>
                <div style="color:#6B7280;font-size:.8rem;">
                  💰 Cost: {step.get('cost','—')}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start Over", key="clear_gen"):
        st.session_state.generate_result = None
        st.rerun()


# ─────────────────────────────────────────────────────────
#  DATASET ANALYSIS — shown after every analysis
# ─────────────────────────────────────────────────────────

def _render_dataset_analysis(domain: str, ai_sp: int, ai_ss: int, ds: dict):
    if not ds:
        return

    st.markdown("<hr style='border-color:var(--border);margin:2.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        '''<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.3rem;">
          <div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;">📂 Dataset Analysis</div>
          <div style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);
               border-radius:20px;padding:.2rem .8rem;font-size:.75rem;color:#00E5A0;font-weight:600;">
            startup_intelligence_dataset.csv · 500 records · 2024–2026 · ₹ INR
          </div>
        </div>''',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1.5rem;">'
        f'Real data from <strong style="color:#CBD5E1;">{ds["funded_startups"]} actual startups'
        f'</strong> in the <strong style="color:#CBD5E1;">{domain}</strong> domain — '
        f'used to benchmark and validate the AI analysis above. All monetary values in ₹ INR.</div>',
        unsafe_allow_html=True)

    # ── Score Cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Dataset-Calculated Scores</div>', unsafe_allow_html=True)

    def _score_bar(label, score, formula, interpretation, what_it_means, color):
        pct     = min(score, 100)
        verdict = "🟢 Strong" if score >= 70 else "🟡 Moderate" if score >= 45 else "🔴 Weak"
        st.markdown(
            f'''<div style="background:var(--surface);border:1px solid var(--border);
                 border-radius:12px;padding:1rem 1.2rem;margin-bottom:.9rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
                <span style="color:#CBD5E1;font-size:.95rem;font-weight:600;">{label}</span>
                <div style="display:flex;align-items:center;gap:.6rem;">
                  <span style="font-size:.78rem;color:#9CA3AF;">{verdict}</span>
                  <span style="color:{color};font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;">{score}/100</span>
                </div>
              </div>
              <div style="background:var(--border);border-radius:6px;height:10px;overflow:hidden;margin-bottom:.75rem;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:6px;"></div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;font-size:.8rem;">
                <div style="background:rgba(255,255,255,.03);border-radius:8px;padding:.5rem .7rem;">
                  <div style="color:#6B7280;font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.2rem;">📐 Formula</div>
                  <div style="color:#9CA3AF;line-height:1.5;">{formula}</div>
                </div>
                <div style="background:rgba(255,255,255,.03);border-radius:8px;padding:.5rem .7rem;">
                  <div style="color:#6B7280;font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.2rem;">💡 What It Means</div>
                  <div style="color:#CBD5E1;line-height:1.5;">{what_it_means}</div>
                </div>
              </div>
              <div style="margin-top:.5rem;font-size:.78rem;color:{color};font-style:italic;">{interpretation}</div>
            </div>''', unsafe_allow_html=True)

    mkt = ds["market_demand_score"]
    _score_bar("🛒 Market Demand Score", mkt,
        f"mean(Market_Demand_Score) across {ds['funded_startups']} {domain} rows",
        "✅ Strong real-world demand — investors actively fund this space." if mkt >= 70 else
        "⚠️ Moderate demand — growing but needs a clear differentiator." if mkt >= 45 else
        "❌ Low demand — market is nascent or oversaturated.",
        f"Avg market demand of all {domain} startups = {mkt}/100. Higher = proven investor confidence.",
        "#3B82F6")

    comp_lv  = ds.get("competition_level","Medium")
    comp_val = {"High":78,"Medium":52,"Low":28}.get(comp_lv,52)
    _score_bar("⚔️ Competition Level", comp_val,
        f"mode(Competition_Level) of {ds['funded_startups']} rows → '{comp_lv}' → High=78, Medium=52, Low=28",
        "🔴 Crowded market — strong USP essential." if comp_lv=="High" else
        "🟡 Moderate competition — viable with a clear niche." if comp_lv=="Medium" else
        "🟢 Low competition — first-mover advantage available.",
        f"Most {domain} startups face '{comp_lv}' competition. Lower score = easier market entry.",
        "#F59E0B")

    feas = ds["feasibility_score"]
    _score_bar("🔧 Feasibility Score", feas,
        f"100 − mean(Execution_Difficulty) = 100 − {100-feas} = {feas}",
        "✅ Highly feasible — low execution difficulty." if feas >= 75 else
        "⚠️ Moderate feasibility — needs careful planning." if feas >= 55 else
        "❌ Difficult to execute — high complexity.",
        f"Feasibility = inverse of execution difficulty. Avg team size needed: {ds['avg_team_size']} people.",
        "#00E5A0")

    scal = ds["scalability_score"]
    _score_bar("📈 Scalability Score", scal,
        f"(Series C/D/E/F rows ÷ {ds['funded_startups']}) × 100 × 2.5 (capped 100)",
        "✅ Highly scalable — late-stage funding proven." if scal >= 70 else
        "⚠️ Moderate scalability — growth capital available but limited." if scal >= 40 else
        "❌ Hard to scale — few late-stage exits in dataset.",
        f"Measures % of {domain} startups that reached Series C+. Growth rate: {ds['growth_rate']}% YoY.",
        "#8B5CF6")

    sust = ds["sustainability_score"]
    _score_bar("🌱 Sustainability Score", sust,
        f"mean(Sustainability_Score) of {ds['funded_startups']} {domain} rows",
        "✅ Strong long-term sustainability." if sust >= 75 else
        "⚠️ Moderate sustainability — manageable with effort." if sust >= 55 else
        "❌ Sustainability concerns — environmental or social risks.",
        f"Avg sustainability of real {domain} startups = {sust}/100. Social impact: {ds['social_impact_score']}/100.",
        "#10B981")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Similar Startups ─────────────────────────────────────────────────────
    top_s = ds.get("similar_startups", [])
    if top_s:
        st.markdown('<div class="section-title">🏆 Similar Startups (From Dataset)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#9CA3AF;font-size:.8rem;margin-bottom:.8rem;">'
            f'Top 6 highest-success startups from {domain} domain — all funding in ₹ INR.</div>',
            unsafe_allow_html=True)

        for s in top_s[:6]:
            amt_inr  = s.get("Amount_INR", 0)
            amt_str  = _fmt_inr(amt_inr) if amt_inr > 0 else "Undisclosed"
            sc       = s.get("Success_Rate", 0)
            su       = s.get("Sustainability_Score", 0)
            mkt_s    = s.get("Market_Demand_Score", 0)
            it       = s.get("Idea_Type", "")
            city     = s.get("City_Location", "")
            stage    = s.get("Investment_Stage", "")
            sc_color = "#00E5A0" if sc >= 35 else "#F59E0B" if sc >= 25 else "#EF4444"
            su_color = "#00E5A0" if su >= 75 else "#F59E0B" if su >= 60 else "#EF4444"
            icon     = "🟢" if sc >= 35 else "🟡" if sc >= 25 else "🔴"

            with st.expander(f"{icon} {s['Startup_Name']}  ·  {it}  ·  {sc}% success  ·  {su} sustain  ·  {city}"):
                st.markdown(
                    f'''<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.1rem;">
                      <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);
                        border-radius:20px;padding:.3rem .9rem;font-size:.82rem;font-weight:700;color:{sc_color};">
                        ✅ {sc}% Success Rate</span>
                      <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);
                        border-radius:20px;padding:.3rem .9rem;font-size:.82rem;font-weight:700;color:{su_color};">
                        🌱 {su} Sustainability</span>
                      <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);
                        border-radius:20px;padding:.3rem .9rem;font-size:.82rem;font-weight:700;color:#F59E0B;">
                        📊 {mkt_s} Demand</span>
                      <span style="background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.3);
                        border-radius:20px;padding:.3rem .9rem;font-size:.82rem;font-weight:700;color:#8B5CF6;">
                        💰 {amt_str} · {stage}</span>
                    </div>''', unsafe_allow_html=True)

                desc = _idea_description(
                    idea_type=it, domain=s.get("Domain",""), target_market=s.get("Target_Market",""),
                    competition=s.get("Competition_Level","Medium"), growth_rate=s.get("Growth_Rate_Pct",0),
                    success_rate=sc, sustainability_score=su, market_demand_score=mkt_s,
                    env_impact=s.get("Environmental_Impact","Medium"), social_impact=s.get("Social_Impact_Score",0),
                    tech_complexity=s.get("Technical_Complexity","Medium"), exec_difficulty=s.get("Execution_Difficulty",50),
                    risk_level=s.get("Risk_Level","Medium"), failure_reason=s.get("Failure_Reasons","Unknown"),
                    min_budget_inr=s.get("Min_Budget_INR",0), max_budget_inr=s.get("Max_Budget_INR",0),
                    team_size=s.get("Ideal_Team_Size",0))
                st.markdown(desc)

                st.markdown(
                    f'''<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;
                         margin-top:1rem;padding-top:.8rem;border-top:1px solid var(--border);">
                      <div style="text-align:center;">
                        <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">City</div>
                        <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{city}</div>
                      </div>
                      <div style="text-align:center;">
                        <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Stage</div>
                        <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{stage}</div>
                      </div>
                      <div style="text-align:center;">
                        <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Investor</div>
                        <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{s.get("Lead_Investor","")}</div>
                      </div>
                      <div style="text-align:center;">
                        <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Year</div>
                        <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{s.get("Year","")}</div>
                      </div>
                    </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Idea Types ────────────────────────────────────────────────────────
    top_ideas = ds.get("top_idea_types", [])
    if top_ideas:
        st.markdown('<div class="section-title">💡 Top Performing Idea Types (Dataset)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#9CA3AF;font-size:.8rem;margin-bottom:.8rem;">'
            f'Ranked by average success rate from real {domain} records.</div>',
            unsafe_allow_html=True)

        medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i, it in enumerate(top_ideas):
            sc_it  = int(it["Success_Rate"])
            md_it  = int(it["Market_Demand_Score"])
            su_it  = int(it["Sustainability_Score"])
            medal  = medals[i] if i < len(medals) else f"#{i+1}"
            sc_col = "#00E5A0" if sc_it >= 35 else "#F59E0B" if sc_it >= 25 else "#EF4444"

            with st.expander(f"{medal}  {it['Idea_Type']}  —  {sc_it}% success · {md_it} demand · {su_it} sustainability"):
                st.markdown(
                    f'''<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.1rem;">
                      <span style="background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.3);
                        border-radius:20px;padding:.25rem .8rem;font-size:.8rem;font-weight:700;color:{sc_col};">
                        ✅ {sc_it}% Avg Success Rate</span>
                      <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);
                        border-radius:20px;padding:.25rem .8rem;font-size:.8rem;font-weight:700;color:#3B82F6;">
                        📊 {md_it} Market Demand</span>
                      <span style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);
                        border-radius:20px;padding:.25rem .8rem;font-size:.8rem;font-weight:700;color:#10B981;">
                        🌱 {su_it} Sustainability</span>
                    </div>''', unsafe_allow_html=True)

                df_live = load_dataset()
                if df_live is not None:
                    idea_rows = df_live[df_live["Idea_Type"] == it["Idea_Type"]]
                    if not idea_rows.empty:
                        avg_comp   = idea_rows["Competition_Level"].mode()[0]
                        avg_growth = round(idea_rows["Growth_Rate_Pct"].mean(),1)
                        avg_env    = idea_rows["Environmental_Impact"].mode()[0]
                        avg_social = int(idea_rows["Social_Impact_Score"].mean())
                        avg_tech   = idea_rows["Technical_Complexity"].mode()[0]
                        avg_exec   = int(idea_rows["Execution_Difficulty"].mean())
                        avg_risk   = idea_rows["Risk_Level"].mode()[0]
                        top_fail   = idea_rows["Failure_Reasons"].value_counts().index[0]
                        avg_min    = int(idea_rows["Min_Budget_INR"].mean())
                        avg_max    = int(idea_rows["Max_Budget_INR"].mean())
                        avg_team   = int(idea_rows["Ideal_Team_Size"].mean())
                        top_mkt    = idea_rows["Target_Market"].mode()[0]
                        top_city   = idea_rows["City_Location"].mode()[0]
                        top_stage  = idea_rows["Investment_Stage"].mode()[0]
                        n          = len(idea_rows)

                        desc = _idea_description(
                            idea_type=it["Idea_Type"], domain=idea_rows["Domain"].iloc[0],
                            target_market=top_mkt, competition=avg_comp, growth_rate=avg_growth,
                            success_rate=sc_it, sustainability_score=su_it, market_demand_score=md_it,
                            env_impact=avg_env, social_impact=avg_social, tech_complexity=avg_tech,
                            exec_difficulty=avg_exec, risk_level=avg_risk, failure_reason=top_fail,
                            min_budget_inr=avg_min, max_budget_inr=avg_max, team_size=avg_team)
                        st.markdown(desc)

                        st.markdown(
                            f'''<div style="display:grid;grid-template-columns:repeat(5,1fr);
                                 gap:.5rem;margin-top:1rem;padding-top:.8rem;
                                 border-top:1px solid var(--border);">
                              <div style="text-align:center;">
                                <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Records</div>
                                <div style="font-size:.9rem;font-weight:700;color:#00E5A0;">{n}</div>
                              </div>
                              <div style="text-align:center;">
                                <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Top City</div>
                                <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{top_city}</div>
                              </div>
                              <div style="text-align:center;">
                                <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Typical Stage</div>
                                <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{top_stage}</div>
                              </div>
                              <div style="text-align:center;">
                                <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Avg Team</div>
                                <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{avg_team} ppl</div>
                              </div>
                              <div style="text-align:center;">
                                <div style="font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:.06em;">Top Market</div>
                                <div style="font-size:.85rem;font-weight:600;color:#CBD5E1;">{top_mkt}</div>
                              </div>
                            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── AI vs Dataset Comparison ──────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 AI Scores vs Dataset Scores — Comparison</div>',
                unsafe_allow_html=True)

    col_ai, col_ds = st.columns(2)
    with col_ai:
        ai_sc_col = "#00E5A0" if ai_sp >= 60 else "#F59E0B" if ai_sp >= 40 else "#EF4444"
        ai_su_col = "#00E5A0" if ai_ss >= 60 else "#F59E0B" if ai_ss >= 40 else "#EF4444"
        st.markdown(
            f'''<div style="background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.25);
                 border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
              <div style="font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;
                   color:#00E5A0;margin-bottom:.6rem;">🤖 AI-Generated Scores</div>
              <div style="font-size:.85rem;line-height:2.2;color:#CBD5E1;">
                <div style="display:flex;justify-content:space-between;">
                  <span>Success Probability</span>
                  <strong style="color:{ai_sc_col};">{ai_sp}%</strong>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Sustainability Score</span>
                  <strong style="color:{ai_su_col};">{ai_ss}%</strong>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Domain Growth Momentum</span>
                  <strong style="color:#9CA3AF;">{DOMAIN_TRENDS.get(domain,50)}</strong>
                </div>
              </div>
              <div style="font-size:.72rem;color:#6B7280;margin-top:.5rem;">
                Source: LLM reasoning on your specific idea inputs
              </div>
            </div>''', unsafe_allow_html=True)

    with col_ds:
        st.markdown(
            f'''<div style="background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.25);
                 border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
              <div style="font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;
                   color:#3B82F6;margin-bottom:.6rem;">📂 Dataset Benchmark Scores</div>
              <div style="font-size:.85rem;line-height:2.2;color:#CBD5E1;">
                <div style="display:flex;justify-content:space-between;">
                  <span>Market Demand</span><strong style="color:#3B82F6;">{ds["market_demand_score"]}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Feasibility</span><strong style="color:#3B82F6;">{ds["feasibility_score"]}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Scalability</span><strong style="color:#3B82F6;">{ds["scalability_score"]}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Sustainability</span><strong style="color:#3B82F6;">{ds["sustainability_score"]}</strong>
                </div>
              </div>
              <div style="font-size:.72rem;color:#6B7280;margin-top:.5rem;">
                Source: mean of {ds["funded_startups"]} real {domain} records
              </div>
            </div>''', unsafe_allow_html=True)

    gaps = [
        ("Success vs Market Demand",   ai_sp, ds["market_demand_score"]),
        ("Sustainability vs Dataset",  ai_ss, ds["sustainability_score"]),
        ("AI Success vs Feasibility",  ai_sp, ds["feasibility_score"]),
        ("AI Success vs Scalability",  ai_sp, ds["scalability_score"]),
    ]
    gap_rows = ""
    for gname, ai_val, ds_val in gaps:
        diff = ai_val - ds_val
        diff_color = "#00E5A0" if diff >= 0 else "#EF4444"
        if diff > 10:    diff_label = f"+{diff} — AI is optimistic"
        elif diff < -10: diff_label = f"{diff} — AI is pessimistic"
        else:            diff_label = f"{diff:+d} — Aligned"
        gap_rows += (
            f'<tr style="border-bottom:1px solid var(--border);">'
            f'<td style="padding:.6rem .5rem;color:#9CA3AF;font-size:.82rem;">{gname}</td>'
            f'<td style="padding:.6rem .5rem;text-align:center;font-weight:700;color:#00E5A0;">{ai_val}</td>'
            f'<td style="padding:.6rem .5rem;text-align:center;font-weight:700;color:#3B82F6;">{ds_val}</td>'
            f'<td style="padding:.6rem .5rem;text-align:center;font-weight:700;color:{diff_color};">{diff_label}</td>'
            f'</tr>'
        )
    st.markdown(
        f'''<div class="card" style="overflow:hidden;margin-bottom:1rem;">
          <div class="section-title" style="margin-bottom:.75rem;">📐 Gap Analysis — AI vs Real Data</div>
          <table style="width:100%;border-collapse:collapse;font-size:.82rem;">
            <thead>
              <tr style="background:var(--surface2);">
                <th style="padding:.5rem;text-align:left;color:#6B7280;font-weight:500;">Metric</th>
                <th style="padding:.5rem;text-align:center;color:#00E5A0;font-weight:500;">AI Score</th>
                <th style="padding:.5rem;text-align:center;color:#3B82F6;font-weight:500;">Dataset Avg</th>
                <th style="padding:.5rem;text-align:center;color:#9CA3AF;font-weight:500;">Gap</th>
              </tr>
            </thead>
            <tbody>{gap_rows}</tbody>
          </table>
        </div>''', unsafe_allow_html=True)

    chart_rows = {
        "🤖 AI Success":          ai_sp,
        "🤖 AI Sustainability":   ai_ss,
        "📂 Market Demand":       ds["market_demand_score"],
        "📂 Feasibility":         ds["feasibility_score"],
        "📂 Scalability":         ds["scalability_score"],
        "📂 Sustainability":      ds["sustainability_score"],
        "📈 Growth Momentum":     DOMAIN_TRENDS.get(domain, 50),
    }
    chart_df = pd.DataFrame({"Score (0–100)": chart_rows})
    st.bar_chart(chart_df, height=320)

    sp_gap = ai_sp - ds["market_demand_score"]
    ss_gap = ai_ss - ds["sustainability_score"]
    feas   = ds["feasibility_score"]
    scal   = ds["scalability_score"]

    if sp_gap > 15:
        sp_insight = (f"Your AI success score (<strong style='color:#00E5A0;'>{ai_sp}%</strong>) is "
                      f"<strong style='color:#F59E0B;'>+{sp_gap} points above</strong> the domain's "
                      f"real market demand benchmark ({ds['market_demand_score']}). Validate assumptions carefully.")
    elif sp_gap < -15:
        sp_insight = (f"Your AI success score (<strong style='color:#EF4444;'>{ai_sp}%</strong>) is "
                      f"<strong>{abs(sp_gap)} points below</strong> the domain's market demand benchmark "
                      f"({ds['market_demand_score']}). Review improvements suggested above.")
    else:
        sp_insight = (f"Your AI success score (<strong style='color:#00E5A0;'>{ai_sp}%</strong>) is "
                      f"closely aligned with the domain's market demand benchmark ({ds['market_demand_score']}) — gap of only {sp_gap:+d} points.")

    if ss_gap > 10:
        ss_insight = f"AI sustainability ({ai_ss}%) is above dataset avg ({ds['sustainability_score']}) by +{ss_gap} — stronger long-term viability."
    elif ss_gap < -10:
        ss_insight = f"AI sustainability ({ai_ss}%) is below dataset avg ({ds['sustainability_score']}) by {ss_gap} — address environmental impact."
    else:
        ss_insight = f"Sustainability scores aligned (AI: {ai_ss}%, Dataset: {ds['sustainability_score']}) — matches domain profile."

    feas_insight = (f"Feasibility is strong ({feas}/100)." if feas >= 70 else
                    f"Feasibility is moderate ({feas}/100) — plan carefully." if feas >= 50 else
                    f"Feasibility is low ({feas}/100) — budget extra runway.")
    scal_insight = (f"Scalability proven ({scal}/100) — late-stage funding exists." if scal >= 70 else
                    f"Scalability moderate ({scal}/100) — growth possible." if scal >= 40 else
                    f"Scalability limited ({scal}/100) — focus on sustainable small-scale growth first.")

    st.markdown(
        f'''<div style="background:var(--surface);border:1px solid var(--border);
             border-radius:14px;padding:1.2rem 1.4rem;margin-top:.8rem;">
          <div style="font-family:Syne,sans-serif;font-weight:700;font-size:.95rem;
               color:#CBD5E1;margin-bottom:.9rem;">📝 Chart Analysis</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;">
            <div style="border-left:3px solid #00E5A0;padding:.5rem .8rem;background:rgba(0,229,160,.04);border-radius:0 8px 8px 0;">
              <div style="font-size:.7rem;color:#00E5A0;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem;">Success Score</div>
              <div style="font-size:.83rem;color:#CBD5E1;line-height:1.7;">{sp_insight}</div>
            </div>
            <div style="border-left:3px solid #10B981;padding:.5rem .8rem;background:rgba(16,185,129,.04);border-radius:0 8px 8px 0;">
              <div style="font-size:.7rem;color:#10B981;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem;">Sustainability</div>
              <div style="font-size:.83rem;color:#CBD5E1;line-height:1.7;">{ss_insight}</div>
            </div>
            <div style="border-left:3px solid #00E5A0;padding:.5rem .8rem;background:rgba(0,229,160,.04);border-radius:0 8px 8px 0;">
              <div style="font-size:.7rem;color:#00E5A0;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem;">Feasibility</div>
              <div style="font-size:.83rem;color:#CBD5E1;line-height:1.7;">{feas_insight}</div>
            </div>
            <div style="border-left:3px solid #8B5CF6;padding:.5rem .8rem;background:rgba(139,92,246,.04);border-radius:0 8px 8px 0;">
              <div style="font-size:.7rem;color:#8B5CF6;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem;">Scalability</div>
              <div style="font-size:.83rem;color:#CBD5E1;line-height:1.7;">{scal_insight}</div>
            </div>
          </div>
          <div style="margin-top:.9rem;padding-top:.8rem;border-top:1px solid var(--border);font-size:.8rem;color:#9CA3AF;line-height:1.7;">
            Domain growth rate: <strong style="color:#F59E0B;">{ds["growth_rate"]}% YoY</strong> —
            {"expanding rapidly — a good time to enter." if ds["growth_rate"] >= 25 else "growing steadily — viable but competitive." if ds["growth_rate"] >= 15 else "growing slowly — differentiation is critical."}
          </div>
        </div>''', unsafe_allow_html=True)


def _render_results(r, meta={}):
    st.markdown("<hr style='border-color:var(--border);margin:2rem 0;'>", unsafe_allow_html=True)

    sp  = r.get("success_probability", 0)
    ss  = r.get("sustainability_score", 0)
    sp_c = "#00E5A0" if sp >= 60 else "#F59E0B" if sp >= 40 else "#EF4444"
    ss_c = "#00E5A0" if ss >= 60 else "#F59E0B" if ss >= 40 else "#EF4444"

    decision = r.get("decision", {})
    verdict  = decision.get("verdict", "").strip()
    dreason  = decision.get("reason", "")
    if verdict == "Proceed":
        dec_color, dec_icon, dec_bg = "#00E5A0", "✅", "rgba(0,229,160,.1)"
        dec_border = "rgba(0,229,160,.4)"
    elif verdict == "Improve":
        dec_color, dec_icon, dec_bg = "#F59E0B", "⚠️", "rgba(245,158,11,.1)"
        dec_border = "rgba(245,158,11,.4)"
    else:
        dec_color, dec_icon, dec_bg = "#EF4444", "❌", "rgba(239,68,68,.1)"
        dec_border = "rgba(239,68,68,.4)"
    if verdict:
        st.markdown(
            f'''<div style="background:{dec_bg};border:2px solid {dec_border};
                 border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1.5rem;
                 display:flex;align-items:center;gap:1rem;">
              <div style="font-size:2.2rem;">{dec_icon}</div>
              <div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;
                     font-size:1.4rem;color:{dec_color};">Final Decision: {verdict}</div>
                <div style="color:#CBD5E1;font-size:.92rem;margin-top:.3rem;">{dreason}</div>
              </div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div class="alert-success">✅ Analysis complete.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        st.markdown(f'<div style="text-align:center;">{_score_ring(sp,"Success",sp_c)}</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="text-align:center;">{_score_ring(ss,"Sustainability",ss_c)}</div>',
                    unsafe_allow_html=True)
    with c3:
        summary = r.get("summary", "")
        if summary:
            st.markdown(f'<div class="card"><div class="section-title">Overview</div>'
                        f'<p style="color:#CBD5E1;line-height:1.7;font-size:.95rem;">{summary}</p></div>',
                        unsafe_allow_html=True)

    expl = r.get("score_explanations", {})
    sr   = expl.get("success_reason", "")
    sur  = expl.get("sustainability_reason", "")
    if sr or sur:
        st.markdown("<br>", unsafe_allow_html=True)
        se_html = ''
        if sr:
            se_html += (f'<div style="display:flex;align-items:flex-start;gap:.75rem;'
                        f'background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.2);'
                        f'border-radius:10px;padding:.8rem 1rem;margin-bottom:.6rem;">'
                        f'<span style="font-size:1.1rem;">📊</span>'
                        f'<div><div style="font-size:.75rem;color:#3B82F6;font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:.2rem;">'
                        f'Why {sp}% success</div>'
                        f'<div style="color:#CBD5E1;font-size:.9rem;line-height:1.6;">{sr}</div>'
                        f'</div></div>')
        if sur:
            se_html += (f'<div style="display:flex;align-items:flex-start;gap:.75rem;'
                        f'background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);'
                        f'border-radius:10px;padding:.8rem 1rem;">'
                        f'<span style="font-size:1.1rem;">🌱</span>'
                        f'<div><div style="font-size:.75rem;color:#00E5A0;font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:.2rem;">'
                        f'Why {ss}% sustainability</div>'
                        f'<div style="color:#CBD5E1;font-size:.9rem;line-height:1.6;">{sur}</div>'
                        f'</div></div>')
        st.markdown(f'<div class="section-title">💬 Score Explanations</div>{se_html}',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ba, bb = st.columns(2)
    with ba:
        _download_report(r, meta)
    with bb:
        if st.button("🔄 New Analysis", key="clear_results"):
            st.session_state.analysis_result = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    analysis = r.get("analysis", {})
    if analysis:
        st.markdown('<div class="section-title">📋 Detailed Analysis</div>', unsafe_allow_html=True)
        rows_html = ""
        for k, label in [("market_demand","🛒 Market Demand"),("competition","⚔️ Competition"),
                          ("feasibility","🔧 Feasibility"),("scalability","📈 Scalability")]:
            val = analysis.get(k, "")
            if val:
                rows_html += (f'<div class="analysis-row"><div class="analysis-label">{label}</div>'
                              f'<div style="color:#CBD5E1;line-height:1.6;">{val}</div></div>')
        st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    improvements = r.get("improvements", [])
    if improvements:
        st.markdown('<div class="section-title">🔧 Suggested Improvements</div>', unsafe_allow_html=True)
        items_html = ""
        for i, imp in enumerate(improvements, 1):
            items_html += (f'<div class="improvement-item"><span class="num">0{i}</span>'
                           f'<span style="color:#CBD5E1;line-height:1.6;">{imp}</span></div>')
        st.markdown(items_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    new_ideas = r.get("new_ideas", [])
    if new_ideas:
        st.markdown('<div class="section-title">💡 Alternative Business Ideas</div>',
                    unsafe_allow_html=True)
        cards_html = '<div class="idea-grid">'
        for item in new_ideas:
            cards_html += (f'<div class="idea-card"><div class="idea-title">{item.get("title","")}</div>'
                           f'<div style="color:#9CA3AF;font-size:.88rem;line-height:1.5;margin-bottom:.75rem;">'
                           f'{item.get("description","")}</div>'
                           f'{_badge(item.get("market_potential","Medium"))}</div>')
        st.markdown(cards_html + "</div>", unsafe_allow_html=True)

    domain  = meta.get("domain", "")
    keyword = DOMAIN_TO_KEYWORD.get(domain, "")
    df_real = load_dataset()
    ds      = get_domain_stats(df_real, keyword) if keyword else {}
    _render_dataset_analysis(domain, sp, ss, ds)


# ─────────────────────────────────────────────────────────
#  TAB: HISTORY
# ─────────────────────────────────────────────────────────

def _tab_history():
    st.markdown('<h3 style="margin-bottom:1rem;">Analysis History</h3>', unsafe_allow_html=True)
    analyses = get_user_analyses(st.session_state.user["id"], limit=50)
    if not analyses:
        st.markdown('<div style="color:#9CA3AF;text-align:center;padding:3rem;">No analyses yet.</div>', unsafe_allow_html=True)
        return
    domains = sorted(set(a["domain"] for a in analyses if a.get("domain")))
    filter_domain = st.selectbox("Filter by Domain", ["All"] + domains, key="hist_filter")
    filtered = analyses if filter_domain == "All" else [a for a in analyses if a["domain"] == filter_domain]
    st.markdown(f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1rem;">{len(filtered)} result(s)</div>', unsafe_allow_html=True)
    for a in filtered:
        result = json.loads(a["result"]) if a.get("result") else {}
        sp = result.get("success_probability", 0)
        ss = result.get("sustainability_score", 0)
        sp_c = "#00E5A0" if sp >= 60 else "#F59E0B" if sp >= 40 else "#EF4444"
        ss_c = "#00E5A0" if ss >= 60 else "#F59E0B" if ss >= 40 else "#EF4444"
        title = a.get("title") or a.get("idea", "")[:40] + "..."
        date  = a.get("created_at", "")[:16]
        icon  = "🟢" if sp >= 60 else "🟡" if sp >= 40 else "🔴"
        with st.expander(f"{icon} {title} — {sp}% success"):
            hc1, hc2, hc3 = st.columns([1, 1, 3])
            with hc1:
                st.markdown(f'<div style="text-align:center;">{_score_ring(sp,"Success",sp_c)}</div>', unsafe_allow_html=True)
            with hc2:
                st.markdown(f'<div style="text-align:center;">{_score_ring(ss,"Sustain",ss_c)}</div>', unsafe_allow_html=True)
            with hc3:
                st.markdown(f'<div style="color:#9CA3AF;font-size:.85rem;line-height:1.9;">📅 {date}<br>🌍 {a.get("locality","")} &nbsp;|&nbsp; 💰 {a.get("budget","")} &nbsp;|&nbsp; 👥 {a.get("team_size","")} people<br>🏭 {a.get("domain","")}</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("📄 View Full", key=f"view_{a['id']}"):
                    st.session_state.analysis_result = result
                    st.session_state.analysis_meta = {k: a.get(k, "") for k in ["locality","budget","team_size","domain","idea","title"]}
                    st.session_state.active_tab = "analyze"; st.rerun()
            with b2:
                _download_report(result, {k: a.get(k, "") for k in ["locality","budget","team_size","domain","idea","title"]})
            with b3:
                if st.button("🗑️ Delete", key=f"del_{a['id']}"):
                    delete_analysis(a["id"], st.session_state.user["id"]); st.rerun()


# ─────────────────────────────────────────────────────────
#  TAB: COMPARE
# ─────────────────────────────────────────────────────────

def _tab_compare():
    st.markdown('<h3 style="margin-bottom:1rem;">Compare Two Ideas</h3>', unsafe_allow_html=True)
    analyses = get_user_analyses(st.session_state.user["id"], limit=50)
    if len(analyses) < 2:
        st.markdown('<div style="color:#9CA3AF;text-align:center;padding:3rem;">You need at least 2 saved analyses to compare.</div>', unsafe_allow_html=True)
        return
    options = {f"{a.get('title') or a['idea'][:35]}... ({a.get('domain','')})": a for a in analyses}
    labels  = list(options.keys())
    cc1, cc2 = st.columns(2)
    with cc1: sel_a = st.selectbox("Idea A", labels, key="cmp_a")
    with cc2: sel_b = st.selectbox("Idea B", labels, index=min(1, len(labels)-1), key="cmp_b")
    if sel_a == sel_b:
        st.markdown('<div class="alert-error">⚠️ Please select two different ideas.</div>', unsafe_allow_html=True)
        return
    a_data = options[sel_a]; b_data = options[sel_b]
    r_a = json.loads(a_data["result"]) if a_data.get("result") else {}
    r_b = json.loads(b_data["result"]) if b_data.get("result") else {}
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_mid, col_b = st.columns([5, 1, 5])
    def _comp_card(r, data, label):
        sp = r.get("success_probability", 0); ss = r.get("sustainability_score", 0)
        sp_c = "#00E5A0" if sp >= 60 else "#F59E0B" if sp >= 40 else "#EF4444"
        ss_c = "#00E5A0" if ss >= 60 else "#F59E0B" if ss >= 40 else "#EF4444"
        title = data.get("title") or data.get("idea", "")[:40]
        st.markdown(f'<div class="comp-col"><div style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:1rem;margin-bottom:1rem;color:#00E5A0;">{label}</div><div style="font-size:.9rem;color:#CBD5E1;margin-bottom:1rem;">{title}</div><div style="display:flex;gap:1rem;justify-content:center;margin-bottom:1.2rem;">{_score_ring(sp,"Success",sp_c)}{_score_ring(ss,"Sustain",ss_c)}</div><div style="font-size:.82rem;color:#9CA3AF;line-height:1.9;">🌍 {data.get("locality","")}<br>💰 {data.get("budget","")}<br>🏭 {data.get("domain","")}<br>👥 {data.get("team_size","")} people</div><hr style="border-color:var(--border);margin:1rem 0;"><div style="font-size:.82rem;color:#9CA3AF;"><b style="color:#CBD5E1;">Market Demand</b><br>{r.get("analysis",{}).get("market_demand","")[:160]}…</div></div>', unsafe_allow_html=True)
    with col_a: _comp_card(r_a, a_data, "Idea A")
    with col_mid:
        winner = "A" if r_a.get("success_probability", 0) >= r_b.get("success_probability", 0) else "B"
        st.markdown(f'<div style="text-align:center;padding-top:5rem;"><div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.5rem;color:#F59E0B;">VS</div><br><div style="font-size:.75rem;color:#9CA3AF;">Winner</div><div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.8rem;color:#00E5A0;">{winner}</div></div>', unsafe_allow_html=True)
    with col_b: _comp_card(r_b, b_data, "Idea B")


# ─────────────────────────────────────────────────────────
#  TAB: LEADERBOARD
# ─────────────────────────────────────────────────────────

def _tab_leaderboard():
    st.markdown('<h3 style="margin-bottom:1rem;">Idea Leaderboard</h3>', unsafe_allow_html=True)
    analyses = get_user_analyses(st.session_state.user["id"], limit=50)
    if not analyses:
        st.markdown('<div style="color:#9CA3AF;text-align:center;padding:3rem;">No ideas yet.</div>', unsafe_allow_html=True)
        return
    sort_by = st.radio("Sort by", ["Success Score", "Sustainability Score", "Combined Score"], horizontal=True, key="lb_sort")
    scored = []
    for a in analyses:
        r = json.loads(a["result"]) if a.get("result") else {}
        sp = r.get("success_probability", 0); ss = r.get("sustainability_score", 0)
        scored.append({**a, "_sp": sp, "_ss": ss, "_combined": (sp + ss) // 2})
    key_map = {"Success Score": "_sp", "Sustainability Score": "_ss", "Combined Score": "_combined"}
    scored.sort(key=lambda x: x[key_map[sort_by]], reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for rank, a in enumerate(scored, 1):
        sp = a["_sp"]; ss = a["_ss"]; comb = a["_combined"]
        sp_c = "#00E5A0" if sp >= 60 else "#F59E0B" if sp >= 40 else "#EF4444"
        title = a.get("title") or a.get("idea", "")[:40] + "..."
        medal = medals[rank - 1] if rank <= 3 else f"#{rank}"
        st.markdown(f'<div class="leaderboard-row"><div class="rank-num">{medal}</div><div style="flex:1;"><div style="font-family:\'Syne\',sans-serif;font-weight:600;color:#CBD5E1;">{title}</div><div style="font-size:.78rem;color:#9CA3AF;">{a.get("domain","")} · {a.get("locality","")}</div></div><div style="text-align:center;min-width:60px;"><div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.3rem;color:{sp_c};">{sp}%</div><div style="font-size:.7rem;color:#9CA3AF;">success</div></div><div style="text-align:center;min-width:60px;"><div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.3rem;color:#3B82F6;">{ss}%</div><div style="font-size:.7rem;color:#9CA3AF;">sustain</div></div><div style="text-align:center;min-width:60px;"><div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.3rem;color:#F59E0B;">{comb}%</div><div style="font-size:.7rem;color:#9CA3AF;">combined</div></div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  TAB: Dataset Analysis
# ─────────────────────────────────────────────────────────

def _tab_dataset():
    """Standalone Dataset Analysis tab — full explorer of startup_intelligence_dataset.csv"""
    st.markdown('<h3 style="margin-bottom:.4rem;">📂 Dataset Analysis</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1.2rem;">'
        'Analysis of <code>startup_intelligence_dataset.csv</code> — '
        '500 real startup records · 2024–2026 · 25 fields · '
        '<strong style="color:#f97316;">All monetary values in ₹ INR</strong></div>',
        unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.markdown('<div class="alert-error">⚠️ Dataset file not found.</div>', unsafe_allow_html=True)
        return

    # ── Summary cards — INR ───────────────────────────────────────────────────
    total_inr = df["Amount_INR"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",   f"{len(df):,}",                    "startup entries")
    m2.metric("Total Funding",   _fmt_inr(total_inr),               "raised in dataset · ₹ INR")
    m3.metric("Domains",         str(df["Domain"].nunique()),        "sectors covered")
    m4.metric("Years",           f"{int(df['Year'].min())}–{int(df['Year'].max())}", "data coverage")

    st.markdown("<br>", unsafe_allow_html=True)

    selected_domain = st.selectbox(
        "🔍 Filter by Domain",
        ["All Domains"] + sorted(df["Domain"].unique().tolist()),
        key="ds_domain_filter"
    )
    filtered = df if selected_domain == "All Domains" else df[df["Domain"] == selected_domain]
    st.markdown(
        f'<div style="color:#9CA3AF;font-size:.8rem;margin-bottom:.8rem;">'
        f'Showing {len(filtered)} records</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if selected_domain != "All Domains":
        st.markdown(f'<div class="section-title">📊 {selected_domain} — Domain Summary</div>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Avg Success Rate",  f"{int(filtered['Success_Rate'].mean())}%")
        s2.metric("Avg Sustainability", str(int(filtered["Sustainability_Score"].mean())))
        s3.metric("Avg Market Demand",  str(int(filtered["Market_Demand_Score"].mean())))
        s4.metric("Avg Growth Rate",    f"{round(filtered['Growth_Rate_Pct'].mean(),1)}%")
        s5.metric("Avg Social Impact",  str(int(filtered["Social_Impact_Score"].mean())))
        s6.metric("Avg Team Size",      str(int(filtered["Ideal_Team_Size"].mean())))
        st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌱 Sustainability", "📈 Market & Growth", "💰 Investment (₹ INR)", "⚠️ Risk & Failure"]
    )

    def _insight_box(text, color="#9CA3AF", bg="rgba(255,255,255,.03)"):
        st.markdown(
            f'''<div style="background:{bg};border-left:3px solid {color};
                 border-radius:0 8px 8px 0;padding:.65rem 1rem;margin-top:.5rem;
                 margin-bottom:1.2rem;font-size:.83rem;color:#CBD5E1;line-height:1.7;">{text}</div>''',
            unsafe_allow_html=True)

    with tab1:
        st.markdown("**Average Sustainability Score by Domain**")
        sus = (filtered.groupby("Domain")["Sustainability_Score"]
               .mean().sort_values(ascending=False).rename("Avg Sustainability Score"))
        st.bar_chart(sus, height=280)
        top_sus = sus.index[0]; low_sus = sus.index[-1]
        top_val = int(sus.iloc[0]); low_val = int(sus.iloc[-1])
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#00E5A0;'>{top_sus}</strong> leads "
            f"with an avg sustainability score of <strong>{top_val}/100</strong>. "
            f"<strong style='color:#EF4444;'>{low_sus}</strong> scores lowest at <strong>{low_val}/100</strong>.",
            color="#00E5A0", bg="rgba(0,229,160,.05)")

        st.markdown("**Environmental Impact Distribution**")
        env = filtered["Environmental_Impact"].value_counts().rename("Count")
        st.bar_chart(env, height=200)
        low_env_pct = int(round(env.get("Low", 0) / len(filtered) * 100))
        _insight_box(
            f"<strong>Key Insight:</strong> {low_env_pct}% of startups have "
            f"<strong style='color:#00E5A0;'>Low</strong> environmental impact.",
            color="#3B82F6", bg="rgba(59,130,246,.05)")

    with tab2:
        st.markdown("**Average Market Demand Score by Domain**")
        mkt = (filtered.groupby("Domain")["Market_Demand_Score"]
               .mean().sort_values(ascending=False).rename("Avg Market Demand"))
        st.bar_chart(mkt, height=280)
        top_mkt = mkt.index[0]; top_mval = int(mkt.iloc[0]); avg_mval = int(mkt.mean())
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#3B82F6;'>{top_mkt}</strong> has "
            f"the highest market demand score ({top_mval}/100). Overall avg: <strong>{avg_mval}/100</strong>.",
            color="#3B82F6", bg="rgba(59,130,246,.05)")

        st.markdown("**Average YoY Growth Rate by Domain (%)**")
        gr = (filtered.groupby("Domain")["Growth_Rate_Pct"]
              .mean().sort_values(ascending=False).rename("Avg Growth Rate (%)"))
        st.bar_chart(gr, height=280)
        top_gr = gr.index[0]; top_grval = round(gr.iloc[0], 1); avg_gr = round(gr.mean(), 1)
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#F59E0B;'>{top_gr}</strong> is "
            f"the fastest-growing domain at <strong>{top_grval}% YoY</strong>. Dataset avg: <strong>{avg_gr}% YoY</strong>.",
            color="#F59E0B", bg="rgba(245,158,11,.05)")

    with tab3:
        # ── All funding charts now use Amount_INR ─────────────────────────────
        st.markdown("**Investment Stage Distribution**")
        stage = filtered["Investment_Stage"].value_counts().rename("Count")
        st.bar_chart(stage, height=250)
        top_stage = stage.index[0]; top_stg_cnt = int(stage.iloc[0])
        seed_pct  = int(round(stage.get("Seed", 0) / len(filtered) * 100))
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#8B5CF6;'>{top_stage}</strong> "
            f"is most common with <strong>{top_stg_cnt}</strong> startups. "
            f"<strong>{seed_pct}%</strong> are at Seed stage.",
            color="#8B5CF6", bg="rgba(139,92,246,.05)")

        st.markdown("**Total Funding by Domain (₹ INR — Crores)**")
        fund = (filtered.groupby("Domain")["Amount_INR"]
                .sum().sort_values(ascending=False) / 1e7)  # convert to Crores
        fund = fund.rename("Total Funding (₹ Crores)")
        st.bar_chart(fund, height=280)
        top_fund     = fund.index[0]
        top_fund_cr  = round(fund.iloc[0], 1)
        total_cr     = round(fund.sum(), 1)
        top_fund_pct = int(round(fund.iloc[0] / fund.sum() * 100))
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#00E5A0;'>{top_fund}</strong> "
            f"has raised the most — <strong>₹{top_fund_cr} Cr</strong> — "
            f"representing <strong>{top_fund_pct}%</strong> of the total <strong>₹{total_cr} Cr</strong>.",
            color="#00E5A0", bg="rgba(0,229,160,.05)")

        st.markdown("**Total Funding by City (₹ INR — Crores)**")
        city_fund = (filtered.groupby("City_Location")["Amount_INR"]
                     .sum().sort_values(ascending=False) / 1e7)
        city_fund = city_fund.rename("Total Funding (₹ Crores)")
        st.bar_chart(city_fund, height=250)
        top_city_f   = city_fund.index[0]
        top_city_cr  = round(city_fund.iloc[0], 1)
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#22d3ee;'>{top_city_f}</strong> "
            f"leads with <strong>₹{top_city_cr} Cr</strong> in total funding.",
            color="#22d3ee", bg="rgba(34,211,238,.05)")

    with tab4:
        st.markdown("**Average Success Rate by Domain (%)**")
        suc = (filtered.groupby("Domain")["Success_Rate"]
               .mean().sort_values(ascending=False).rename("Avg Success Rate (%)"))
        st.bar_chart(suc, height=280)
        top_suc = suc.index[0]; top_suc_v = int(suc.iloc[0])
        low_suc = suc.index[-1]; low_suc_v = int(suc.iloc[-1])
        overall_sr = int(suc.mean())
        _insight_box(
            f"<strong>Key Insight:</strong> <strong style='color:#00E5A0;'>{top_suc}</strong> "
            f"has the highest success rate at <strong>{top_suc_v}%</strong>. "
            f"Weakest: <strong style='color:#EF4444;'>{low_suc}</strong> at <strong>{low_suc_v}%</strong>. "
            f"Dataset avg: <strong>{overall_sr}%</strong>.",
            color="#00E5A0", bg="rgba(0,229,160,.05)")

        st.markdown("**Risk Level Distribution**")
        risk = filtered["Risk_Level"].value_counts().rename("Count")
        st.bar_chart(risk, height=200)
        high_risk_pct = int(round(risk.get("High", 0) / len(filtered) * 100))
        med_risk_pct  = int(round(risk.get("Medium", 0) / len(filtered) * 100))
        _insight_box(
            f"<strong>Key Insight:</strong> <strong>{high_risk_pct}%</strong> carry "
            f"<strong style='color:#EF4444;'>High</strong> risk and <strong>{med_risk_pct}%</strong> carry "
            f"<strong style='color:#F59E0B;'>Medium</strong> risk.",
            color="#F59E0B", bg="rgba(245,158,11,.05)")

        st.markdown("**Top Failure Reasons**")
        fail = filtered["Failure_Reasons"].value_counts().head(8).rename("Count")
        st.bar_chart(fail, height=260)
        top_fail = fail.index[0]; top_fail_cnt = int(fail.iloc[0])
        _insight_box(
            f"<strong>Key Insight:</strong> Most common failure reason: "
            f"<strong style='color:#EF4444;'>'{top_fail}'</strong> — affecting "
            f"<strong>{top_fail_cnt}</strong> startups.",
            color="#EF4444", bg="rgba(239,68,68,.05)")


# ─────────────────────────────────────────────────────────
#  TAB: ADVISOR AVATAR
# ─────────────────────────────────────────────────────────

ADVISOR_SYSTEM = """You are a startup advisor and business consultant specialising in Indian entrepreneurship.

Your approach:
- Honest and practical — give real assessments, not just encouragement
- Step-by-step — break advice into clear actions
- India-specific — reference Indian regulations, funding bodies (Startup India, SIDBI, NASSCOM), and market realities
- All monetary values in Indian Rupees (₹ INR) — never use USD

You help with: MVP building, hiring, legal registration (GST, Pvt Ltd), funding (angels, VCs, grants),
marketing, customer acquisition, analysis interpretation, and general business strategy.

End every response with one concrete action the user can take today.
"""

QUICK_PROMPTS = [
    "How do I register my startup in India?",
    "What is the cheapest way to build an MVP?",
    "How do I find co-founders?",
    "Which government grants can I apply for?",
    "How do I pitch to angel investors?",
    "How do I get my first 100 customers?",
    "Should I bootstrap or raise funding?",
    "How do I protect my idea legally?",
    "What mistakes do first-time founders make?",
    "How do I calculate my startup valuation?",
]


def _tab_avatar():
    hc1, hc2, hc3 = st.columns([1, 2, 1])
    with hc2:
        ctx_html = ""
        if st.session_state.analysis_result:
            r = st.session_state.analysis_result; meta = st.session_state.analysis_meta
            ctx_html = (f'<div style="background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);'
                        f'border-radius:10px;padding:.7rem 1rem;font-size:.82rem;color:#9CA3AF;margin-bottom:1rem;">'
                        f'<strong style="color:#00E5A0;">Context loaded:</strong> '
                        f'{meta.get("title","Your idea")} — {r.get("success_probability",0)}% success · '
                        f'{r.get("sustainability_score",0)}% sustainability</div>')
        st.markdown(f'''
        <div style="text-align:center;padding:1.5rem 0 .5rem;">
          <div class="avatar-face">👨‍💼</div>
          <h2 style="font-family:'Syne',sans-serif;font-weight:800;margin-bottom:.2rem;">Your Startup Advisor</h2>
          <div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1rem;">
            Business Consultant · India Startup Specialist · ₹ INR Focus
          </div>
          {ctx_html}
        </div>''', unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:.5rem;color:#9CA3AF;font-size:.82rem;">Common questions:</div>', unsafe_allow_html=True)
    qcols = st.columns(5)
    for i, qp in enumerate(QUICK_PROMPTS):
        with qcols[i % 5]:
            if st.button(qp[:26] + ("…" if len(qp) > 26 else ""), key=f"qp_{i}", use_container_width=True):
                st.session_state._advisor_prefill = qp

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.avatar_history:
        st.markdown('''<div class="avatar-bubble-ai">
          Hello! I am your startup advisor, specialising in the Indian business ecosystem.<br><br>
          I can help you with <strong>implementing your idea</strong>, overcoming <strong>operational blockers</strong>,
          navigating <strong>legal and funding</strong> requirements (all amounts in ₹ INR), and creating a concrete action plan
          from your analysis results.<br><br>
          What would you like help with today?
        </div>''', unsafe_allow_html=True)
    for msg in st.session_state.avatar_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="avatar-bubble-user">You: {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="avatar-bubble-ai"><strong>Advisor:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    prefill = st.session_state.pop("_advisor_prefill", "") if "_advisor_prefill" in st.session_state else ""

    with st.form("avatar_form", clear_on_submit=True):
        user_input = st.text_area("Your question", value=prefill, height=90,
                                  placeholder="e.g. I cannot find technical co-founders in Hyderabad. What should I do?",
                                  label_visibility="collapsed")
        sc1, sc2, sc3 = st.columns([3, 1, 1])
        with sc1: send = st.form_submit_button("Ask Advisor", use_container_width=True)
        with sc2: clear = st.form_submit_button("Clear", use_container_width=True)
        with sc3:
            deep_dive = st.form_submit_button("Deep Dive", use_container_width=True) if st.session_state.analysis_result else False

    if clear:
        st.session_state.avatar_history = []; st.rerun()

    if deep_dive and st.session_state.analysis_result:
        r = st.session_state.analysis_result; meta = st.session_state.analysis_meta
        user_input = (f"Give me a complete deep dive on my idea: {meta.get('idea','')}. "
                      f"It scored {r.get('success_probability',0)}% success and {r.get('sustainability_score',0)}% sustainability. "
                      f"Cover: (1) top 3 risks, (2) 30-60-90 day action plan, (3) how to improve the scores, "
                      f"(4) the single most important step this week. Use ₹ INR for all monetary figures.")
        send = True

    if send and user_input.strip():
        question = user_input.strip()
        st.session_state.avatar_history.append({"role": "user", "content": question})
        system = ADVISOR_SYSTEM
        if st.session_state.analysis_result:
            r = st.session_state.analysis_result; meta = st.session_state.analysis_meta
            system += (f"\n\nUser's startup context:\nIdea: {meta.get('idea','')}\n"
                       f"Domain: {meta.get('domain','')} | Market: {meta.get('locality','')} | "
                       f"Budget: {meta.get('budget','')} | Team: {meta.get('team_size','')} people\n"
                       f"Scores — Success: {r.get('success_probability',0)}% | Sustainability: {r.get('sustainability_score',0)}%\n"
                       f"Summary: {r.get('summary','')[:400]}\nImprovements: {r.get('improvements',[])}")
        messages = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.avatar_history[-12:]]
        with st.spinner("Advisor is responding…"):
            try:
                reply = _chat_call(messages)
                st.session_state.avatar_history.append({"role": "assistant", "content": reply})
            except Exception as exc:
                st.session_state.avatar_history.append({"role": "assistant", "content": f"Error: {exc}. Please try again."})
        st.rerun()


# ═══════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════

import urllib.request

def main():
    page = st.session_state.page
    if st.session_state.user and page in ("home", "login", "register"):
        st.session_state.page = "dashboard"; st.rerun()
    if not st.session_state.user and page == "dashboard":
        st.session_state.page = "login"; st.rerun()
    {"home": page_home, "login": page_login, "register": page_register, "dashboard": page_dashboard}.get(page, page_home)()


if __name__ == "__main__":
    main()