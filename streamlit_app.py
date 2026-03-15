import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import torch
import psycopg2
from sentence_transformers import SentenceTransformer, util
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIGURATION
#
# Create a file named `.env` in the same folder as this script
# and fill in your own values. Example .env file:
#
#   DB_NAME=your_database_name
#   DB_USER=your_database_user
#   DB_PASS=your_database_password
#   DB_HOST=your_database_host
#
# NEVER commit your .env file to GitHub.
# Add `.env` to your .gitignore file.
# ─────────────────────────────────────────────
load_dotenv()

def _require_env(var: str) -> str:
    """Return the value of an environment variable, or exit with a clear error."""
    value = os.getenv(var)
    if not value:
        sys.exit(
            f"\n[CONFIG ERROR] Environment variable '{var}' is not set.\n"
            f"Please create a .env file with your credentials.\n"
            f"See the CONFIGURATION section in streamlit_app.py for instructions.\n"
        )
    return value

DB_CONFIG = dict(
    dbname=_require_env("DB_NAME"),
    user=_require_env("DB_USER"),
    password=_require_env("DB_PASS"),
    host=_require_env("DB_HOST"),
)
TABLE = "papers"

TIER_SQL = """
    CASE
        WHEN github_stars >= 1000 THEN 'Star Projects (1000+)'
        WHEN github_stars >= 100  THEN 'Popular Projects (100-999)'
        ELSE 'Hidden Gems (20-99)'
    END
"""

# ─────────────────────────────────────────────
# FONT CONSTANTS — change here to affect ALL charts
# ─────────────────────────────────────────────
FS = 16   # base font size for everything

def apply_font(fig, bg="#1e2130", height=480, top=70, bottom=30, extra=None):
    """Apply consistent font sizes and layout to every Plotly chart."""
    layout = dict(
        height=height,
        margin=dict(t=top, b=bottom, l=10, r=10),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(size=FS, color="white"),
        xaxis=dict(
            tickfont=dict(size=FS, color="white"),
            title_font=dict(size=FS, color="white")
        ),
        yaxis=dict(
            tickfont=dict(size=FS, color="white"),
            title_font=dict(size=FS, color="white")
        ),
        legend=dict(font=dict(size=FS, color="white")),
    )
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    fig.update_traces(textfont=dict(size=FS-1, color="white"))
    return fig

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Knowledge Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    .metric-card {{
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px; padding: 20px;
        border-left: 4px solid #7c83fd; margin-bottom: 10px;
    }}
    .metric-value {{ font-size: 2rem; font-weight: bold; color: #7c83fd; }}
    .metric-label {{ font-size: 1.05rem; color: #aaaaaa; margin-top: 4px; }}
    .section-header {{
        font-size: 1.5rem; font-weight: bold; color: #ffffff;
        border-bottom: 2px solid #7c83fd; padding-bottom: 8px; margin-bottom: 16px;
    }}
    .numpy-box {{
        background: #1a1d2e; border: 1px solid #7c83fd; border-radius: 8px;
        padding: 16px; font-family: monospace; font-size: 1.05rem;
        color: #7cfdb4; margin-bottom: 12px;
    }}
    section[data-testid="stSidebar"] {{ font-size: 1.1rem !important; }}
    section[data-testid="stSidebar"] label {{ font-size: 1.1rem !important; }}
    section[data-testid="stSidebar"] p {{ font-size: 1.1rem !important; }}
    section[data-testid="stSidebar"] span {{ font-size: 1.05rem !important; }}
    .stRadio label {{ font-size: 1.1rem !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
@st.cache_resource
def get_engine():
    h = _require_env("DB_HOST")
    u = _require_env("DB_USER")
    p = _require_env("DB_PASS")
    d = _require_env("DB_NAME")
    return create_engine(f"postgresql+psycopg2://{u}:{p}@{h}/{d}")

@st.cache_data(ttl=300)
def run_query(query, params=None):
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params=params)

@st.cache_data(ttl=300)
def load_full_dataset():
    return run_query(f"""
        SELECT github_stars, score_gemini, usability_score,
               topic_category, primary_language, code_maturity,
               failure_category, deployment_target, country,
               published_year, has_demo, license_type,
               {TIER_SQL} as tier
        FROM {TABLE} WHERE github_stars IS NOT NULL AND github_stars >= 20
    """)

def tier_where(tier_filter):
    conditions = []
    if "Hidden Gems (20-99)"        in tier_filter: conditions.append("(github_stars >= 20  AND github_stars < 100)")
    if "Popular Projects (100-999)" in tier_filter: conditions.append("(github_stars >= 100 AND github_stars < 1000)")
    if "Star Projects (1000+)"      in tier_filter: conditions.append("(github_stars >= 1000)")
    return " OR ".join(conditions) if conditions else "1=1"

def clean_paper_link(pid):
    if not pid: return "#"
    pid = str(pid).strip()
    if "arxiv.org" in pid:
        return pid.replace("http:/", "http://").replace("https:/", "https://")
    return f"https://arxiv.org/abs/{pid}"

def render_card(row):
    pid=row[0]; title=row[1]; fw=row[4] if len(row)>4 else ""
    aff=row[5] if len(row)>5 else ""; stars=row[6] if len(row)>6 else 0
    gh_link=row[7] if len(row)>7 else "#"; year=row[9] if len(row)>9 else ""
    g_score=row[10] if len(row)>10 else None
    g_opinion=row[11] if len(row)>11 else ""; g_improve=row[12] if len(row)>12 else ""
    paper_link = clean_paper_link(pid)
    if g_score is not None:
        color = "green" if g_score >= 85 else "orange" if g_score >= 70 else "red"
        gemini_badge = f":{color}[**Gemini Score: {g_score}/100**]"
        has_judgment = True
    else:
        gemini_badge = "🤖 Not Judged"; has_judgment = False
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(title)
            st.markdown(f"**📂 Repo:** [{gh_link}]({gh_link})  |  **📄 Paper:** [{paper_link}]({paper_link})")
            st.markdown(f"📅 {year} • {gemini_badge}")
            if has_judgment:
                with st.expander("🗣️ Read Gemini's Verdict"):
                    st.markdown(f"**Opinion:** _{g_opinion}_")
                    if g_improve:
                        st.divider(); st.markdown("**🔧 Suggested Improvements:**"); st.markdown(g_improve)
        with c2:
            st.info(f"{fw or 'No Framework'}\n\n{aff or 'Unknown'}\n\n⭐ {stars or '?'}")
        st.divider()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_semantic_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute(f"""
        SELECT paper_id, title, embedding, NULL, framework, affiliation_type,
               github_stars, github_link, topic_category, published_year,
               score_gemini, opinion_gemini, improvements_gemini
        FROM {TABLE} WHERE embedding IS NOT NULL AND usability_score IS NOT NULL AND github_stars >= 20;
    """)
    rows = cur.fetchall(); conn.close()
    return [r[1] for r in rows], torch.tensor([r[2] for r in rows]), rows

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🧠 AI Knowledge Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview","🔍 Semantic Search","🤖 Gemini Tribunal",
    "📊 Analytics","🔬 Deep Analysis","⚠️ Risk Explorer","🌍 Global Map",
])
analytics_pages = ["🏠 Overview","📊 Analytics","🔬 Deep Analysis","⚠️ Risk Explorer","🌍 Global Map"]
if page in analytics_pages:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Filters")
    tier_filter = st.sidebar.multiselect(
        "Project Tier",
        ["Hidden Gems (20-99)", "Popular Projects (100-999)", "Star Projects (1000+)"],
        default=["Hidden Gems (20-99)", "Popular Projects (100-999)", "Star Projects (1000+)"]
    )
    if not tier_filter:
        tier_filter = ["Hidden Gems (20-99)", "Popular Projects (100-999)", "Star Projects (1000+)"]
    tw = tier_where(tier_filter)
else:
    tw = "github_stars >= 20"

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🧠 AI Research Knowledge Engine")
    st.markdown("##### From Academic Papers to Production Code — 18,447 Projects Analyzed")
    st.markdown("---")

    kpis = run_query(f"""
        SELECT COUNT(*) as total_projects, ROUND(AVG(score_gemini)::numeric,1) as avg_ai_score,
               MAX(github_stars) as max_stars, COUNT(DISTINCT topic_category) as unique_topics,
               COUNT(DISTINCT primary_language) as unique_languages, COUNT(DISTINCT country) as unique_countries
        FROM {TABLE} WHERE {tw} AND github_stars IS NOT NULL
    """)
    for col, (label, value, icon) in zip(st.columns(6), [
        ("Total Projects", f"{int(kpis['total_projects'][0]):,}", "📁"),
        ("Avg AI Score",   f"{kpis['avg_ai_score'][0]}/100",      "🤖"),
        ("Max Stars",      f"{int(kpis['max_stars'][0]):,}",       "⭐"),
        ("Topics",         f"{int(kpis['unique_topics'][0])}",     "🏷️"),
        ("Languages",      f"{int(kpis['unique_languages'][0])}",  "💻"),
        ("Countries",      f"{int(kpis['unique_countries'][0])}",  "🌍")]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1.5rem">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tier Distribution
    st.markdown('<div class="section-header">Project Distribution by Tier</div>', unsafe_allow_html=True)
    df = run_query(f"""SELECT
        CASE WHEN github_stars >= 1000 THEN 'Star (1000+)'
             WHEN github_stars >= 100  THEN 'Popular (100-999)'
             ELSE 'Hidden Gems (20-99)' END as tier, COUNT(*) as count
        FROM {TABLE} WHERE github_stars IS NOT NULL AND {tw} GROUP BY tier ORDER BY count DESC""")
    fig = px.bar(df, x="tier", y="count", color="tier", text="count",
                 color_discrete_sequence=["#7c83fd","#fd7c83","#7cfdb4"])
    apply_font(fig, height=460, top=80, extra=dict(
        showlegend=False, xaxis_title="", yaxis_title="Count",
        yaxis=dict(range=[0, df["count"].max()*1.18], tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig.update_traces(textposition="outside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig, use_container_width=True)

    # Top Topics
    st.markdown('<div class="section-header">Top 10 Research Topics</div>', unsafe_allow_html=True)
    df2 = run_query(f"""SELECT topic_category, COUNT(*) as count FROM {TABLE}
        WHERE github_stars IS NOT NULL AND topic_category IS NOT NULL
        GROUP BY topic_category ORDER BY count DESC LIMIT 10""")
    fig2 = px.bar(df2, x="count", y="topic_category", orientation="h",
                  color="count", color_continuous_scale="Plasma", text="count")
    apply_font(fig2, height=480, extra=dict(
        yaxis_title="", xaxis_title="Number of Projects", coloraxis_showscale=False,
        margin=dict(t=20, b=20, l=220, r=80),
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig2.update_traces(textposition="outside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig2, use_container_width=True, key="top_topics")

    # Top Languages
    st.markdown('<div class="section-header">Top Programming Languages</div>', unsafe_allow_html=True)
    df3 = run_query(f"""SELECT primary_language, COUNT(*) as count FROM {TABLE}
        WHERE {tw} AND primary_language IS NOT NULL
        GROUP BY primary_language ORDER BY count DESC LIMIT 12""")
    fig3 = px.bar(df3, x="primary_language", y="count", color="count",
                  color_continuous_scale="Blues", text="count")
    apply_font(fig3, height=480, top=80, extra=dict(
        xaxis_title="", yaxis_title="Count", coloraxis_showscale=False,
        yaxis=dict(range=[0, df3["count"].max()*1.18], tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig3.update_traces(textposition="outside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2 — SEMANTIC SEARCH
# ─────────────────────────────────────────────
elif page == "🔍 Semantic Search":
    st.title("🔍 Semantic Search")
    st.caption("Find projects by describing what you want to build.")
    with st.spinner("Loading knowledge base..."):
        model = load_model()
        _, all_embeddings, all_rows = load_semantic_data()
    st.sidebar.markdown("---"); st.sidebar.header("🔍 Filters")
    unique_topics = sorted(set(r[8] for r in all_rows if r[8]))
    unique_fw     = sorted(set(r[4] for r in all_rows if r[4]))
    unique_aff    = sorted(set(r[5] for r in all_rows if r[5]))
    sel_topic = st.sidebar.selectbox("Topic",       ["All"] + unique_topics)
    sel_fw    = st.sidebar.selectbox("Framework",   ["All"] + unique_fw)
    sel_aff   = st.sidebar.selectbox("Affiliation", ["All"] + unique_aff)
    sel_stars = st.sidebar.selectbox("Popularity",
        ["All","🔥 High (>1k Stars)","👍 Medium (100-1k Stars)","🌱 Emerging (<100 Stars)"])
    filtered = []
    for i, r in enumerate(all_rows):
        if sel_topic != "All" and r[8] != sel_topic: continue
        if sel_fw    != "All" and r[4] != sel_fw:    continue
        if sel_aff   != "All" and r[5] != sel_aff:   continue
        s = r[6] or 0
        if sel_stars == "🔥 High (>1k Stars)"     and s < 1000:              continue
        if sel_stars == "👍 Medium (100-1k Stars)" and not (100 <= s < 1000): continue
        if sel_stars == "🌱 Emerging (<100 Stars)" and s >= 100:              continue
        filtered.append(i)
    if not filtered:
        st.error("No projects match these filters.")
    else:
        cur_emb  = all_embeddings[filtered]
        cur_rows = [all_rows[i] for i in filtered]
        st.sidebar.write(f"**Active Projects:** {len(filtered)}")
        query = st.text_input("Search query:", placeholder="e.g. Real-time object detection on mobile")
        if query:
            q_emb  = model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(q_emb, cur_emb)[0]
            top    = torch.topk(scores, k=min(10, len(cur_rows)))
            st.markdown(f"### Results for '{query}'")
            for _, idx in zip(top[0], top[1]):
                render_card(cur_rows[int(idx)])

# ─────────────────────────────────────────────
# PAGE 3 — GEMINI TRIBUNAL
# ─────────────────────────────────────────────
elif page == "🤖 Gemini Tribunal":
    st.title("🤖 Gemini Tribunal")
    st.caption("Projects judged and scored by Gemini AI — ranked by quality.")
    with st.spinner("Loading judgements..."):
        _, _, all_rows = load_semantic_data()
    judged = sorted([r for r in all_rows if r[10] is not None], key=lambda x: x[10], reverse=True)
    if not judged:
        st.warning("No judged projects yet.")
    else:
        st.success(f"🏆 {len(judged)} projects judged — highest to lowest.")
        scores_df = pd.DataFrame({"score": [r[10] for r in judged]})
        fig = px.histogram(scores_df, x="score", nbins=20,
                           color_discrete_sequence=["#7c83fd"], title="Distribution of Gemini Scores")
        apply_font(fig, extra=dict(xaxis_title="Score", yaxis_title="Count"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        for row in judged:
            render_card(row)

# ─────────────────────────────────────────────
# PAGE 4 — ANALYTICS
# ─────────────────────────────────────────────
elif page == "📊 Analytics":
    st.title("📊 Quality & Popularity Analytics")
    st.markdown("---")

    # Chart 1
    st.markdown('<div class="section-header">AI Score by Code Maturity</div>', unsafe_allow_html=True)
    df = run_query(f"""SELECT code_maturity,
           ROUND(AVG(score_gemini)::numeric,1) as avg_score, COUNT(*) as total
        FROM {TABLE} WHERE github_stars IS NOT NULL AND code_maturity IS NOT NULL AND score_gemini IS NOT NULL
        GROUP BY code_maturity ORDER BY avg_score DESC""")
    fig = px.bar(df, x="avg_score", y="code_maturity", orientation="h",
                 text="avg_score", color="avg_score", color_continuous_scale="RdYlGn")
    apply_font(fig, bg="#16213e", extra=dict(
        yaxis_title="", xaxis_title="Avg AI Score", coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig.update_traces(textposition="outside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig, use_container_width=True, key="code_maturity")

    # Chart 2
    st.markdown('<div class="section-header">Popularity vs AI Score by Tier</div>', unsafe_allow_html=True)
    df2 = run_query(f"""SELECT {TIER_SQL} as tier,
           ROUND(AVG(score_gemini)::numeric,1) as avg_score,
           ROUND(AVG(github_stars)::numeric,0) as avg_stars,
           ROUND(AVG(usability_score)::numeric,1) as avg_usability
        FROM {TABLE} WHERE github_stars IS NOT NULL GROUP BY tier""")
    fig2 = px.scatter(df2, x="tier", y="avg_score", size="avg_usability",
                      color="tier", text="avg_score",
                      color_discrete_sequence=["#7c83fd","#fd7c83","#7cfdb4"], size_max=80)
    apply_font(fig2, bg="#16213e", height=500, extra=dict(
        showlegend=False, xaxis_title="Project Tier", yaxis_title="Avg AI Score",
        yaxis=dict(range=[60,80], tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig2.update_traces(textposition="middle center", textfont=dict(size=FS+2, color="white"))
    st.plotly_chart(fig2, use_container_width=True, key="scatter_tier")

    # Chart 3
    st.markdown('<div class="section-header">Score Distribution by Tier</div>', unsafe_allow_html=True)
    df3 = run_query(f"""SELECT {TIER_SQL} as tier,
           ROUND(score_gemini::numeric,0) as score_bucket, COUNT(*) as count
        FROM {TABLE} WHERE score_gemini IS NOT NULL AND github_stars IS NOT NULL
        GROUP BY tier, ROUND(score_gemini::numeric,0) ORDER BY score_bucket""")
    fig3 = px.line(df3, x="score_bucket", y="count", color="tier",
                   color_discrete_sequence=["#7c83fd","#fd7c83","#7cfdb4"])
    apply_font(fig3, bg="#16213e", extra=dict(xaxis_title="AI Score", yaxis_title="Projects"))
    st.plotly_chart(fig3, use_container_width=True, key="score_dist")

# ─────────────────────────────────────────────
# PAGE 5 — DEEP ANALYSIS
# ─────────────────────────────────────────────
elif page == "🔬 Deep Analysis":
    st.title("🔬 Deep Analysis")
    st.markdown("##### Statistical analysis powered by NumPy, Pandas & Matplotlib")
    st.markdown("---")
    with st.spinner("Loading dataset..."):
        df = load_full_dataset()
    df = df[df["tier"].isin(tier_filter)].copy()
    df_scored = df[df["score_gemini"].notna()].copy()

    st.markdown('<div class="section-header">📋 Pandas — Dataset Summary Statistics</div>', unsafe_allow_html=True)
    st.caption("Using: `df.describe()`, `df.groupby()`, `df.corr()`")
    numeric_cols = ["github_stars", "score_gemini", "usability_score"]
    summary = df_scored[numeric_cols].describe().round(2)
    st.dataframe(summary, use_container_width=True)
    st.markdown("**Grouped statistics by Tier — using `df.groupby('tier').agg()`**")
    grouped = df_scored.groupby("tier").agg(
        Count=("score_gemini","count"), Avg_AI_Score=("score_gemini","mean"),
        Std_AI_Score=("score_gemini","std"), Avg_Stars=("github_stars","mean"),
        Max_Stars=("github_stars","max"), Avg_Usability=("usability_score","mean")
    ).round(2).reset_index()
    st.dataframe(grouped, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🔢 NumPy — Statistical Computations</div>', unsafe_allow_html=True)
    st.caption("Using: `np.mean()`, `np.std()`, `np.percentile()`, `np.corrcoef()`, `np.log1p()`")
    scores  = df_scored["score_gemini"].values
    stars   = df_scored["github_stars"].values
    log_stars = np.log1p(stars)
    corr_value = np.corrcoef(log_stars, scores)[0, 1]
    p25, p50, p75, p90 = np.percentile(scores, [25, 50, 75, 90])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="numpy-box">', unsafe_allow_html=True)
        st.markdown(f"""**Score Statistics**
- `np.mean()` → **{np.mean(scores):.2f}**
- `np.median()` → **{np.median(scores):.2f}**
- `np.std()` → **{np.std(scores):.2f}**
- `np.var()` → **{np.var(scores):.2f}**""")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="numpy-box">', unsafe_allow_html=True)
        st.markdown(f"""**Percentile Breakdown**
- P25 → **{p25:.1f}**
- P50 (Median) → **{p50:.1f}**
- P75 → **{p75:.1f}**
- P90 → **{p90:.1f}**""")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="numpy-box">', unsafe_allow_html=True)
        st.markdown(f"""**Correlation Analysis**
- Stars vs Score → **{corr_value:.4f}**
- Log-transformed Stars
- `np.corrcoef()` used
- {'Weak' if abs(corr_value) < 0.3 else 'Moderate'} correlation""")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Matplotlib — Visual Analysis</div>', unsafe_allow_html=True)
    st.caption("Using: `plt.hist()`, `plt.scatter()`, `plt.boxplot()`, `plt.bar()`, `plt.subplots()`")
    plt.style.use("dark_background")
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
                          "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
    colors = {"Hidden Gems (20-99)": "#7c83fd", "Popular Projects (100-999)": "#fd7c83", "Star Projects (1000+)": "#7cfdb4"}

    st.markdown("**Chart 1 — Score Distribution Histogram by Tier**")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for tier in df_scored["tier"].unique():
        tier_scores = df_scored[df_scored["tier"] == tier]["score_gemini"].values
        ax1.hist(tier_scores, bins=30, alpha=0.6, label=tier, color=colors.get(tier, "gray"), edgecolor="none")
    ax1.axvline(np.mean(scores), color="white", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(scores):.1f}")
    ax1.axvline(np.median(scores), color="yellow", linestyle=":", linewidth=1.5, label=f"Median: {np.median(scores):.1f}")
    ax1.set_xlabel("AI Score"); ax1.set_ylabel("Number of Projects")
    ax1.set_title("Score Distribution by Project Tier")
    ax1.legend(facecolor="#1e2130", edgecolor="#7c83fd", labelcolor="white")
    ax1.set_facecolor("#1e2130"); fig1.patch.set_facecolor("#1e2130")
    st.pyplot(fig1); plt.close(fig1)

    st.markdown("**Chart 2 — Score Boxplot by Tier**")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    tier_groups = [df_scored[df_scored["tier"] == t]["score_gemini"].values
                   for t in ["Hidden Gems (20-99)","Popular Projects (100-999)","Star Projects (1000+)"]]
    bp = ax2.boxplot(tier_groups, labels=["Hidden Gems","Popular","Star Projects"],
                     patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], ["#7c83fd","#fd7c83","#7cfdb4"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.set_ylabel("AI Score"); ax2.set_title("AI Score Boxplot by Tier")
    ax2.set_facecolor("#1e2130"); fig2.patch.set_facecolor("#1e2130"); ax2.tick_params(colors="white")
    st.pyplot(fig2); plt.close(fig2)

    st.markdown("**Chart 3 — Stars vs AI Score Scatter (log scale)**")
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for tier in df_scored["tier"].unique():
        sub = df_scored[df_scored["tier"] == tier]
        ax3.scatter(np.log1p(sub["github_stars"].values), sub["score_gemini"].values,
                    alpha=0.3, s=10, color=colors.get(tier, "gray"), label=tier)
    x_all = np.log1p(df_scored["github_stars"].values); y_all = df_scored["score_gemini"].values
    z = np.polyfit(x_all, y_all, 1); p = np.poly1d(z)
    x_line = np.linspace(x_all.min(), x_all.max(), 200)
    ax3.plot(x_line, p(x_line), color="white", linewidth=2, linestyle="--", label=f"Trend (slope={z[0]:.2f})")
    ax3.set_xlabel("Log(GitHub Stars + 1)"); ax3.set_ylabel("AI Score")
    ax3.set_title("Popularity vs Quality — Are They Correlated?")
    ax3.legend(facecolor="#1e2130", edgecolor="#7c83fd", labelcolor="white")
    ax3.set_facecolor("#1e2130"); fig3.patch.set_facecolor("#1e2130"); ax3.tick_params(colors="white")
    st.pyplot(fig3); plt.close(fig3)

    st.markdown("**Chart 4 — Top 10 Topics by Average AI Score**")
    topic_stats = df_scored.groupby("topic_category")["score_gemini"].agg(["mean","count"]).reset_index()
    topic_stats = topic_stats[topic_stats["count"] >= 10].sort_values("mean", ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    bars = ax4.barh(topic_stats["topic_category"], topic_stats["mean"],
                    color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(topic_stats))))
    ax4.set_xlabel("Avg AI Score"); ax4.set_title("Top 10 Topics by Average AI Score (min 10 projects)")
    ax4.set_facecolor("#1e2130"); fig4.patch.set_facecolor("#1e2130"); ax4.tick_params(colors="white")
    for bar, val in zip(bars, topic_stats["mean"]):
        ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}", va="center", color="white", fontsize=13)
    st.pyplot(fig4); plt.close(fig4)

    st.markdown("---")
    st.markdown("**Pandas Correlation Matrix — `df.corr()`**")
    corr_df = df_scored[["github_stars","score_gemini","usability_score"]].corr().round(3)
    st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
    st.markdown("---")
    st.markdown("**Top 10 projects by AI Score — `df.sort_values()`**")
    top10 = df_scored.nlargest(10, "score_gemini")[
        ["topic_category","tier","github_stars","score_gemini","usability_score","code_maturity"]
    ].reset_index(drop=True)
    top10.index += 1
    st.dataframe(top10, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 6 — RISK EXPLORER
# ─────────────────────────────────────────────
elif page == "⚠️ Risk Explorer":
    st.title("⚠️ Risk & Failure Analysis")
    st.markdown("---")

    # Chart 1
    st.markdown('<div class="section-header">Top Failure Categories</div>', unsafe_allow_html=True)
    df = run_query(f"""SELECT failure_category, COUNT(*) as count FROM {TABLE}
        WHERE {tw} AND failure_category IS NOT NULL AND failure_category != 'Unknown'
        GROUP BY failure_category ORDER BY count DESC LIMIT 15""")
    fig = px.bar(df, x="count", y="failure_category", orientation="h",
                 color="count", color_continuous_scale="Reds", text="count")
    apply_font(fig, bg="#16213e", height=520, extra=dict(
        yaxis_title="", coloraxis_showscale=False, margin=dict(t=20, b=20, l=220, r=100),
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig.update_traces(textposition="outside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig, use_container_width=True, key="fail_cat")

    # Chart 2
    st.markdown('<div class="section-header">Failure Treemap by Tier</div>', unsafe_allow_html=True)
    df2 = run_query(f"""
        WITH ranked AS (
            SELECT {TIER_SQL} as tier, failure_category, COUNT(*) as count,
                   ROW_NUMBER() OVER (PARTITION BY {TIER_SQL} ORDER BY COUNT(*) DESC) as rn
            FROM {TABLE} WHERE failure_category IS NOT NULL AND failure_category != 'Unknown'
              AND github_stars IS NOT NULL GROUP BY tier, failure_category
        ) SELECT tier, failure_category, count FROM ranked WHERE rn <= 5
    """)
    total = df2["count"].sum()
    df2["pct"] = (df2["count"] / total * 100).round(1).astype(str) + "%"
    df2["label"] = df2["failure_category"] + "<br>" + df2["count"].astype(str) + " (" + df2["pct"] + ")"
    fig2 = px.treemap(df2, path=["tier","failure_category"], values="count",
                      color="count", color_continuous_scale="Turbo", custom_data=["label"])
    fig2.update_traces(texttemplate="%{customdata[0]}", textfont=dict(size=FS, color="white"), textposition="middle center")
    fig2.update_layout(paper_bgcolor="#16213e", font=dict(size=FS, color="white"),
                       height=700, coloraxis_showscale=False, uniformtext=dict(minsize=13, mode="hide"))
    st.plotly_chart(fig2, use_container_width=True, key="fail_tree")

    # Chart 3
    st.markdown('<div class="section-header">Deployment Patterns Across Tiers</div>', unsafe_allow_html=True)
    df3 = run_query(f"""SELECT {TIER_SQL} as tier, deployment_target, COUNT(*) as count
        FROM {TABLE} WHERE {tw} AND deployment_target IS NOT NULL AND deployment_target != 'Unknown'
          AND github_stars IS NOT NULL GROUP BY tier, deployment_target""")
    fig3 = px.bar(df3, x="tier", y="count", color="deployment_target", barmode="stack",
                  color_discrete_sequence=px.colors.qualitative.Bold, text="count")
    apply_font(fig3, bg="#16213e", height=520, extra=dict(
        xaxis_title="", yaxis_title="Count",
        legend=dict(font=dict(size=FS, color="white"))))
    fig3.update_traces(textposition="inside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig3, use_container_width=True, key="deploy")

# ─────────────────────────────────────────────
# PAGE 7 — GLOBAL MAP
# ─────────────────────────────────────────────
elif page == "🌍 Global Map":
    st.title("🌍 Global Distribution of AI Projects")
    st.markdown("---")
    df = run_query(f"""SELECT country, COUNT(*) as project_count,
               ROUND(AVG(score_gemini)::numeric,1) as avg_score,
               ROUND(AVG(github_stars)::numeric,0) as avg_stars
        FROM {TABLE} WHERE {tw} AND country IS NOT NULL AND country != 'Unknown'
        GROUP BY country ORDER BY project_count DESC""")
    fig = px.choropleth(df, locations="country", locationmode="country names",
                        color="project_count", hover_name="country",
                        hover_data={"avg_score":True,"avg_stars":True,"project_count":True},
                        color_continuous_scale="Viridis")
    fig.update_layout(paper_bgcolor="#1e2130", font=dict(size=FS, color="white"),
                      geo=dict(bgcolor="#1e2130", showframe=False), height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Top 20 Countries</div>', unsafe_allow_html=True)
    fig2 = px.bar(df.head(20), x="country", y="project_count",
                  color="avg_score", color_continuous_scale="Viridis", text="project_count")
    apply_font(fig2, top=80, extra=dict(
        xaxis_title="", yaxis_title="Projects",
        yaxis=dict(range=[0, df.head(20)["project_count"].max()*1.18], tickfont=dict(size=FS,color="white"), title_font=dict(size=FS,color="white"))))
    fig2.update_traces(textposition="outside", textfont=dict(size=FS, color="white"))
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**ABB Tech Academy**")
st.sidebar.markdown("AI Research Knowledge Engine")
st.sidebar.markdown("Dataset: 18,447 papers with code")
