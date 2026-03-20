import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SuccessLens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0b0f1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(20,184,166,.12) 0%, transparent 60%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid rgba(255,255,255,.07);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.93rem;
    color: #94a3b8;
    padding: 6px 0;
    transition: color .2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0; }

/* ── Sidebar brand ── */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #818cf8 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.sidebar-tagline {
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 28px;
}

/* ── Page title ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em !important;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.1rem;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 0.88rem;
    color: #475569;
    margin-bottom: 32px;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: 22px 24px;
    transition: border-color .25s, transform .2s;
}
.metric-card:hover {
    border-color: rgba(129,140,248,.35);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-accent { color: #818cf8; }
.metric-green  { color: #34d399; }
.metric-red    { color: #f87171; }

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #cbd5e1;
    border-left: 3px solid #818cf8;
    padding-left: 12px;
    margin: 32px 0 16px;
}

/* ── Chart wrapper ── */
.chart-box {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px;
    padding: 20px;
}

/* ── Pill badge ── */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.pill-high   { background: rgba(52,211,153,.15); color: #34d399; border: 1px solid rgba(52,211,153,.3); }
.pill-medium { background: rgba(251,191,36,.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,.3); }
.pill-low    { background: rgba(248,113,113,.12); color: #f87171; border: 1px solid rgba(248,113,113,.3); }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSlider [data-baseweb="slider"] { padding-top: 8px; }

/* ── Progress bar override ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #818cf8, #34d399) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,.08) !important;
    border-radius: 999px !important;
}

/* ── Alert / info boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,.07) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ─────────────────────────────────────────────────────
def apply_chart_style(fig, ax):
    fig.patch.set_facecolor("#0f1621")
    ax.set_facecolor("#0f1621")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.xaxis.label.set_color("#64748b")
    ax.yaxis.label.set_color("#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.07))
        spine.set_linewidth(0.8)
    ax.grid(axis="y", color=(1, 1, 1, 0.05), linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()


INDIGO = "#818cf8"
TEAL   = "#34d399"
AMBER  = "#fbbf24"
RED    = "#f87171"
PALETTE = [INDIGO, TEAL, AMBER, RED, "#a78bfa", "#fb923c"]


# ── Data ─────────────────────────────────────────────────────────────────────
DATA_URL = "https://github.com/Tejas-007-11/zerve-datasets/raw/refs/heads/main/successlens_user_scores.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">SuccessLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">User Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Platform Overview", "User Segmentation", "Behavior Insights", "User Explorer", "Success Simulator"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.72rem;color:#334155;">Dataset · <b style="color:#475569">{len(df):,}</b> users</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Platform Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Platform Overview":
    st.markdown('<div class="page-title">Platform Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">High-level health of your user base</div>', unsafe_allow_html=True)

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Total Users",            f"{len(df):,}",                                  "metric-accent"),
        ("Avg Success Prob.",      f"{df['success_probability'].mean():.3f}",        "metric-accent"),
        ("High-Success Users",     f"{(df['success_segment']=='High').sum():,}",     "metric-green"),
        ("Low-Success Users",      f"{(df['success_segment']=='Low').sum():,}",      "metric-red"),
    ]
    for col, (label, value, cls) in zip([c1, c2, c3, c4], kpis):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Charts ──
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-header">Success Probability Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["success_probability"], bins=40, ax=ax, color=INDIGO, alpha=0.85, edgecolor="none")
        ax.set_xlabel("Success Probability")
        ax.set_ylabel("Users")
        apply_chart_style(fig, ax)
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">User Segments</div>', unsafe_allow_html=True)
        seg_order = ["High", "Medium", "Low"]
        seg_colors = [TEAL, AMBER, RED]
        seg_counts = df["success_segment"].value_counts().reindex(seg_order).fillna(0)

        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        bars = ax2.bar(seg_counts.index, seg_counts.values, color=seg_colors, width=0.5, zorder=3)
        for bar, val in zip(bars, seg_counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{int(val):,}", ha="center", va="bottom", fontsize=9, color="#94a3b8")
        ax2.set_xlabel("Segment")
        ax2.set_ylabel("Users")
        apply_chart_style(fig2, ax2)
        st.pyplot(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: User Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "User Segmentation":
    st.markdown('<div class="page-title">User Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Filter and explore users by success segment</div>', unsafe_allow_html=True)

    col_filter, col_count = st.columns([2, 1])
    with col_filter:
        segment_filter = st.selectbox("Segment", ["All", "High", "Medium", "Low"], label_visibility="visible")

    filtered = df if segment_filter == "All" else df[df["success_segment"] == segment_filter]

    pill_cls = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}.get(segment_filter, "pill-high")
    badge = f'<span class="pill {pill_cls}">{segment_filter}</span>' if segment_filter != "All" else ""

    st.markdown(
        f'<div style="margin: 12px 0 20px; color:#94a3b8; font-size:.9rem;">'
        f'Showing <b style="color:#e2e8f0">{len(filtered):,}</b> users {badge}</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Behavior Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Behavior Insights":
    st.markdown('<div class="page-title">Behavior Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Feature correlations and success drivers</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr, cmap="coolwarm", ax=ax3, linewidths=0.4,
        linecolor=(1, 1, 1, 0.05), annot=False,
        cbar_kws={"shrink": 0.75, "pad": 0.02},
    )
    ax3.tick_params(colors="#64748b", labelsize=8)
    fig3.patch.set_facecolor("#0f1621")
    ax3.set_facecolor("#0f1621")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    # Top features
    st.markdown('<div class="section-header">Top Features Correlated with Success</div>', unsafe_allow_html=True)
    success_corr = corr["success_probability"].sort_values(ascending=False)[1:11]

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    colors_bar = [TEAL if v > 0 else RED for v in success_corr.values]
    ax4.barh(success_corr.index, success_corr.values, color=colors_bar, height=0.55, zorder=3)
    ax4.axvline(0, color=(1, 1, 1, 0.15), linewidth=0.8)
    ax4.set_xlabel("Correlation")
    apply_chart_style(fig4, ax4)
    st.pyplot(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: User Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "User Explorer":
    st.markdown('<div class="page-title">User Success Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter a User ID to inspect individual metrics</div>', unsafe_allow_html=True)

    user_id = st.text_input("User ID", placeholder="e.g. usr_00123")

    if user_id:
        result = df[df["distinct_id"] == user_id]

        if len(result) > 0:
            prob = result["success_probability"].values[0]
            seg  = result["success_segment"].values[0]
            pill_cls = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}.get(seg, "pill-high")

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="metric-card"><div class="metric-label">Success Probability</div>'
                f'<div class="metric-value metric-accent">{prob:.3f}</div></div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="metric-card"><div class="metric-label">Segment</div>'
                f'<div style="margin-top:10px"><span class="pill {pill_cls}">{seg}</span></div></div>',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="metric-card"><div class="metric-label">User ID</div>'
                f'<div class="metric-value" style="font-size:1rem;color:#64748b">{user_id}</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-header">All Features</div>', unsafe_allow_html=True)
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning("No user found with that ID. Please double-check and try again.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Success Simulator
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Success Simulator":
    st.markdown('<div class="page-title">Success Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Estimate success probability from behavioral inputs</div>', unsafe_allow_html=True)

    st.markdown("")
    s1, s2 = st.columns(2, gap="large")

    with s1:
        events   = st.slider("Number of Events",        1, 500, 50)
        sessions = st.slider("Sessions",                1, 100, 10)
        features = st.slider("Unique Features Used",    1,  50,  5)

    score = min((events * 0.002) + (sessions * 0.01) + (features * 0.02), 1.0)

    if   score > 0.7: label, pill_cls, bar_color = "High Success",   "pill-high",   TEAL
    elif score > 0.3: label, pill_cls, bar_color = "Medium Success", "pill-medium", AMBER
    else:             label, pill_cls, bar_color = "Low Success",    "pill-low",    RED

    with s2:
        st.markdown(
            f'<div class="metric-card" style="margin-top:4px">'
            f'<div class="metric-label">Predicted Probability</div>'
            f'<div class="metric-value metric-accent" style="font-size:3rem;margin:8px 0">{score:.3f}</div>'
            f'<span class="pill {pill_cls}">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-header">Probability Gauge</div>', unsafe_allow_html=True)
    st.progress(score)

    # Mini breakdown chart
    st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
    breakdown = {
        "Events": events * 0.002,
        "Sessions": sessions * 0.01,
        "Features": features * 0.02,
    }
    fig5, ax5 = plt.subplots(figsize=(6, 2.5))
    ax5.barh(list(breakdown.keys()), list(breakdown.values()),
             color=[INDIGO, TEAL, AMBER], height=0.45, zorder=3)
    ax5.set_xlabel("Score Contribution")
    ax5.set_xlim(0, max(breakdown.values()) * 1.35)
    for i, (k, v) in enumerate(breakdown.items()):
        ax5.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, color="#94a3b8")
    apply_chart_style(fig5, ax5)
    st.pyplot(fig5, use_container_width=True)