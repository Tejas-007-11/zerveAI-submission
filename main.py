from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_URL = "https://github.com/Tejas-007-11/zerve-datasets/raw/refs/heads/main/successlens_user_scores.csv"
INDIGO = "#818cf8"
TEAL   = "#34d399"
AMBER  = "#fbbf24"
RED    = "#f87171"
PALETTE = [INDIGO, TEAL, AMBER, RED, "#a78bfa", "#fb923c"]

# ── Data ──────────────────────────────────────────────────────────────────────
_df_cache = None

def load_data():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_URL)
    return _df_cache

# ── Chart helpers ─────────────────────────────────────────────────────────────
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

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── API: Platform Overview ────────────────────────────────────────────────────
@app.route("/api/overview")
def api_overview():
    df = load_data()
    seg_order = ["High", "Medium", "Low"]
    seg_colors = [TEAL, AMBER, RED]
    seg_counts = df["success_segment"].value_counts().reindex(seg_order).fillna(0)

    # Chart 1 – histogram
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df["success_probability"], bins=40, ax=ax1,
                 color=INDIGO, alpha=0.85, edgecolor="none")
    ax1.set_xlabel("Success Probability")
    ax1.set_ylabel("Users")
    apply_chart_style(fig1, ax1)
    hist_img = fig_to_base64(fig1)

    # Chart 2 – bar
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    bars = ax2.bar(seg_counts.index, seg_counts.values,
                   color=seg_colors, width=0.5, zorder=3)
    for bar, val in zip(bars, seg_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{int(val):,}", ha="center", va="bottom",
                 fontsize=9, color="#94a3b8")
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Users")
    apply_chart_style(fig2, ax2)
    seg_img = fig_to_base64(fig2)

    return jsonify({
        "total_users":       len(df),
        "avg_success":       round(float(df["success_probability"].mean()), 3),
        "high_users":        int((df["success_segment"] == "High").sum()),
        "low_users":         int((df["success_segment"] == "Low").sum()),
        "hist_chart":        hist_img,
        "seg_chart":         seg_img,
    })

# ── API: User Segmentation ────────────────────────────────────────────────────
@app.route("/api/segmentation")
def api_segmentation():
    df = load_data()
    segment = request.args.get("segment", "All")
    filtered = df if segment == "All" else df[df["success_segment"] == segment]
    records = filtered.head(100).to_dict(orient="records")
    columns = list(filtered.columns)
    return jsonify({
        "count":   len(filtered),
        "columns": columns,
        "rows":    records,
    })

# ── API: Behavior Insights ────────────────────────────────────────────────────
@app.route("/api/insights")
def api_insights():
    df = load_data()
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    # Heatmap
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", ax=ax3, linewidths=0.4,
                linecolor=(1, 1, 1, 0.05), annot=False,
                cbar_kws={"shrink": 0.75, "pad": 0.02})
    ax3.tick_params(colors="#64748b", labelsize=8)
    fig3.patch.set_facecolor("#0f1621")
    ax3.set_facecolor("#0f1621")
    fig3.tight_layout()
    heatmap_img = fig_to_base64(fig3)

    # Top features bar
    success_corr = corr["success_probability"].sort_values(ascending=False)[1:11]
    colors_bar = [TEAL if v > 0 else RED for v in success_corr.values]
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.barh(success_corr.index, success_corr.values,
             color=colors_bar, height=0.55, zorder=3)
    ax4.axvline(0, color=(1, 1, 1, 0.15), linewidth=0.8)
    ax4.set_xlabel("Correlation")
    apply_chart_style(fig4, ax4)
    topfeat_img = fig_to_base64(fig4)

    return jsonify({
        "heatmap_chart":  heatmap_img,
        "topfeat_chart":  topfeat_img,
    })

# ── API: User Explorer ────────────────────────────────────────────────────────
@app.route("/api/user")
def api_user():
    df = load_data()
    user_id = request.args.get("id", "").strip()
    if not user_id:
        return jsonify({"found": False, "error": "No user ID provided."})
    result = df[df["distinct_id"] == user_id]
    if len(result) == 0:
        return jsonify({"found": False, "error": "No user found with that ID."})
    row = result.iloc[0]
    return jsonify({
        "found":       True,
        "prob":        round(float(row["success_probability"]), 3),
        "segment":     str(row["success_segment"]),
        "user_id":     user_id,
        "columns":     list(result.columns),
        "rows":        result.to_dict(orient="records"),
    })

# ── API: Success Simulator ────────────────────────────────────────────────────
@app.route("/api/simulator")
def api_simulator():
    events   = int(request.args.get("events",   50))
    sessions = int(request.args.get("sessions", 10))
    features = int(request.args.get("features",  5))

    score = min((events * 0.002) + (sessions * 0.01) + (features * 0.02), 1.0)

    if score > 0.7:
        label, pill_cls = "High Success",   "pill-high"
    elif score > 0.3:
        label, pill_cls = "Medium Success", "pill-medium"
    else:
        label, pill_cls = "Low Success",    "pill-low"

    bar_color = TEAL if score > 0.7 else (AMBER if score > 0.3 else RED)

    breakdown = {
        "Events":   round(events   * 0.002, 4),
        "Sessions": round(sessions * 0.01,  4),
        "Features": round(features * 0.02,  4),
    }

    fig5, ax5 = plt.subplots(figsize=(6, 2.5))
    ax5.barh(list(breakdown.keys()), list(breakdown.values()),
             color=[INDIGO, TEAL, AMBER], height=0.45, zorder=3)
    ax5.set_xlabel("Score Contribution")
    max_v = max(breakdown.values())
    ax5.set_xlim(0, max_v * 1.35 if max_v > 0 else 1)
    for i, (k, v) in enumerate(breakdown.items()):
        ax5.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9, color="#94a3b8")
    apply_chart_style(fig5, ax5)
    breakdown_img = fig_to_base64(fig5)

    return jsonify({
        "score":         round(score, 3),
        "label":         label,
        "pill_cls":      pill_cls,
        "breakdown_img": breakdown_img,
    })

if __name__ == "__main__":
    app.run(debug=True)