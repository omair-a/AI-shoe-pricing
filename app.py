"""
╔══════════════════════════════════════════════════════════╗
║         AI SHOE PRICING SYSTEM — AFS Retail              ║
║         Streamlit + XGBoost + Groq API                   ║
╚══════════════════════════════════════════════════════════╝

Run:
    pip install streamlit xgboost scikit-learn pandas numpy groq plotly
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ------------------- Page Configuration ---------------------

st.set_page_config(
    page_title="Shoe Pricing Intelligence",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- CSS and Styling ---------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e4dc;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .stMarkdown { color: #9e9ab5; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #4ecdc4);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Select boxes and inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #13131f;
    border-color: #1e1e2e;
    color: #e8e4dc;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6e6a82;
    font-family: 'Syne', sans-serif;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: #1e1e2e !important;
    color: #e8e4dc !important;
}

/* AI recommendation box */
.ai-rec-box {
    background: #1e1e35;
    border: 1px solid #4a4a7a;
    border-left: 4px solid #6c63ff;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 8px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #f0eeff;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0f0f1a 0%, #13131f 50%, #0f0f1a 100%);
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}

/* Status pill */
.pill-green { background: #0d2e1a; color: #4ade80; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.pill-red   { background: #2e0d0d; color: #f87171; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.pill-amber { background: #2e1f0d; color: #fb923c; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }

div[data-testid="stExpander"] {
    background: #16162a;
    border: 1px solid #2e2e50;
    border-radius: 10px;
    color: #e8e4dc;
}

div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] span {
    color: #d4d0e8 !important;
}

div[data-testid="stExpander"] strong {
    color: #f0eeff !important;
}

div[data-testid="stExpander"] code {
    background: #2a2a45;
    color: #a78bfa;
    padding: 1px 5px;
    border-radius: 4px;
}

hr { border-color: #1e1e2e; }
</style>
""", unsafe_allow_html=True)


#  ---------------------- Data Loading and Pre-processing ----------------------

@st.cache_data
def load_data(path="shoe_pricing_data_v2.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["competitor_price", "our_cost", "min_margin", "our_current_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["price_gap"] = df["competitor_price"] - df["our_current_price"]
    df["current_margin"] = (df["our_current_price"] - df["our_cost"]) / df["our_current_price"]
    df["min_allowed_price"] = df["our_cost"] / (1 - df["min_margin"])
    df["month"] = df["date"].dt.month.fillna(1).astype(int)
    return df


#  ------------------- ML Model: XGBoost --------------------

@st.cache_resource
def train_model(df):
    cat_cols = ["brand", "category", "color", "material", "store", "gender", "season", "region"]
    encoders = {}
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    df_enc["optimal_price"] = df_enc.apply(
        lambda r: max(
            r["min_allowed_price"],
            min(r["competitor_price"], r["competitor_price"] * 0.97)
        ), axis=1
    )

    feature_cols = [c + "_enc" for c in cat_cols] + [
        "competitor_price", "our_cost", "min_margin", "month"
    ]

    X = df_enc[feature_cols].dropna()
    y = df_enc.loc[X.index, "optimal_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, encoders, feature_cols, mae, r2


def predict_optimal(df, model, encoders, feature_cols):
    df_enc = df.copy()
    cat_cols = ["brand", "category", "color", "material", "store", "gender", "season", "region"]
    for col in cat_cols:
        le = encoders[col]
        df_enc[col + "_enc"] = df_enc[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    X = df_enc[feature_cols].fillna(0)
    df["optimal_price"] = model.predict(X).round(2)
    df["price_adjustment"] = (df["optimal_price"] - df["our_current_price"]).round(2)
    df["adj_pct"] = ((df["price_adjustment"] / df["our_current_price"]) * 100).round(1)

    def tag(row):
        if abs(row["adj_pct"]) < 2:
            return "Maintain"
        elif row["price_adjustment"] > 0:
            return "Increase"
        else:
            return "Reduce"
    df["recommendation_tag"] = df.apply(tag, axis=1)
    return df


# ------------------ Groq API: Natural Language Recommendations ----------------------

def get_groq_recommendation(row: dict, api_key: str) -> str:
    client = Groq(api_key=api_key)
    prompt = f"""You are an expert retail pricing strategist. Analyze this shoe product and give a sharp, 2-3 sentence pricing recommendation.

Product: {row['product_name']}
Brand: {row['brand']} | Category: {row['category']} | Gender: {row.get('gender', 'N/A')} | Season: {row.get('season', 'N/A')}
Region: {row.get('region', 'N/A')} | Store: {row['store']}
Our Current Price: ${row['our_current_price']:.2f}
Competitor Price: ${row['competitor_price']:.2f}
Our Cost: ${row['our_cost']:.2f}
Min Required Margin: {row['min_margin']*100:.0f}%
ML-Suggested Optimal Price: ${row.get('optimal_price', 0):.2f}
Price Gap vs Competitor: ${row['price_gap']:.2f}
Current Margin: {row['current_margin']*100:.1f}%

Be direct and actionable. Mention specific dollar amounts. Factor in the gender segment, season, and regional context where relevant. Focus on profit opportunity or competitive risk."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


#  -------------------------- Sidebar --------------------------

with st.sidebar:
    st.markdown("## 👟 Shoe Pricing")
    st.markdown("**AI Pricing Intelligence**")
    st.divider()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()

    st.markdown("### Filters")
    df_raw = load_data() if not uploaded else load_data(uploaded)

    selected_brands = st.multiselect("Brand", sorted(df_raw["brand"].unique()),
                                      default=list(df_raw["brand"].unique()))
    selected_cats = st.multiselect("Category", sorted(df_raw["category"].unique()),
                                    default=list(df_raw["category"].unique()))
    selected_genders = st.multiselect("Gender", sorted(df_raw["gender"].unique()),
                                       default=list(df_raw["gender"].unique()))
    selected_seasons = st.multiselect("Season", sorted(df_raw["season"].unique()),
                                       default=list(df_raw["season"].unique()))
    selected_regions = st.multiselect("Region", sorted(df_raw["region"].unique()),
                                       default=list(df_raw["region"].unique()))
    selected_stores = st.multiselect("Store", sorted(df_raw["store"].unique()),
                                      default=list(df_raw["store"].unique()))
    st.divider()
    st.caption("Built for Shoe Retail")


# ------------------ Load and Filter Data ---------------------

df = df_raw.copy()
if selected_brands:
    df = df[df["brand"].isin(selected_brands)]
if selected_cats:
    df = df[df["category"].isin(selected_cats)]
if selected_genders:
    df = df[df["gender"].isin(selected_genders)]
if selected_seasons:
    df = df[df["season"].isin(selected_seasons)]
if selected_regions:
    df = df[df["region"].isin(selected_regions)]
if selected_stores:
    df = df[df["store"].isin(selected_stores)]

model, encoders, feature_cols, mae, r2 = train_model(df_raw)
df = predict_optimal(df, model, encoders, feature_cols)

# Load Groq API key from Streamlit secrets (deployed) or environment variable (local)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("GROQ_API_KEY", "")


#  ----------------------- Header ---------------------------

st.markdown("""
<div class="header-banner">
  <h1 style="margin:0; font-size:2rem; font-family:'Syne',sans-serif;">
    👟 Shoe Pricing Intelligence
  </h1>
  <p style="margin:6px 0 0; color:#6e6a82; font-size:0.95rem;">
    XGBoost Model Price Predictions · AI Recommendations by Groq · Live Competitor Analysis
  </p>
</div>
""", unsafe_allow_html=True)


# --------------------- KPI Metrics --------------------------

total = len(df)
increase_count = (df["recommendation_tag"] == "Increase").sum()
reduce_count   = (df["recommendation_tag"] == "Reduce").sum()
maintain_count = (df["recommendation_tag"] == "Maintain").sum()
avg_gap        = df["price_gap"].mean()
avg_margin     = df["current_margin"].mean() * 100
revenue_opportunity = (df["price_adjustment"].clip(lower=0)).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Products Analyzed", f"{total:,}")
c2.metric("Increase Price", f"{increase_count}", help="Products priced below optimal")
c3.metric("Reduce Price", f"{reduce_count}", help="Products priced above competitor")
c4.metric("Average Competitor Gap", f"{'+' if avg_gap >= 0 else '-'}${abs(avg_gap):.2f}")
c5.metric("Upside Opportunity", f"${revenue_opportunity:,.0f}", help="Sum of positive price adjustments")

st.markdown("")


#  ------------------- Tabs ------------------------

tab1, tab2, tab3, tab4 = st.tabs([" 📋 Pricing Table ", " 🤖 AI Recommendations ", " 📊 Analytics ", " 🔬 Model Info "])


# ---------------- Tab 1: Pricing Table ---------------------

with tab1:
    st.markdown("### Optimal Price Recommendations")
    st.caption(f"Showing {len(df):,} products · Sorted by largest price adjustment · All prices in USD")

    def style_tag(val):
        if val == "Increase":  return "background-color:#0d2e1a; color:#4ade80"
        if val == "Reduce":    return "background-color:#2e0d0d; color:#f87171"
        return "background-color:#1a1a2e; color:#94a3b8"

    def fmt_adjustment(v):
        if v >= 0:
            return f"+ ${v:.2f}"
        else:
            return f"- ${abs(v):.2f}"

    display_cols = ["sku", "product_name", "brand", "category", "gender", "season", "region", "store",
                    "our_current_price", "competitor_price", "optimal_price",
                    "price_adjustment", "adj_pct", "current_margin", "recommendation_tag"]

    display_df = df[display_cols].copy()
    display_df["current_margin"] = (display_df["current_margin"] * 100).round(1)
    display_df.columns = ["SKU", "Product", "Brand", "Category", "Gender", "Season", "Region", "Store",
                           "Current Price", "Competitor Price", "Optimal Price",
                           "Adjustment", "Change %", "Cur. Margin %", "Action"]
    display_df = display_df.sort_values("Adjustment", key=abs, ascending=False)

    styled = display_df.style\
        .map(style_tag, subset=["Action"])\
        .format({
            "Current Price":    "${:.2f}",
            "Competitor Price": "${:.2f}",
            "Optimal Price":    "${:.2f}",
            "Adjustment":   fmt_adjustment,
            "Change %":        "{:+.1f}%",
            "Cur. Margin %":     "{:.1f}%",
        })\
        .background_gradient(subset=["Adjustment"], cmap="RdYlGn", vmin=-10, vmax=10)

    st.dataframe(styled, use_container_width=True, height=500, hide_index=True)

    csv_dl = display_df.to_csv(index=False).encode()
    st.download_button("⬇ Download as CSV", csv_dl, "pricing_recommendations.csv", "text/csv")


# -------------------- Tab 2: AI Recommendations -----------------------

with tab2:
    st.markdown("### 🤖 Groq AI Recommendations")
    st.caption("Select a product to get a natural language pricing recommendation powered by Groq (Llama 3.3 70B)")

    if not api_key:
        st.error("Groq API key not configured. Please add GROQ_API_KEY to your Streamlit secrets or environment variables.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            product_options = df["product_name"].tolist()
            selected_product = st.selectbox("Select Product", product_options)
        with col_b:
            batch_n = st.number_input("Or batch analyze top N products", min_value=1, max_value=20, value=5)

        row_data = df[df["product_name"] == selected_product].iloc[0].to_dict()

        # ── Product Card ──────────────────────────────────────────
        action     = row_data["recommendation_tag"]
        action_color = "#4ade80" if action == "Increase" else "#f87171" if action == "Reduce" else "#94a3b8"
        action_bg    = "#0d2e1a" if action == "Increase" else "#2e0d0d" if action == "Reduce" else "#1a1a2e"

        # Pick an icon based on category
        category_icons = {
            "Running": "🏃", "Basketball": "🏀", "Casual": "👟",
            "Training": "💪", "Hiking": "🥾", "Sandals": "👡",
            "Boots": "🥿", "Lifestyle": "✨",
        }
        cat_icon = category_icons.get(row_data.get("category", ""), "👟")

        # Price bar — show where current sits vs competitor and optimal
        cur  = row_data["our_current_price"]
        comp = row_data["competitor_price"]
        opt  = row_data.get("optimal_price", cur)
        bar_min = min(cur, comp, opt) * 0.90
        bar_max = max(cur, comp, opt) * 1.10
        bar_range = bar_max - bar_min if bar_max != bar_min else 1

        def pct(val):
            return round(((val - bar_min) / bar_range) * 100, 1)

        cur_pct  = pct(cur)
        comp_pct = pct(comp)
        opt_pct  = pct(opt)

        adj      = row_data.get("price_adjustment", 0)
        adj_fmt  = f"+ ${adj:.2f}" if adj >= 0 else f"- ${abs(adj):.2f}"
        margin   = row_data.get("current_margin", 0) * 100

        action       = row_data["recommendation_tag"]
        action_color = "#4ade80" if action == "Increase" else "#f87171" if action == "Reduce" else "#94a3b8"
        action_bg    = "#0d2e1a" if action == "Increase" else "#2e0d0d" if action == "Reduce" else "#1a1a2e"

        category_icons = {
            "Running": "🏃", "Basketball": "🏀", "Casual": "👟",
            "Training": "💪", "Hiking": "🥾", "Sandals": "👡",
            "Boots": "🥿", "Lifestyle": "✨",
        }
        cat_icon = category_icons.get(row_data.get("category", ""), "👟")

        cur  = row_data["our_current_price"]
        comp = row_data["competitor_price"]
        opt  = row_data.get("optimal_price", cur)
        adj  = row_data.get("price_adjustment", 0)
        adj_fmt = f"+ ${adj:.2f}" if adj >= 0 else f"- ${abs(adj):.2f}"
        margin  = row_data.get("current_margin", 0) * 100

        bar_min   = min(cur, comp, opt) * 0.90
        bar_max   = max(cur, comp, opt) * 1.10
        bar_range = bar_max - bar_min if bar_max != bar_min else 1
        cur_pct   = round(((cur  - bar_min) / bar_range) * 100, 1)
        comp_pct  = round(((comp - bar_min) / bar_range) * 100, 1)
        opt_pct   = round(((opt  - bar_min) / bar_range) * 100, 1)

        tags = "".join([
            f'<span style="background:#1e1e35;border:1px solid #2a2a45;color:#c4c0d4;'
            f'border-radius:20px;padding:2px 10px;font-size:0.75rem;margin:2px;">{v}</span>'
            for v in [
                row_data.get("brand",""), row_data.get("category",""),
                row_data.get("gender",""), row_data.get("season",""),
                row_data.get("region",""), row_data.get("material",""),
                row_data.get("color",""), f'Size {row_data.get("size","")}'
            ] if v and v != "Size "
        ])

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#13131f,#1a1a2e);border:1px solid #2a2a45;border-radius:16px;padding:24px 28px;margin-bottom:20px;">
  <div style="display:flex;align-items:flex-start;gap:20px;margin-bottom:20px;">
    <div style="background:#1e1e35;border:1px solid #4a4a7a;border-radius:14px;width:80px;height:80px;display:flex;align-items:center;justify-content:center;font-size:2.2rem;flex-shrink:0;">{cat_icon}</div>
    <div style="flex:1;min-width:0;">
      <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:#f0eeff;margin-bottom:6px;">{row_data["product_name"]}</div>
      <div style="font-size:0.78rem;color:#9e9ab5;margin-bottom:10px;">SKU: <strong style="color:#a78bfa">{row_data.get("sku","N/A")}</strong></div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">{tags}</div>
    </div>
    <div style="background:{action_bg};border:1px solid {action_color};color:{action_color};border-radius:10px;padding:8px 16px;font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;text-align:center;flex-shrink:0;">{action.upper()}<br><span style="font-size:1rem;">{adj_fmt}</span></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;">
    <div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:10px;padding:12px 14px;">
      <div style="color:#6e6a82;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Current Price</div>
      <div style="color:#e8e4dc;font-size:1.2rem;font-weight:700;">${cur:.2f}</div>
    </div>
    <div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:10px;padding:12px 14px;">
      <div style="color:#6e6a82;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Competitor Price</div>
      <div style="color:#e8e4dc;font-size:1.2rem;font-weight:700;">${comp:.2f}</div>
    </div>
    <div style="background:#0f0f1a;border:1px solid {action_color}44;border-radius:10px;padding:12px 14px;">
      <div style="color:#6e6a82;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Optimal Price</div>
      <div style="color:{action_color};font-size:1.2rem;font-weight:700;">${opt:.2f}</div>
    </div>
    <div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:10px;padding:12px 14px;">
      <div style="color:#6e6a82;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Current Margin</div>
      <div style="color:#e8e4dc;font-size:1.2rem;font-weight:700;">{margin:.1f}%</div>
    </div>
  </div>
  <div>
    <div style="color:#6e6a82;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Price Comparison</div>
    <div style="position:relative;height:10px;background:#1e1e2e;border-radius:10px;margin-bottom:14px;">
      <div style="position:absolute;left:{comp_pct}%;transform:translateX(-50%);width:3px;height:10px;background:#f87171;border-radius:2px;"></div>
      <div style="position:absolute;left:{opt_pct}%;transform:translateX(-50%);width:3px;height:10px;background:{action_color};border-radius:2px;"></div>
      <div style="position:absolute;left:{cur_pct}%;transform:translateX(-50%);width:14px;height:14px;background:#6c63ff;border:2px solid #e8e4dc;border-radius:50%;top:-2px;"></div>
    </div>
    <div style="display:flex;gap:16px;font-size:0.72rem;color:#6e6a82;">
      <span><span style="color:#6c63ff;">&#9679;</span> Our Price ${cur:.2f}</span>
      <span><span style="color:#f87171;">&#9646;</span> Competitor ${comp:.2f}</span>
      <span><span style="color:{action_color};">&#9646;</span> Optimal ${opt:.2f}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)



        if st.button("🤖 Get AI Recommendation"):
            with st.spinner("Groq is analyzing..."):
                try:
                    rec = get_groq_recommendation(row_data, api_key)
                    st.markdown(f'<div class="ai-rec-box">💡 {rec}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API error: {e}")

        st.divider()
        if st.button(f"⚡ Batch Analyze Top {batch_n} High-Impact Products"):
            top_products = df.nlargest(batch_n, "price_adjustment", keep="all")
            progress = st.progress(0)
            results = []
            for i, (_, row) in enumerate(top_products.iterrows()):
                with st.spinner(f"Analyzing {row['product_name']}..."):
                    try:
                        rec = get_groq_recommendation(row.to_dict(), api_key)
                        results.append((row["product_name"], row["recommendation_tag"], rec))
                    except Exception as e:
                        results.append((row["product_name"], "Error", str(e)))
                progress.progress((i + 1) / batch_n)

            for name, tag, rec in results:
                tag_color = "#4ade80" if tag == "Increase" else "#f87171" if tag == "Reduce" else "#94a3b8"
                st.markdown(f"""
                <div class="ai-rec-box">
                  <strong style="color:{tag_color}">[{tag}]</strong> <strong>{name}</strong><br>
                  <span style="color:#e8e4dc">{rec}</span>
                </div>""", unsafe_allow_html=True)



# ------------------ Tab 3: Analytics -----------------------

with tab3:
    st.markdown("### 📊 Pricing Analytics")

    plot_bg    = "#0f0f1a"
    plot_paper = "#0a0a0f"
    font_color = "#e8e4dc"
    grid_color = "#1e1e2e"
    title_font = dict(color="#e8e4dc", size=14, family="Syne")

    def style_fig(fig):
        fig.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_paper,
            font=dict(color=font_color, family="DM Sans"),
            title_font=title_font,
            xaxis=dict(gridcolor=grid_color, zeroline=False),
            yaxis=dict(gridcolor=grid_color, zeroline=False),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        tag_counts = df["recommendation_tag"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=tag_counts.index,
            values=tag_counts.values,
            hole=0.55,
            marker_colors=["#4ade80", "#f87171", "#6c63ff"],
            textinfo="label+percent",
            textfont=dict(color=font_color),
        ))
        fig_donut.update_layout(
            title=dict(text="Recommendation Distribution", font=title_font),
            plot_bgcolor=plot_bg, paper_bgcolor=plot_paper,
            font=dict(color=font_color),
            showlegend=False, margin=dict(t=50, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_c2:
        gap_by_brand = df.groupby("brand")["price_gap"].mean().sort_values()
        colors = ["#f87171" if v < 0 else "#4ade80" for v in gap_by_brand.values]
        fig_bar = go.Figure(go.Bar(
            x=gap_by_brand.values,
            y=gap_by_brand.index,
            orientation="h",
            marker_color=colors,
        ))
        fig_bar.update_layout(
            title=dict(text="Avg. Competitor Price Gap by Brand", font=title_font),
            plot_bgcolor=plot_bg, paper_bgcolor=plot_paper,
            font=dict(color=font_color),
            xaxis=dict(gridcolor=grid_color, zeroline=True, zerolinecolor="#2a2a45"),
            yaxis=dict(gridcolor=grid_color),
            margin=dict(t=50, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        fig_scatter = px.scatter(
            df, x="our_current_price", y="competitor_price",
            color="recommendation_tag",
            color_discrete_map={"Increase": "#4ade80", "Reduce": "#f87171", "Maintain": "#6c63ff"},
            hover_data=["product_name", "brand", "store"],
            labels={"our_current_price": "Our Price ($)", "competitor_price": "Competitor Price ($)"},
            title="Our Price vs. Competitor Price"
        )
        max_val = max(df["our_current_price"].max(), df["competitor_price"].max())
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                               line=dict(dash="dash", color="#a78bfa", width=1.5))
        style_fig(fig_scatter)
        fig_scatter.update_layout(legend=dict(
            font=dict(color="#e8e4dc", size=12),
            bgcolor="#1e1e35",
            bordercolor="#4a4a7a",
            borderwidth=1,
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_c2:
        fig_box = px.box(
            df, x="category", y="current_margin",
            color="category",
            title="Margin Distribution by Category",
            labels={"current_margin": "Current Margin", "category": "Category"}
        )
        style_fig(fig_box)
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Price Adjustment Heatmap: Brand × Category")
    pivot = df.pivot_table(values="price_adjustment", index="brand", columns="category", aggfunc="mean").fillna(0)
    fig_heat = px.imshow(
        pivot, color_continuous_scale=["#f87171", "#0a0a0f", "#4ade80"],
        zmin=-10, zmax=10,
        labels=dict(color="Avg Adj $"),
        aspect="auto"
    )
    fig_heat.update_layout(
        plot_bgcolor=plot_bg, paper_bgcolor=plot_paper,
        font=dict(color=font_color, family="DM Sans"),
        margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


#  Tab 4: Model Info

with tab4:
    st.markdown("### 🔬 Model Details")
    st.caption("How the AI analyzes shoe data to calculate optimal pricing recommendations")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Algorithm", "XGBoost Regressor")
    col_m2.metric("MAE (test set)", f"${mae:.2f}", help="Average dollar difference between predicted and actual optimal price on unseen data")
    col_m3.metric("R² Score", f"{r2:.3f}", help="How well the model explains price variation. Closer to 1.0 is better.")

    st.markdown("")

    # ── How the AI calculates optimal price ──────────────────────────
    with st.expander("🧠 How the AI Calculates the Optimal Price"):
        st.markdown("""
The AI calculates the optimal price using a supervised machine learning model built with XGBoost, a gradient boosting algorithm designed for structured 
data. Instead of relying on fixed pricing rules, the model learns patterns from historical product data to predict what price works best under different 
conditions.

The target variable, optimal price, is calculated using 
business constraints. Specifically, the target price is set to stay competitive with the market (typically at or slightly below the competitor price) while also 
respecting a minimum margin requirement to ensure profitability. This creates a realistic benchmark for the model to learn from.

The model is trained on a combination of categorical and numerical features. Categorical variables such as brand, category, gender, season, region, material, and store are encoded into numerical form. Numerical features include competitor price, our cost, minimum margin, and time. Together, these inputs give the model full context about both the product and the market environment.

During training, XGBoost builds an ensemble of decision trees that improve prediction accuracy by minimizing error. It learns complex relationships between features—for example, how pricing power varies by brand, how demand shifts across seasons, or how regional differences impact acceptable price ranges.

Once trained, the model can take a new product as input and predict its optimal price based on these learned patterns. The output is a data-driven price recommendation, along with the suggested adjustment from the current price. This allows the system to identify whether a product should be increased, reduced, or maintained.

        """)


    # ── Side-by-side example ──────────────────────────────────────────
    with st.expander("📊 Feature Importance"):
        st.markdown("""
The chart below shows which features the model relies on most when generating a recommendation.
A higher importance score means that feature has more influence over the predicted optimal price.
        """)

        fi = pd.Series(
            model.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=True)

        # Clean up feature names for display
        clean_labels = []
        label_map = {
            "brand_enc": "Brand", "category_enc": "Category", "color_enc": "Color",
            "material_enc": "Material", "store_enc": "Store", "gender_enc": "Gender",
            "season_enc": "Season", "region_enc": "Region",
            "competitor_price": "Competitor Price", "our_cost": "Our Cost",
            "min_margin": "Min Margin", "month": "Month",
        }
        clean_labels = [label_map.get(f, f) for f in fi.index]

        bar_colors = ["#4ade80" if v >= fi.values.max() * 0.6
                      else "#6c63ff" if v >= fi.values.max() * 0.3
                      else "#4a4a7a" for v in fi.values]

        fig_fi = go.Figure(go.Bar(
            x=fi.values,
            y=clean_labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in fi.values],
            textposition="outside",
            textfont=dict(color="#9e9ab5", size=11),
        ))
        fig_fi.update_layout(
            plot_bgcolor="#0f0f1a", paper_bgcolor="#0a0a0f",
            font=dict(color="#e8e4dc", family="DM Sans"),
            xaxis=dict(gridcolor="#1e1e2e", title="Importance Score", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12)),
            margin=dict(t=10, b=10, l=10, r=60),
            height=380,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Training methodology ──────────────────────────────────────────
    with st.expander("⚙️ Training Methodology"):
        st.markdown("""
**How the training targets are generated**

To train the model, we first compute a reference optimal price for each product in the dataset.
This reference price is anchored to two real-world constraints:

- **Competitiveness:** the price should be positioned at or slightly below the competitor, making
  the product attractive to buyers without unnecessarily giving away margin
- **Profitability:** the price must never fall below the minimum margin floor — the point at which
  the business stops making money on the sale

These constraints produce a training target for every product. The model then learns to predict
these targets from the product features alone — without ever seeing the current price.

**Why exclude the current price?**

Including our current price as a feature would cause the model to anchor its recommendations
to whatever we're already charging. That defeats the purpose entirely — the model would just
echo the status quo rather than reasoning independently about what the market supports.

**Training configuration**

| Parameter | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Trees (estimators) | 200 |
| Max tree depth | 5 |
| Learning rate | 0.08 |
| Row subsampling | 85% per tree |
| Feature subsampling | 85% per tree |
| Train / test split | 80% / 20% |
| Random seed | 42 (reproducible) |
        """)

    # ── Groq integration ─────────────────────────────────────────────
    with st.expander("🤖 Groq AI Integration"):
        st.markdown("""
Once the ML model produces an optimal price, the system passes the full product context to
**Groq's LLaMA 3.3 70B** — a state-of-the-art large language model — which generates a
natural language explanation of the recommendation.

The AI receives the product's brand, category, gender, season, region, store, current price,
competitor price, cost, margin, and ML-predicted optimal price, and returns a 2-3 sentence
analysis that explains *why* the adjustment makes sense in plain English.

This two-layer approach combines the precision of machine learning with the explainability
of generative AI — the model tells you *what* to do, and the AI tells you *why*.

**Model:** `llama-3.3-70b-versatile` via Groq (free tier)
        """)

    with st.expander("📋 Raw Data Sample"):
        st.dataframe(df_raw.head(20), use_container_width=True)