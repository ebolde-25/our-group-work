# app.py  — African Natural Disaster Impact Predictor (XGBoost)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="African Natural Disaster Impact Predictor",
    layout="wide",
    page_icon="🌍"
)

# Minimal dark UI polish (adjust palette if you want exact colors)
st.markdown("""
<style>
:root { --panel:#121621; --card:#151a26; --stroke:#2a3042; --pill:#1b1f2a; }
[data-testid="stSidebar"]{background:#0f1116;border-right:1px solid var(--stroke);}
.sidebar-card{background:#1b1f2a;padding:14px;border-radius:16px;border:1px solid var(--stroke);}
.banner{background:var(--panel);border:1px solid var(--stroke);padding:14px 16px;border-radius:16px;}
.badge{display:inline-block;padding:6px 10px;border-radius:999px;background:var(--pill);
  border:1px solid var(--stroke);margin-right:6px;font-size:.85rem;}
.result-card{background:var(--card);border:1px solid var(--stroke);padding:20px;border-radius:20px;}
.result-big{font-size:2.6rem;font-weight:800;color:#ff6b6b;}
.summary{background:var(--panel);border:1px solid var(--stroke);padding:14px;border-radius:16px;}
.block-title{font-size:1.05rem;font-weight:700;margin:4px 0 10px 0;}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Artifacts
# =========================
@st.cache_resource
def load_artifacts():
    # All three PKLs must be in repo root
    model = joblib.load("xgb_model.pkl")                  # Pipeline(SimpleImputer + XGBRegressor)
    features = list(joblib.load("model_features_xgb.pkl"))# exact feature order
    y_info = joblib.load("y_transform_xgb.pkl")           # {"transform":"log1p","inverse":"expm1"}
    return model, features, y_info

model, MODEL_FEATURES, Y_INFO = load_artifacts()
FEATURE_SET = set(MODEL_FEATURES)

def inv_transform(yhat_log: np.ndarray) -> np.ndarray:
    if (Y_INFO or {}).get("inverse") == "expm1":
        return np.expm1(yhat_log)
    return yhat_log

# =========================
# Safe Feature Mapping
# =========================
def set_if_exists(d:dict, col:str, value):
    """Only set if feature exists in trained model."""
    if col in FEATURE_SET:
        d[col] = value

def build_feature_row(params: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame in the exact feature order the model expects.
    Any feature not explicitly set here is left at 0 (safe default).
    """
    x = {c: 0 for c in MODEL_FEATURES}

    # Core time features (common in your dataset)
    set_if_exists(x, "Year", params["year"])
    set_if_exists(x, "Start Year", params["year"])
    set_if_exists(x, "End Year", params["year"])
    set_if_exists(x, "Start Month", params["month"])
    set_if_exists(x, "End Month", params["month"])
    set_if_exists(x, "Seq", 1)

    # User “expected” priors (only applied if such columns exist)
    set_if_exists(x, "Total Deaths", params["exp_deaths"])
    set_if_exists(x, "No Injured", params["exp_injured"])
    set_if_exists(x, "No Homeless", params["exp_homeless"])

    # Region / Disaster one-hots — try common naming styles
    region = params["region"]; disaster = params["disaster"]
    for pat in [f"Region_{region}", f"Africa_Region_{region}", f"Region:{region}", region]:
        set_if_exists(x, pat, 1)
    for pat in [f"Disaster_{disaster}", f"Disaster Type_{disaster}", f"Disaster:{disaster}", disaster]:
        set_if_exists(x, pat, 1)

    # Economic/damage placeholders if present in your features
    for guess in [
        "Aid Contribution ('000 US$)",
        "Reconstruction Costs ('000 US$)",
        "Insured Damages ('000 US$)",
        "Total Damages ('000 US$)"
    ]:
        set_if_exists(x, guess, 0)

    return pd.DataFrame([[x[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)

def predict_one(params: dict) -> float:
    Xrow = build_feature_row(params)
    yhat_log = model.predict(Xrow)
    return float(inv_transform(yhat_log)[0])

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Disaster Prediction Parameters")

    regions = ["Eastern Africa","Western Africa","Northern Africa","Middle Africa","Southern Africa"]
    disasters = ["Flood","Drought","Storm","Epidemic","Earthquake","Landslide","Wildfire"]

    region = st.selectbox("Select African Region", regions, index=0)
    disaster = st.selectbox("Disaster Type", disasters, index=0)

    current_year = datetime.utcnow().year
    year = st.number_input("Year", min_value=1990, max_value=2100, value=current_year, step=1)
    month = st.slider("Month", 1, 12, 6)

    st.markdown("### 📊 Expected Impact Metrics")
    c1, c2 = st.columns(2)
    with c1:
        exp_deaths   = st.number_input("🕯 Expected Deaths",   min_value=0, value=50,   step=1)
        exp_injured  = st.number_input("🩺 Expected Injuries",  min_value=0, value=200,  step=1)
    with c2:
        exp_homeless = st.number_input("🏚 Expected Homeless", min_value=0, value=1000, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Top Banner + Tabs
# =========================
st.title("🌍 African Natural Disaster Impact Predictor")

leftBanner, rightBanner = st.columns([3,1])
with leftBanner:
    st.markdown(
        f"""
        <div class="banner">
          <span class="badge">🔮 Predicting disaster impacts across African regions</span>
          <span class="badge">📈 Model: XGBoost Regressor</span>
          <span class="badge">🧩 Features: {len(MODEL_FEATURES)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with rightBanner:
    st.metric("Model R² (val)", "—")  # put your true R² if you saved it

tabs = st.tabs(["🔴 Prediction", "📊 Analytics", "🧠 Model Info", "✅ Recommendations"])

# =========================
# Prediction Tab
# =========================
with tabs[0]:
    st.write("")
    predict_btn = st.button("🚨 Predict Disaster Impact", use_container_width=True)

    colL, colR = st.columns([2,1], gap="large")
    if predict_btn:
        params = {
            "region": region,
            "disaster": disaster,
            "year": int(year),
            "month": int(month),
            "exp_deaths": int(exp_deaths),
            "exp_injured": int(exp_injured),
            "exp_homeless": int(exp_homeless),
        }
        try:
            y = predict_one(params)
            with colL:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### 🧾 Prediction Results")
                st.markdown(f"<div class='result-big'>{y:,.0f} people</div> expected to be affected.", unsafe_allow_html=True)
                st.caption("Note: Estimate depends on available predictors and historical patterns.")
                st.markdown("</div>", unsafe_allow_html=True)

            with colR:
                st.markdown('<div class="summary">', unsafe_allow_html=True)
                st.markdown("### 🗂 Input Summary")
                st.write(f"*Region:* {region}")
                st.write(f"*Disaster:* {disaster}")
                st.write(f"*Year:* {year}")
                st.write(f"*Month:* {month}")
                st.write(f"*Expected Deaths:* {exp_deaths}")
                st.write(f"*Expected Injuries:* {exp_injured}")
                st.write(f"*Expected Homeless:* {exp_homeless}")
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error("Prediction failed. Your trained features may not match these controls.")
            with st.expander("See error details"):
                st.write(str(e))
    else:
        st.info("Set parameters on the left, then click *Predict Disaster Impact*.")

# =========================
# Analytics Tab
# =========================
with tabs[1]:
    st.subheader("Feature Vector Preview (what goes into the model)")
    preview = build_feature_row({
        "region": region, "disaster": disaster, "year": int(year), "month": int(month),
        "exp_deaths": int(exp_deaths), "exp_injured": int(exp_injured), "exp_homeless": int(exp_homeless),
    })
    st.dataframe(preview, use_container_width=True)
    st.caption("Exact columns and order the model receives.")

# =========================
# Model Info Tab
# =========================
with tabs[2]:
    st.subheader("Model & Features")
    st.write("- *Algorithm:* XGBoost Regressor (wrapped in a scikit-learn Pipeline with SimpleImputer).")
    st.write(f"- *Features used:* {len(MODEL_FEATURES)}")
    st.code(", ".join(MODEL_FEATURES[:80]) + ("..." if len(MODEL_FEATURES) > 80 else ""))

# =========================
# Recommendations Tab
# =========================
with tabs[3]:
    st.subheader("Operational Recommendations (General)")
    st.markdown("""
- *Early Warning & Communication:* Use radio/SMS/community alerts 48–72 hours before peak risk.
- *Resource Staging:* Pre-position medical kits, shelter materials, and water treatment in regional hubs.
- *Evacuation Readiness:* Confirm routes/transport for vulnerable groups.
- *Post-Event Triage:* Set up triage near shelters; log headcount and injuries.
- *Data Feedback:* After events, record outcomes to improve future predictions.
""")
