import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────── PAGE CONFIG ──────────────────────────── #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Customer Travel Churn Prediction")
st.markdown("Fill in the customer details below and click **Predict** to see whether they are likely to churn.")

# ─────────────────────────── LOAD FILES ───────────────────────────── #
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    # FIX 1: Dataset is TAB-delimited, not comma-delimited
    return pd.read_csv("Customertravel.csv", sep="\t")

try:
    model = load_model()
    df    = load_data()
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# ─────────────────────────── VALIDATE DATA ────────────────────────── #
# FIX 2: Target column is named 'Target', not 'Churn'
TARGET_COL = "Target"
if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' not found. Available columns: {df.columns.tolist()}")
    st.stop()

FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

# ─────────────────────────── SIDEBAR FORM ─────────────────────────── #
st.sidebar.header("📋 Customer Details")

input_data = {}

# FIX 3: Handle pandas StringDtype correctly alongside object dtype
for col in FEATURE_COLS:
    col_dtype = str(df[col].dtype)
    if col_dtype in ("object", "str") or col_dtype.startswith("string"):
        options = sorted(df[col].dropna().unique().tolist())
        input_data[col] = st.sidebar.selectbox(col, options)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.sidebar.number_input(
            col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=1.0 if df[col].dtype in ("int64", "int32") else 0.01
        )

input_df = pd.DataFrame([input_data])

# ─────────────────────────── ENCODING ─────────────────────────────── #
# FIX 4: Fit LabelEncoder on original data, then transform input
# (avoids unseen-label errors and encoding mismatch with trained model)
for col in FEATURE_COLS:
    col_dtype = str(df[col].dtype)
    if col_dtype in ("object", "str") or col_dtype.startswith("string"):
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_df[col] = le.transform(input_df[col].astype(str))

# Ensure column order matches training
input_df = input_df[FEATURE_COLS]

# ─────────────────────────── MAIN PANEL ───────────────────────────── #
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Input Summary")
    st.dataframe(
        pd.DataFrame(input_data, index=["Value"]).T.rename(columns={"Value": "Selected"}),
        use_container_width=True
    )

with col2:
    st.subheader("🔍 Prediction Result")

    if st.button("🚀 Predict Churn", use_container_width=True):
        try:
            prediction    = model.predict(input_df)[0]
            probability   = model.predict_proba(input_df)[0]

            churn_prob    = round(probability[1] * 100, 2)
            stay_prob     = round(probability[0] * 100, 2)

            if prediction == 1:
                st.error(f"⚠️ **This customer is likely to CHURN**")
            else:
                st.success(f"✅ **This customer is likely to STAY**")

            st.metric("Churn Probability",  f"{churn_prob}%")
            st.metric("Retention Probability", f"{stay_prob}%")

            # Confidence bar
            st.progress(int(churn_prob), text=f"Churn risk: {churn_prob}%")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ─────────────────────────── DATASET PREVIEW ──────────────────────── #
with st.expander("📂 View Dataset Sample"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("📈 Churn Distribution"):
    churn_counts = df[TARGET_COL].value_counts().rename({0: "Stay", 1: "Churn"})
    st.bar_chart(churn_counts)
