import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Online Retail Churn Predictor",
    page_icon="🛍️",
    layout="wide"
)

sns.set_style("whitegrid")

# Load model + scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_churn_model_1.pkl")
    scaler = joblib.load("scaler_1.pkl")
    return model, scaler

model, scaler = load_artifacts()

FEATURE_COLUMNS = [
    "Recency", "Frequency", "Monetary",
    "CustomerLifetime", "AvgBasketSize",
    "AvgPurchaseInterval", "Cluster"
]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def predict_single(features_list):
    scaled = scaler.transform([features_list])
    prob = model.predict_proba(scaled)[0][1]
    label = "Churn" if prob >= 0.5 else "Not Churn"
    return prob, label

def predict_batch(df):
    raw = df.copy()
    scaled = scaler.transform(raw[FEATURE_COLUMNS])
    probs = model.predict_proba(scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    raw["Churn_Prob"] = probs
    raw["Prediction"] = preds
    raw["Prediction_Label"] = raw["Prediction"].replace({1: "Churn", 0: "Not Churn"})
    return raw

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🛍️ Online Retail — Churn Prediction System")
st.write("Predict churn for **single customers**, **multiple customers**, and visualize insights in a dashboard.")

tabs = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)", "📊 Dashboard"])

# ============================================================
# TAB 1 — SINGLE PREDICTION
# ============================================================

with tabs[0]:
    st.header("🔮 Single Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        Recency = st.number_input("Recency (days)", 0, 2000, 100)
        Frequency = st.number_input("Frequency (# invoices)", 1, 500, 10)
        Monetary = st.number_input("Monetary (£)", 0.0, 50000.0, 500.0)

    with col2:
        Lifetime = st.number_input("Customer Lifetime (days)", 0, 5000, 365)
        AvgBasket = st.number_input("Avg Basket Size", 0.1, 500.0, 5.0)
        AvgInterval = st.number_input("Avg Purchase Interval (days)", 0.0, 500.0, 30.0)

    with col3:
        Cluster = st.selectbox("Customer Segment Cluster", [0, 1, 2, 3])
        st.info("""
        **Cluster Definitions:**
        • 0: High-value, frequent  
        • 1: Moderate and stable  
        • 2: Low frequency, high churn  
        • 3: New / occasional customers
        """)

    if st.button("Predict Churn"):
        features = [
            Recency, Frequency, Monetary,
            Lifetime, AvgBasket, AvgInterval, Cluster
        ]

        prob, label = predict_single(features)
        prob, label = predict_single(features)

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{prob*100:.2f}%")

        if label == "Not Churn":
            st.success(f"Prediction: **{label}**")
        else:
            st.error(f"Prediction: **{label}**")



        # Radar chart for single customer
        categories = ["Recency","Frequency","Monetary","Lifetime","AvgBasket","AvgInterval"]
        values = [Recency, Frequency, Monetary, Lifetime, AvgBasket, AvgInterval]
        values += values[:1]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(5,5))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.3)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        plt.title("Customer Behaviour Radar Chart")

        st.pyplot(fig)

# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================

with tabs[1]:
    st.header("📂 Batch Prediction — Upload CSV File")

    st.write("""
    Upload a CSV with these columns:
    - Recency  
    - Frequency  
    - Monetary  
    - CustomerLifetime  
    - AvgBasketSize  
    - AvgPurchaseInterval  
    - Cluster  
    """)

    uploaded = st.file_uploader("Upload your customer file (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Validate
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            results = predict_batch(df)

            st.subheader("Prediction Results")
            st.dataframe(results.style.format({"Churn_Prob": "{:.2%}"}))

            # Download option
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Results CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            st.success("Batch prediction completed!")

# ============================================================
# TAB 3 — DASHBOARD
# ============================================================

with tabs[2]:
    st.header("📊 Customer Summary Dashboard")

    st.write("Upload a dataset to explore customer behaviour and churn insights.")

    uploaded_dash = st.file_uploader("Upload CSV for dashboard", type=["csv"], key="dash")

    if uploaded_dash:
        # Read the CSV first (df_raw will always exist)
        df_raw = pd.read_csv(uploaded_dash)

        # If predictions not included, compute them automatically
        if "Churn_Prob" not in df_raw.columns:
            st.info("ℹ️ No predictions found in uploaded file. Generating predictions automatically...")
            df_dash = predict_batch(df_raw)
        else:
            df_dash = df_raw

        # Display preview
        st.subheader("Dataset Preview")
        st.dataframe(df_dash.head())

        # ===== Charts =====

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Churn Probability Distribution")
            fig1, ax1 = plt.subplots(figsize=(5,4))
            sns.histplot(df_dash["Churn_Prob"], bins=20, kde=True, ax=ax1)
            st.pyplot(fig1)

        with colB:
            st.subheader("Prediction Counts")
            fig2, ax2 = plt.subplots(figsize=(5,4))
            sns.countplot(x=df_dash["Prediction_Label"], palette="coolwarm", ax=ax2)
            st.pyplot(fig2)

        st.subheader("Cluster Breakdown")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.countplot(x=df_dash["Cluster"], ax=ax3)
        st.pyplot(fig3)

        st.subheader("Monetary vs Recency (Scatter)")
        fig4, ax4 = plt.subplots(figsize=(7,5))
        sns.scatterplot(
            data=df_dash,
            x="Monetary",
            y="Recency",
            hue="Prediction_Label",
            palette="coolwarm",
            ax=ax4
        )
        st.pyplot(fig4)

        st.success("Dashboard generated successfully!")


st.markdown(
    """
    <div style="text-align:center; padding-top: 20px; color: gray;">
    <hr>
    <p style="font-size:14px;">Built with ❤️ using PyTorch & Streamlit by ROYKEANE </p>
    </div>
    """,
    unsafe_allow_html=True
)
