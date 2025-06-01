
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

st.set_page_config(layout="wide")

st.title("🧪 A/B Testing Power Tool")

# Load sample dataset or user upload
uploaded_file = st.sidebar.file_uploader("📤 Upload your CSV", type="csv")
use_sample = st.sidebar.checkbox("Use built-in sample dataset")
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("sample_data.csv")

# Sequential Testing
def run_sequential_testing(df):
    st.subheader("⏱️ Sequential Testing with Bayesian Bandits")
    df["treatment"] = (df["variant"] == df["variant"].unique()[1]).astype(int)
    success_t = df[df["treatment"] == 1]["metric"].sum()
    success_c = df[df["treatment"] == 0]["metric"].sum()
    trials_t = df[df["treatment"] == 1].shape[0]
    trials_c = df[df["treatment"] == 0].shape[0]
    samples_t = np.random.beta(1 + success_t, 1 + trials_t - success_t, 1000)
    samples_c = np.random.beta(1 + success_c, 1 + trials_c - success_c, 1000)
    prob = (samples_t > samples_c).mean()
    st.metric("P(B > A)", f"{prob:.2%}")

# FDR
def apply_fdr_correction(pvals_dict):
    st.subheader("🎯 Multiple Testing Corrections")
    method = st.selectbox("Correction Method", ["Bonferroni", "Benjamini-Hochberg"])
    df_p = pd.DataFrame(pvals_dict.items(), columns=["metric", "p_value"])
    if method == "Bonferroni":
        df_p["adj_p"] = df_p["p_value"] * len(df_p)
    else:
        df_p = df_p.sort_values("p_value").reset_index(drop=True)
        df_p["rank"] = df_p.index + 1
        df_p["adj_p"] = df_p["p_value"] * len(df_p) / df_p["rank"]
    df_p["significant"] = df_p["adj_p"] < 0.05
    st.write(df_p)

# Uplift
def run_uplift_modeling(df):
    st.subheader("📈 Uplift Modeling - T Learner")
    features = st.multiselect("Choose features", [col for col in df.columns if col not in ["variant", "metric"]])
    if not features:
        st.warning("Please select features")
        return
    df["treatment"] = (df["variant"] == df["variant"].unique()[1]).astype(int)
    X = df[features]
    y = df["metric"]
    model_t = RandomForestClassifier().fit(X[df["treatment"] == 1], y[df["treatment"] == 1])
    model_c = clone(model_t).fit(X[df["treatment"] == 0], y[df["treatment"] == 0])
    uplift = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]
    st.write("Avg uplift:", np.mean(uplift))
    st.line_chart(uplift)

# Trends
def run_trend_check(df):
    st.subheader("📈 Pre/Post Trend Analysis")
    if "date" not in df.columns:
        st.error("Missing 'date' column")
        return
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby(["date", "variant"])["metric"].mean().unstack()
    fig, ax = plt.subplots()
    daily.plot(ax=ax)
    ax.set_title("Parallel Trends")
    st.pyplot(fig)

# Simulator
def design_simulator():
    st.subheader("🧪 Experiment Design Simulator")
    base = st.slider("Baseline Rate", 0.01, 0.3, 0.1)
    mde = st.slider("Minimum Detectable Effect", 0.005, 0.1, 0.02)
    power = st.slider("Power", 0.5, 0.99, 0.8)
    alpha = st.slider("Alpha", 0.01, 0.1, 0.05)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled = np.sqrt(2 * base * (1 - base))
    sample = ((z_alpha + z_beta) * pooled / mde) ** 2
    st.metric("Sample Size per Group", f"{int(np.ceil(sample))}")

# Education
def educational_toggle():
    st.subheader("📘 Explain Mode")
    mode = st.radio("Choose View", ["Beginner", "Advanced", "ELI5"])
    if mode == "Beginner":
        st.markdown("**A/B testing** compares group averages to test for significance.")
    elif mode == "Advanced":
        st.markdown("You may use t-tests, FDR corrections, or uplift modeling.")
    else:
        st.markdown("💡 *A/B tests ask: 'Is version B better than version A?'*")

# Sidebar Navigation
tab = st.sidebar.radio("Choose Tool", [
    "Design Simulator",
    "Run Uplift Modeling",
    "Sequential Testing",
    "Multiple Testing Correction",
    "Pre/Post Trends",
    "Education"
])

# Run selected tool if df available
if tab == "Design Simulator":
    design_simulator()
elif tab == "Run Uplift Modeling":
    if df is not None:
        run_uplift_modeling(df)
    else:
        st.warning("Please upload data or select sample.")
elif tab == "Sequential Testing":
    if df is not None:
        run_sequential_testing(df)
    else:
        st.warning("Please upload data or select sample.")
elif tab == "Multiple Testing Correction":
    apply_fdr_correction({"Metric A": 0.03, "Metric B": 0.04, "Metric C": 0.06})
elif tab == "Pre/Post Trends":
    if df is not None:
        run_trend_check(df)
    else:
        st.warning("Please upload data with date column.")
elif tab == "Education":
    educational_toggle()
