
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, mannwhitneyu, chi2_contingency, shapiro
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="A/B Testing Pro", layout="wide")
st.title("🧪 A/B Testing Power Tool")

# Upload or Sample Data
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV", type="csv")
use_sample = st.sidebar.checkbox("Use built-in sample dataset")
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("https://raw.githubusercontent.com/streamlit/example-data/master/ab_test_sample_data.csv")

if df is not None:
    st.markdown("### 👁️ Sample Dataset Preview")
    st.dataframe(df.head())

# Sample Size Calculator
def power_analysis():
    st.subheader("🧮 Sample Size Calculator")
    p1 = st.slider("Baseline Conversion Rate", 0.01, 0.5, 0.1)
    mde = st.slider("Minimum Detectable Effect", 0.01, 0.3, 0.05)
    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05)
    power = st.slider("Statistical Power (1 - β)", 0.7, 0.99, 0.8)

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled = np.sqrt(2 * p1 * (1 - p1))
    sample = ((z_alpha + z_beta) * pooled / mde) ** 2
    st.success("📊 You need approximately {:,} users per group.".format(int(np.ceil(sample))))

# SRM Check
def check_srm(df):
    st.subheader("📊 Sample Ratio Mismatch (SRM) Check")
    counts = df["variant"].value_counts()
    obs = counts.values
    total = sum(obs)
    expected = [total / len(obs)] * len(obs)
    stat, p = chi2_contingency([obs, expected])[:2]
    st.write("Observed Counts:", {k: int(v) for k, v in counts.items()})
    st.metric("Chi-square test p-value", f"{p:.4f}")
    if p < 0.05:
        st.warning("⚠️ Possible SRM detected!")
    else:
        st.success("✅ No SRM detected.")
    st.bar_chart(counts)

    with st.expander("📘 What is SRM?"):
        st.markdown("""
        SRM (Sample Ratio Mismatch) occurs when your variant group sizes are imbalanced despite randomization. 
        This can bias your results. We check this using a chi-square test.
        """)

# Normality Test
def check_normality(df):
    st.subheader("🧪 Normality Check")
    variants = df["variant"].unique()
    for v in variants:
        st.write(f"Variant {v}")
        st.hist(df[df["variant"] == v]["metric"], bins=10)
        stat, p = shapiro(df[df["variant"] == v]["metric"])
        st.write(f"Shapiro-Wilk p-value: {p:.4f}")
        if p < 0.05:
            st.warning("Not normally distributed.")
        else:
            st.success("Looks normally distributed.")

# A/B Testing Segmented
def ab_test(df):
    st.subheader("📊 A/B Testing by Segment")
    metric = "metric"
    segment = st.selectbox("Segment By", [c for c in df.columns if c not in ["metric", "variant"]])
    for val in df[segment].unique():
        st.markdown(f"#### Segment: {segment} = {val}")
        d = df[df[segment] == val]
        groups = d["variant"].unique()
        g1, g2 = d[d["variant"] == groups[0]][metric], d[d["variant"] == groups[1]][metric]
        stat, p = mannwhitneyu(g1, g2)
        st.write(f"Mann-Whitney U p-value: {p:.4f}")
        if p < 0.05:
            st.success("Statistically significant!")
        else:
            st.info("No statistical significance.")

# Page Tabs
tab = st.sidebar.radio("Navigation", ["Sample Size", "SRM & Normality", "Segmented A/B Test"])

if tab == "Sample Size":
    power_analysis()
elif tab == "SRM & Normality" and df is not None:
    check_srm(df)
    check_normality(df)
elif tab == "Segmented A/B Test" and df is not None:
    ab_test(df)
else:
    st.info("Upload data or use sample to continue.")
