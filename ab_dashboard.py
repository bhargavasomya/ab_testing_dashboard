
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, chi2_contingency, shapiro
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

st.set_page_config(layout="wide")
st.title("🧪 A/B Testing Power Tool")

# --- Data Upload ---
uploaded_file = st.sidebar.file_uploader("📤 Upload your CSV", type="csv")
use_sample = st.sidebar.checkbox("Use built-in sample dataset")
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("sample_data.csv")

if use_sample:
    st.markdown("### 👁️ Sample Dataset Preview")
    st.dataframe(df.head())


# --- Functions ---

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
    st.metric("📊 Sample Size per Group", f"{int(np.ceil(sample))}")

    with st.expander("📘 What is Power Analysis?"):
        st.markdown("Power analysis determines the sample size required to detect a meaningful effect, balancing Type I and Type II errors.")

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

    with st.expander("📘 What is SRM and Why Does It Matter?"):
        st.markdown("SRM occurs when your variant group sizes are imbalanced despite randomization. This can bias your results.")

def check_normality(df):
    st.subheader("🧪 Normality Check")
    variants = df["variant"].unique()
    for v in variants:
        p_val = shapiro(df[df["variant"] == v]["metric"])[1]
        st.write(f"Variant {v} Shapiro-Wilk p-value:", p_val)
        st.hist(df[df["variant"] == v]["metric"], bins=10, alpha=0.5, label=v)
        if p_val < 0.05:
            st.warning(f"⚠️ Variant {v} data may not be normally distributed.")
        else:
            st.success(f"✅ Variant {v} passes normality test.")
    st.pyplot(plt.gcf())

def run_ab_test(df):
    st.subheader("📈 Run A/B Test")
    alternative = st.radio("Test Type", ["Two-sided", "One-sided"])
    var = df["variant"].unique()
    data1 = df[df["variant"] == var[0]]["metric"]
    data2 = df[df["variant"] == var[1]]["metric"]
    stat, p = ttest_ind(data1, data2, equal_var=False)
    if alternative == "One-sided":
        p /= 2
    st.write(f"t-stat: {stat:.4f}, p-value: {p:.4f}")
    if p < 0.05:
        st.success("✅ Statistically significant difference.")
    else:
        st.info("ℹ️ No significant difference found.")

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

# Design Simulator removed
    base = st.slider("Baseline Rate", 0.01, 0.3, 0.1)
    mde = st.slider("Minimum Detectable Effect", 0.005, 0.1, 0.02)
    power = st.slider("Power", 0.5, 0.99, 0.8)
    alpha = st.slider("Alpha", 0.01, 0.1, 0.05)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled = np.sqrt(2 * base * (1 - base))
    sample = ((z_alpha + z_beta) * pooled / mde) ** 2
    st.metric("Sample Size per Group", f"{int(np.ceil(sample))}")

def educational_toggle():
    st.subheader("📘 Explain Mode")
    mode = st.radio("Choose View", ["Beginner", "Advanced", "ELI5"])
    if mode == "Beginner":
        st.markdown("**A/B testing** compares group averages to test for significance.")
    elif mode == "Advanced":
        st.markdown("You may use t-tests, FDR corrections, or uplift modeling.")
    else:
        st.markdown("💡 *A/B tests ask: 'Is version B better than version A?'*")

# --- Navigation ---
tab = st.sidebar.radio("Choose Tool", [
    "Sample Size Calculator",
    "Check Data Quality",
    "Run A/B Test",
    "Run Uplift Modeling",
    "Pre/Post Trends",
    "Multiple Testing Correction",
        "Education"
])

# --- Run Modules ---
if tab == "Sample Size Calculator":
    power_analysis()
elif tab == "Check Data Quality":
    if df is not None:
        check_srm(df)
        check_normality(df)
    else:
        st.warning("Please upload or select sample data.")
elif tab == "Run A/B Test":
    if df is not None:
        run_ab_test(df)
    else:
        st.warning("Please upload or select sample data.")
elif tab == "Run Uplift Modeling":
    if df is not None:
        run_uplift_modeling(df)
    else:
        st.warning("Please upload or select sample data.")
elif tab == "Pre/Post Trends":
    if df is not None:
        run_trend_check(df)
    else:
        st.warning("Please upload data with a 'date' column.")
elif tab == "Multiple Testing Correction":
    apply_fdr_correction({"Metric A": 0.03, "Metric B": 0.04, "Metric C": 0.06})
elif tab == "Education":
    educational_toggle()
