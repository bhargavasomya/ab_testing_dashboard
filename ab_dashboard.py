import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, mannwhitneyu, chi2_contingency, shapiro
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

# --- Original Functions ---
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
    st.subheader("📉 Sample Ratio Mismatch (SRM) Check")
    variant_col = st.selectbox("Select variant column for SRM check", options=df.columns)
    variants = df[variant_col].value_counts()
    st.bar_chart(variants)
    expected = [len(df) / len(variants)] * len(variants)
    chi2, p, _, _ = chi2_contingency([variants.values, expected])
    st.write(f"Chi-square p-value: {p:.4f}")

# --- Enhancements ---
def is_normal(data):
    stat, p = shapiro(data)
    return p > 0.05

def run_frequentist_test(data):
    variant_col = st.selectbox("Variant column (Frequentist)", options=data.columns)
    metric_col = st.selectbox("Metric column (Frequentist)", options=data.columns)
    group_a = data[data[variant_col] == data[variant_col].unique()[0]][metric_col]
    group_b = data[data[variant_col] == data[variant_col].unique()[1]][metric_col]
    if is_normal(group_a) and is_normal(group_b):
        stat, p = ttest_ind(group_a, group_b)
        method = "T-test (parametric)"
    else:
        stat, p = mannwhitneyu(group_a, group_b)
        method = "Mann-Whitney U-test (non-parametric)"
    st.write(f"**{method}** p-value: {p:.4f}")

def run_bayesian_test(data):
    variant_col = st.selectbox("Variant column (Bayesian)", options=data.columns)
    metric_col = st.selectbox("Metric column (Bayesian)", options=data.columns)
    summary = data.groupby(variant_col)[metric_col].agg(['mean', 'std', 'count'])
    fig, ax = plt.subplots()
    for idx, row in summary.iterrows():
        mu = row['mean']
        sigma = row['std'] / np.sqrt(row['count'])
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, label=f"Variant {idx}")
    ax.set_title("Posterior Distributions (approx.)")
    ax.legend()
    st.pyplot(fig)
    for idx, row in summary.iterrows():
        st.write(f"Variant {idx} → mean: {row['mean']:.4f}, std: {row['std']:.4f}, n: {int(row['count'])}")

def multi_metric_analysis(data):
    variant_col = st.selectbox("Variant column (Multi-metric)", options=data.columns)
    metric_cols = st.multiselect("Select multiple metric columns", options=data.select_dtypes(include=np.number).columns.tolist())
    alpha = 0.05 / len(metric_cols)
    for metric in metric_cols:
        group_a = data[data[variant_col] == data[variant_col].unique()[0]][metric]
        group_b = data[data[variant_col] == data[variant_col].unique()[1]][metric]
        p = ttest_ind(group_a, group_b).pvalue
        st.write(f"{metric} — Adjusted p: {p:.4f}, Significant: {p < alpha}")

def segment_analysis(data):
    segment_col = st.selectbox("Segment column", options=data.columns)
    segment_vals = data[segment_col].unique()
    for val in segment_vals:
        st.subheader(f"Segment: {val}")
        seg_data = data[data[segment_col] == val]
        run_frequentist_test(seg_data)

def run_uplift_model(data):
    variant_col = st.selectbox("Variant column (Uplift)", options=data.columns)
    target_col = st.selectbox("Target column (binary)", options=data.columns)
    feature_cols = st.multiselect("Feature columns", [c for c in data.columns if c not in [variant_col, target_col]])
    df_model = data[[variant_col, target_col] + feature_cols].dropna()
    X = df_model[feature_cols + [variant_col]]
    X[variant_col] = X[variant_col].astype('category').cat.codes
    y = df_model[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Plot uplift prediction probabilities
    probs = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots()
    ax.hist(probs[y_test == 0], bins=20, alpha=0.5, label='Control', color='blue')
    ax.hist(probs[y_test == 1], bins=20, alpha=0.5, label='Treatment', color='green')
    ax.set_title("Uplift Prediction Probability Distributions")
    ax.legend()
    st.pyplot(fig)
    uplift_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.write(f"Uplift AUC Score: {uplift_score:.4f}")

# --- Main App Logic ---
if df is not None:
    st.header("1️⃣ Power Analysis")
    power_analysis()

    st.header("2️⃣ SRM Check")
    check_srm(df)

    st.header("3️⃣ Adaptive Frequentist A/B Test")
    run_frequentist_test(df)

    st.header("4️⃣ Bayesian A/B Test")
    run_bayesian_test(df)

    st.header("5️⃣ Multi-Metric Testing with Bonferroni Correction")
    multi_metric_analysis(df)

    st.header("6️⃣ Segment-Level Comparisons")
    segment_analysis(df)

    st.header("7️⃣ Uplift Modeling with Logistic Regression")
    run_uplift_model(df)

    st.header("8️⃣ Multiple Test Correction Explanation")
    st.markdown("""
    When multiple hypotheses are tested simultaneously, the chance of a false positive increases. To address this, we use **Bonferroni Correction**, dividing the alpha level (e.g., 0.05) by the number of tests to control the overall Type I error rate.
    """)
else:
    st.warning("Please upload a CSV file or use the sample dataset to proceed.")
