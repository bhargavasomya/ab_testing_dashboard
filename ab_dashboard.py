import streamlit as st
from ab_testing import simulate_users, check_srm, check_normality, run_statistical_test, calculate_sample_size
import pandas as pd

st.set_page_config(page_title="A/B Testing Tool", layout="centered")

st.title("🧪 A/B Testing Tool Dashboard")
st.markdown("Use this dashboard to simulate A/B experiments, calculate required sample sizes, and analyze experiment results.")

# Step 1: Sample Size Calculation
st.subheader("📐 Step 1: Sample Size Calculator")
baseline = st.number_input("Baseline Conversion Rate", value=0.10)
mde = st.number_input("Minimum Detectable Effect (Absolute)", value=0.02)
alpha = st.number_input("Significance Level (α)", value=0.05)
power = st.number_input("Statistical Power (1-β)", value=0.8)

sample_size = calculate_sample_size(baseline_rate=baseline, mde=mde, alpha=alpha, power=power)
st.write(f"🔍 Required sample size **per group**: **{sample_size} users**")

# Step 2: Simulate or Upload Data
st.subheader("🔧 Step 2: Simulate A/B Test Data")
st.sidebar.header("Simulation Settings")
n_users = st.sidebar.slider("Total Users", 100, 10000, 1000)
conv_rate_a = st.sidebar.slider("Conversion Rate - Group A", 0.0, 1.0, 0.10)
conv_rate_b = st.sidebar.slider("Conversion Rate - Group B", 0.0, 1.0, 0.12)

# Simulate data
users = simulate_users(n_users=n_users, conv_rate_a=conv_rate_a, conv_rate_b=conv_rate_b)

st.subheader("📊 Group Assignment Summary")
group_summary = users.groupby("group")["converted"].agg(["count", "sum", "mean"]).rename(columns={
    "count": "Users",
    "sum": "Conversions",
    "mean": "Conversion Rate"
})
st.dataframe(group_summary)

# Step 3: Data Quality Checks
st.subheader("✅ Step 3: Data Quality Checks")

# SRM check
st.markdown("**Sample Ratio Mismatch (SRM) Check**")
chi2_stat, p_srm = check_srm(users)
st.write(f"Chi2 Statistic: {chi2_stat:.2f}, p-value: {p_srm:.4f}")
if p_srm >= 0.05:
    st.success("✅ Random assignment looks fine.")
else:
    st.warning("⚠️ Possible sample ratio mismatch detected!")

# Normality check
st.markdown("**Normality Check**")
p_norm_a, p_norm_b = check_normality(users)
st.write(f"Shapiro-Wilk p-values - Group A: {p_norm_a:.4f}, Group B: {p_norm_b:.4f}")
if p_norm_a < 0.05 or p_norm_b < 0.05:
    st.warning("⚠️ One or both groups are not normally distributed. Will use non-parametric test.")
else:
    st.success("✅ Both groups appear normally distributed.")

# Step 4: Run A/B Test
st.subheader("📈 Step 4: A/B Test Result")
test_name, stat, p_val = run_statistical_test(users)
st.write(f"Test Used: **{test_name}**")
st.write(f"Test Statistic: {stat:.4f}")
st.write(f"p-value: {p_val:.4f}")
if p_val < 0.05:
    st.success("✅ Statistically significant difference between groups (p < 0.05).")
else:
    st.info("ℹ️ No statistically significant difference between groups.")
