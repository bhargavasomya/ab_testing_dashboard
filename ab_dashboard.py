import streamlit as st
from ab_testing import simulate_users, check_srm, check_normality, run_statistical_test, calculate_sample_size
import pandas as pd

st.title("🧪 A/B Testing Tool Dashboard")

# Parameters
n_users = st.slider("Number of Users", 100, 10000, 1000)
conv_rate_a = st.slider("Conversion Rate A", 0.0, 1.0, 0.10)
conv_rate_b = st.slider("Conversion Rate B", 0.0, 1.0, 0.12)

users = simulate_users(n_users=n_users, conv_rate_a=conv_rate_a, conv_rate_b=conv_rate_b)

st.subheader("Sample Ratio Mismatch (SRM) Check")
chi2_stat, p_srm = check_srm(users)
st.write(f"Chi2 Stat: {chi2_stat:.2f}, p-value: {p_srm:.4f}")
st.success("✅ Random assignment looks fine.") if p_srm >= 0.05 else st.warning("⚠️ Possible sample ratio mismatch!")

st.subheader("Normality Check")
p_norm_a, p_norm_b = check_normality(users)
st.write(f"Group A: p = {p_norm_a:.4f}, Group B: p = {p_norm_b:.4f}")
if p_norm_a < 0.05 or p_norm_b < 0.05:
    st.warning("⚠️ Non-normal distribution detected. Using non-parametric test.")
else:
    st.success("✅ Normal distribution confirmed.")

st.subheader("Statistical Test Results")
test_name, stat, p_val = run_statistical_test(users)
st.write(f"Test Used: {test_name}")
st.write(f"Stat: {stat:.4f}, p-value: {p_val:.4f}")
if p_val < 0.05:
    st.success("✅ Statistically significant result.")
else:
    st.info("ℹ️ No statistically significant difference.")

st.subheader("Sample Size Calculator")
baseline = st.number_input("Baseline Rate", value=0.10)
mde = st.number_input("Minimum Detectable Effect", value=0.02)
alpha = st.number_input("Significance Level (α)", value=0.05)
power = st.number_input("Power (1-β)", value=0.8)

sample_size = calculate_sample_size(baseline_rate=baseline, mde=mde, alpha=alpha, power=power)
st.write(f"🔍 Required Sample Size per Group: {sample_size}")
