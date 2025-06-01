
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, shapiro, ttest_ind, mannwhitneyu, norm

st.set_page_config(page_title="A/B Testing Playground", layout="wide")

st.title("🔬 A/B Testing Playground")
st.markdown("Built to teach and test with real or simulated data. Choose a mode below:")

page = st.sidebar.radio("Choose an option", ["Sample Size Calculator", "A/B Test & Data Quality Checker"])

if page == "Sample Size Calculator":
    st.header("📐 Power & Sample Size Calculator")
    st.markdown("Estimate how many samples you need to detect a meaningful difference.")
    baseline_rate = st.number_input("Baseline conversion rate (e.g., 0.1 = 10%)", 0.01, 0.99, 0.1)
    mde = st.number_input("Minimum detectable effect (absolute diff)", 0.001, 0.5, 0.02)
    power = st.slider("Statistical Power", 0.5, 0.99, 0.8)
    alpha = st.slider("Significance Level (α)", 0.01, 0.2, 0.05)
    tail = st.radio("Tail type", ["Two-sided", "One-sided"])

    z_power = norm.ppf(power)
    z_alpha = norm.ppf(1 - alpha / (2 if tail == "Two-sided" else 1))
    pooled_prob = baseline_rate + mde / 2
    se = np.sqrt(2 * baseline_rate * (1 - baseline_rate))
    sample_size = ((z_alpha + z_power) * se / mde) ** 2
    st.success(f"➡️ Minimum sample size per group: **{int(np.ceil(sample_size))}**")

elif page == "A/B Test & Data Quality Checker":
    st.header("🧪 Upload Data or Use Sample")
    uploaded_file = st.file_uploader("Upload a CSV with `variant` and `metric` columns", type="csv")
    use_sample = st.checkbox("Use built-in sample dataset")

    if uploaded_file or use_sample:
        df = pd.read_csv(uploaded_file) if uploaded_file else pd.DataFrame({
            "group": np.random.choice(["A", "B"], 1000),
            "converted": np.random.binomial(1, 0.12, 1000)
        }).rename(columns={"group": "variant", "converted": "metric"})

        st.success("✅ Data loaded successfully.")
        st.dataframe(df.head())

        st.subheader("🔍 Sample Ratio Mismatch Check (SRM)")
        obs = df["variant"].value_counts().values
        expected = [len(df) / 2] * 2
        stat, p_srm = chisquare(obs, expected)
        st.write(f"Chi-square p-value: {p_srm:.4f}")
        if p_srm < 0.05:
            st.warning("⚠️ Possible sample ratio mismatch!")
        else:
            st.success("✅ Random assignment looks fine.")

        with st.expander("📘 What is SRM and Why Does It Matter?"):
            st.markdown("**SRM occurs when group sizes differ more than expected by chance.** This usually indicates a bug or targeting issue in experiment assignment and can invalidate results.")
            st.latex(r" \chi^2 = \sum rac{(O - E)^2}{E} ")

        st.markdown("#### SRM Visualization")
        fig, ax = plt.subplots()
        counts = df["variant"].value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="pastel")
        ax.axhline(len(df) / 2, ls="--", color="gray", label="Expected")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📏 Normality Check")
        a = df[df["variant"] == df["variant"].unique()[0]]["metric"]
        b = df[df["variant"] == df["variant"].unique()[1]]["metric"]
        stat_a, p_a = shapiro(a)
        stat_b, p_b = shapiro(b)
        st.write(f"Shapiro-Wilk p-values: A = {p_a:.4f}, B = {p_b:.4f}")
        if p_a > 0.05 and p_b > 0.05:
            st.success("✅ Both groups appear normally distributed.")
        else:
            st.warning("⚠️ At least one group fails normality test. Consider using non-parametric tests.")

        st.markdown("#### Normality Visualization")
        fig2, ax2 = plt.subplots()
        sns.histplot(a, kde=True, label="Group A", ax=ax2, bins=5)
        sns.histplot(b, kde=True, label="Group B", ax=ax2, bins=5)
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("📊 Run A/B Significance Test")
        test_type = st.radio("Choose test type", ["Two-sample t-test", "Mann-Whitney U test"])
        if test_type == "Two-sample t-test":
            t_stat, p_val = ttest_ind(a, b, equal_var=False)
        else:
            t_stat, p_val = mannwhitneyu(a, b)
        st.write(f"p-value: {p_val:.4f}")
        if p_val < 0.05:
            st.success("🎉 Statistically significant difference!")
        else:
            st.info("No significant difference detected.")

        with st.expander("🚀 Advanced Features (Experimental Modules)"):
            st.markdown("These are previews of upcoming advanced experimentation features.")

            if st.button("🧪 Explore Uplift Modeling (Coming Soon)"):
                st.info("We’ll train models to detect heterogeneous treatment effects by segment.")

            if st.button("⏱️ Sequential Testing (Coming Soon)"):
                st.info("Bayesian Bandits and Group Sequential Tests planned.")

            if st.button("🎯 Multiple Test Corrections (Coming Soon)"):
                st.info("Bonferroni and Benjamini-Hochberg FDR corrections planned.")

            if st.button("📈 Pre/Post Trend Checks (Coming Soon)"):
                st.info("Visual analysis of time-based trends.")

            if st.button("📊 Decision Dashboard (Coming Soon)"):
                st.success("B is 6% better with 95% CI — recommendation: SHIP.")

            if st.button("🧠 Learn A/B Testing (Coming Soon)"):
                st.info("Toggle between beginner and advanced views with visuals.")
