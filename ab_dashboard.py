
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def run_uplift_modeling(df) if "df" in locals() else st.warning("Please upload a dataset before running uplift modeling."):
    st.subheader("📈 Uplift Modeling")

    st.markdown("""
Uplift modeling estimates how likely someone is to respond *because* of the treatment.
We model outcomes separately for treatment and control groups and then subtract.

This helps answer: **Who is positively influenced by the experiment?**
""")

    feature_cols = st.multiselect("Choose features for uplift modeling", [col for col in df.columns if col not in ["variant", "metric"]])
    if not feature_cols:
        st.info("Please select at least one feature.")
        return

    df = df.dropna(subset=["variant", "metric"] + feature_cols)
    df["treatment"] = (df["variant"] == df["variant"].unique()[1]).astype(int)

    X = df[feature_cols]
    y = df["metric"]
    treatment_mask = df["treatment"] == 1

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X[~treatment_mask], y[~treatment_mask], test_size=0.3, random_state=42)
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X[treatment_mask], y[treatment_mask], test_size=0.3, random_state=42)

    model_c = LogisticRegression().fit(X_train_c, y_train_c)
    model_t = LogisticRegression().fit(X_train_t, y_train_t)

    uplift = model_t.predict_proba(X_test_t)[:,1] - model_c.predict_proba(X_test_t)[:,1]
    uplift_df = pd.DataFrame({
        "uplift_score": uplift,
        "true_outcome": y_test_t.reset_index(drop=True)
    })

    st.write("Average uplift score for treatment group:", uplift_df["uplift_score"].mean())
    st.write(uplift_df.head())

    st.markdown("""
> 🧠 **Note**: In production, you’d use meta-learners (T-, S-, X-Learner), causal forests, or deep uplift models.
""")



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

        
with st.expander("🚀 Advanced Features"):
    st.markdown("These modules showcase advanced experimentation maturity and are recruiter-friendly.")

    if st.button("🧪 Run Uplift Modeling"):
        run_uplift_modeling(df) if "df" in locals() else st.warning("Please upload a dataset before running uplift modeling.")

    if st.button("⏱️ Run Sequential Testing"):
        run_sequential_testing(df)

    if st.button("🎯 Apply FDR Correction"):
        apply_fdr_correction([0.03, 0.04, 0.06])

    if st.button("📈 Run Pre/Post Trend Analysis"):
        run_trend_check(df)

    if st.button("📊 Generate Decision Dashboard"):
        show_decision_dashboard(df)

    if st.button("🧠 Learn A/B Testing"):
        educational_toggle()

    with st.expander("📘 What’s Coming Soon?"):
        st.markdown("""
- 🧪 **Bayesian Bandits / Alpha Spending** for early stopping
- 🎯 **Multiple metric correction UI** if multiple outcome columns
- 📈 **Parallel trends diagnostic visuals**
- 🧠 **Design simulator** for tradeoffs between MDE, power, and error types
- 🌐 **Live datasets** from OpenML or synthetic clickstreams
- 🧵 **Explain Like I'm 5 mode** for beginners
        """)



# ------------------------
# Additional Advanced Modules
# ------------------------

def run_sequential_testing(df):
    st.subheader("⏱️ Sequential Testing (Bayesian Bandits Preview)")
    st.markdown("""
Sequential tests allow for stopping the experiment early if enough evidence builds up.
We'll simulate Bayesian posterior probabilities with placeholder logic here.
""")
    st.info("Coming soon: Thompson Sampling and group sequential methods with live plots.")

def apply_fdr_correction(p_vals):
    st.subheader("🎯 False Discovery Rate (FDR) Control")
    st.markdown("Applying Benjamini-Hochberg procedure to control Type I error rate.")
    df_p = pd.DataFrame({"p_value": p_vals}).sort_values("p_value").reset_index(drop=True)
    df_p["rank"] = df_p.index + 1
    df_p["BH_threshold"] = (df_p["rank"] / len(df_p)) * 0.05
    df_p["significant"] = df_p["p_value"] < df_p["BH_threshold"]
    st.write(df_p)
    st.info("Only p-values below BH threshold are considered statistically significant.")

def run_trend_check(df):
    st.subheader("📈 Pre/Post Trend Analysis")
    st.markdown("Upload data with `date`, `variant`, and `metric` columns to view trends.")
    if not all(col in df.columns for col in ["date", "variant", "metric"]):
        st.warning("Missing required columns: 'date', 'variant', 'metric'.")
        return
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby(["date", "variant"])["metric"].mean().reset_index()
    fig, ax = plt.subplots()
    for key, grp in daily.groupby("variant"):
        ax.plot(grp["date"], grp["metric"], label=key)
    ax.legend()
    ax.set_title("Daily Conversion Rate by Variant")
    st.pyplot(fig)

def show_decision_dashboard(df):
    st.subheader("📊 Decision Dashboard")
    summary = df.groupby("variant")["metric"].agg(["mean", "count"])
    lift = summary.loc["B", "mean"] - summary.loc["A", "mean"]
    st.metric("Lift (B - A)", f"{lift:.3f}")
    st.write("Interpretation:")
    if lift > 0.01:
        st.success("🎯 B performs better — Consider shipping!")
    else:
        st.info("No large difference — Consider rerunning or segmenting.")

def educational_toggle():
    st.subheader("🧠 Learn A/B Testing")
    explain = st.radio("Choose level of explanation:", ["Beginner", "Advanced"])
    if explain == "Beginner":
        st.markdown("**A/B testing** compares two versions of something to see which performs better. It usually uses a t-test or chi-square test.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/AB_testing_Example.svg/1024px-AB_testing_Example.svg.png", width=400)
    else:
        st.markdown("In advanced A/B testing, we consider statistical power, SRM, segmentation, multi-metric correction, and Bayesian inference.")
