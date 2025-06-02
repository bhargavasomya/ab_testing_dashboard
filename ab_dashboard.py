
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, shapiro, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

st.set_page_config(layout="wide", page_title="A/B Testing & Uplift Modeling Dashboard")


def sample_size_calculator():
    st.subheader("📏 Sample Size Calculator")
    try:
        from statsmodels.stats.power import NormalIndPower
    except ModuleNotFoundError:
        st.error("❌ 'statsmodels' is not installed. Please install it with `pip install statsmodels`.")
        return

    mde = st.number_input("Minimum Detectable Effect (%)", value=5.0)
    baseline = st.number_input("Baseline Conversion Rate (%)", value=10.0)
    power = st.number_input("Power (%)", value=80.0)
    alpha = st.number_input("Significance Level (%)", value=5.0)

    effect_size = abs(mde / 100) / np.sqrt((baseline / 100) * (1 - baseline / 100))
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power/100, alpha=alpha/100, ratio=1)
    st.success("📊 You need approximately {:,} users per group.".format(int(sample_size)))
:,} users per group.")

def check_srm(df):
    st.subheader("🔍 Sample Ratio Mismatch (SRM) Check")
    counts = df["variant"].value_counts()
    expected = [len(df)/2, len(df)/2]
    observed = list(counts)
    _, p_srm = chi2_contingency([observed, expected])
    st.metric("SRM p-value", f"{p_srm:.4f}")
    st.bar_chart(counts)

def check_normality(df):
    st.subheader("📊 Normality Check")
    plt.figure(figsize=(8, 4))
    for v in df["variant"].unique():
        p_val = shapiro(df[df["variant"] == v]["metric"])[1]
        st.metric(f"Shapiro-Wilk p-value ({v})", f"{p_val:.4f}")
        plt.hist(df[df["variant"] == v]["metric"], alpha=0.5, bins=15, label=str(v))
    plt.legend()
    st.pyplot(plt.gcf())
    with st.expander("📘 Educational: Normality Test"):
        st.markdown("""
        The **Shapiro-Wilk test** checks if your data is normally distributed. A p-value < 0.05 indicates deviation from normality.
        Many parametric tests (like the t-test) assume this condition.
        """)

def run_ab_test(df, one_sided=False):
    st.subheader("📉 A/B Test Results")
    a = df[df["variant"] == "A"]["metric"]
    b = df[df["variant"] == "B"]["metric"]
    stat, p = ttest_ind(a, b, equal_var=False)
    if one_sided:
        p /= 2
    st.metric("Test Statistic", f"{stat:.4f}")
    st.metric("p-value", f"{p:.4f}")

def run_uplift_modeling(df):
    st.subheader("🎯 Uplift Modeling")
    features = st.multiselect("Select Features", options=[col for col in df.columns if col not in ["variant", "metric"]])
    if not features:
        st.warning("Please select at least one feature.")
        return

    model_choice = st.radio("Choose Uplift Model", ["T-Learner (Two Random Forests)", "Single Logistic Regression with Treatment Interaction"])
    with st.expander("📘 Explanation: Uplift Modeling Approaches"):
        st.markdown("""
        **T-Learner** trains two separate models:
        - One on treatment group
        - One on control group
        Then subtracts predicted probabilities to estimate individual uplift.

        **Logistic Regression** uses a single model with interaction between features and treatment indicator.

        T-Learner can capture non-linear patterns but may overfit with small sample sizes. Logistic Regression is simpler and interpretable.
        """)

    df["treatment"] = (df["variant"] == "B").astype(int)
    X = pd.get_dummies(df[features], drop_first=True)
    y = df["metric"]

    if model_choice == "T-Learner (Two Random Forests)":
        model_t = RandomForestClassifier().fit(X[df["treatment"] == 1], y[df["treatment"] == 1])
        model_c = clone(model_t).fit(X[df["treatment"] == 0], y[df["treatment"] == 0])
        uplift = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]
    else:
        df_model = df[features + ["treatment"]].copy()
        df_model = pd.get_dummies(df_model, drop_first=True)
        df_model["interaction"] = df_model["treatment"] * df_model[df_model.columns[0]]
        X_model = df_model.drop(columns=["treatment"])
        model = RandomForestClassifier().fit(X_model, y)
        uplift = model.predict_proba(X_model)[:, 1] - y.mean()

    df["uplift_score"] = uplift
    st.write(df[["variant"] + features + ["uplift_score"]].head())

def multiple_testing_correction():
    st.subheader("🧪 Multiple Testing Correction")
    np.random.seed(42)
    df_metrics = pd.DataFrame({
        "metric_a": np.random.normal(0.5, 0.1, 100),
        "metric_b": np.random.normal(0.5, 0.1, 100),
        "metric_c": np.random.normal(0.5, 0.1, 100),
        "variant": ["A"]*50 + ["B"]*50
    })
    pvals_dict = {}
    for col in ["metric_a", "metric_b", "metric_c"]:
        a = df_metrics[df_metrics["variant"] == "A"][col]
        b = df_metrics[df_metrics["variant"] == "B"][col]
        _, pval = ttest_ind(a, b)
        pvals_dict[col] = pval
    df_p = pd.DataFrame(pvals_dict.items(), columns=["Metric", "Raw p-value"])
    df_p["rank"] = df_p["Raw p-value"].rank()
    df_p["Adjusted p-value"] = df_p["Raw p-value"] * len(df_p) / df_p["rank"]
    df_p["Significant?"] = df_p["Adjusted p-value"] < 0.05
    st.markdown("**Explanation:** We apply corrections to account for testing multiple hypotheses.\n\n- **Bonferroni** correction is conservative and divides the significance level by number of tests.\n- **Benjamini-Hochberg** controls the False Discovery Rate and is more powerful.\n\nBelow we test 3 dummy metrics for illustration.")
    st.write(df_p)

def education_page():
    st.header("📚 A/B Testing Tutorial")
    st.markdown("""
    ## What is A/B Testing?

    A/B testing is a method of comparing two versions of a product to determine which one performs better for a given outcome.

    ## Sample Ratio Mismatch (SRM)
    A test for whether group sizes are skewed beyond expectation. We use the chi-square test for this.

    ## Normality Check
    We check if each group is normally distributed (Shapiro-Wilk test) before using parametric tests.

    ## Uplift Modeling
    A way to understand *who* benefits from treatment, not just *whether* it works overall.

    ## Multiple Testing Correction
    When testing multiple metrics, we risk more false positives. Techniques like Bonferroni and Benjamini-Hochberg help control that.

    ## Interpreting p-values and Confidence Intervals
    Low p-values mean it's unlikely your results are due to chance. Confidence intervals help gauge magnitude and uncertainty.
    """)

# Main Navigation
tabs = ["Sample Size", "SRM & Normality Checks", "A/B Test & Uplift", "Multiple Correction", "Education"]
tab = st.sidebar.radio("Choose Section", tabs)

if tab == "Sample Size":
    sample_size_calculator()

elif tab == "SRM & Normality Checks":
    st.title("🔍 SRM and Normality Checks")
    uploaded = st.file_uploader("Upload CSV with 'variant' and 'metric' columns", type="csv", key="srm_upload")
    
    
    st.markdown("### 🚀 Use Built-in Sample Data")
    if st.button("Load Sample Data"):
        try:
            df = pd.read_csv("sample_ab_data.csv")
            st.session_state["ab_data"] = df
        except FileNotFoundError:
            st.error("sample_ab_data.csv not found. Please make sure it's in the project root.")


    if "ab_data" in st.session_state:
        df = st.session_state["ab_data"]
        st.success("Using built-in dataset.")
        st.dataframe(df.head())

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        check_srm(df)
        check_normality(df)

elif tab == "A/B Test & Uplift":
    st.title("📊 Upload Your A/B Testing Data")
    uploaded = st.file_uploader("Upload CSV with columns 'variant', 'metric', and optional features", type="csv")
    one_sided = st.checkbox("One-sided Test", value=False)
    
    
    st.markdown("### 🚀 Use Built-in Sample Data")
    if st.button("Load Sample Data"):
        try:
            df = pd.read_csv("sample_ab_data.csv")
            st.session_state["ab_data"] = df
        except FileNotFoundError:
            st.error("sample_ab_data.csv not found. Please make sure it's in the project root.")


    if "ab_data" in st.session_state:
        df = st.session_state["ab_data"]
        st.success("Using built-in dataset.")
        st.dataframe(df.head())

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        check_srm(df)
        check_normality(df)
        run_ab_test(df, one_sided)

        st.markdown("### 📊 Pre/Post Experiment Trends")
        fig, ax = plt.subplots()
        df.groupby("variant")[["pre_metric", "post_metric"]].mean().T.plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.markdown("### 🧮 A/B Testing by Cohorts")
        cohort_feature = st.selectbox("Select a feature to segment by", ["gender", "age", "income"])
        for value in df[cohort_feature].unique():
            subset = df[df[cohort_feature] == value]
            a = subset[subset["variant"] == "A"]["metric"]
            b = subset[subset["variant"] == "B"]["metric"]
            if len(a) > 0 and len(b) > 0:
                stat, p = ttest_ind(a, b, equal_var=False)
                st.markdown(f"**{cohort_feature} = {value}** — p-value: `{p:.4f}`")

        run_uplift_modeling(df)
elif tab == "Multiple Correction":
    multiple_testing_correction()
elif tab == "Education":
    education_page()


def check_srm_and_normality(df):
    st.subheader("📊 SRM and Normality Checks")

    # SRM - Sample Ratio Mismatch
    st.markdown("#### 🧪 Sample Ratio Mismatch")
    counts = df["variant"].value_counts().to_dict()
    expected = [len(df) / 2, len(df) / 2]
    observed = [counts.get("A", 0), counts.get("B", 0)]

    chi2, p_srm = chisquare(f_obs=observed, f_exp=expected)
    st.metric("SRM p-value", f"{p_srm:.4f}")
    st.bar_chart(pd.DataFrame({"Variant": ["A", "B"], "Count": observed}).set_index("Variant"))

    with st.expander("📘 What does this mean?"):
        st.markdown("""
        A low p-value (< 0.05) suggests that the observed group sizes differ significantly from expectation,
        indicating a possible **Sample Ratio Mismatch (SRM)**.
        SRM could be due to tracking bugs or poor randomization and may invalidate your experiment results.
        """)

    # Normality Test (Shapiro-Wilk)
    st.markdown("#### 📈 Normality Test")
    for v in df["variant"].unique():
        stat, p_norm = shapiro(df[df["variant"] == v]["metric"])
        st.metric(f"Normality p-value for {v}", f"{p_norm:.4f}")
        fig, ax = plt.subplots()
        ax.hist(df[df["variant"] == v]["metric"], bins=10, alpha=0.7)
        ax.set_title(f"Histogram - {v}")
        st.pyplot(fig)

    with st.expander("📘 Why check for normality?"):
        st.markdown("""
        Normality tests help determine whether your metric follows a Gaussian distribution.
        If it does **not**, you may want to avoid using parametric tests like the t-test.
        """)

def run_ab_test(df, one_sided):
    st.markdown("## 🧪 A/B Test Results")
    group_a = df[df["variant"] == "A"]["metric"]
    group_b = df[df["variant"] == "B"]["metric"]
    t_stat, p_val = ttest_ind(group_a, group_b, equal_var=False)
    p_val = p_val / 2 if one_sided else p_val
    st.metric("p-value", f"{p_val:.4f}")

    fig, ax = plt.subplots()
    ax.bar(["A", "B"], [group_a.mean(), group_b.mean()], yerr=[group_a.std(), group_b.std()], capsize=5)
    st.pyplot(fig)

    with st.expander("📘 How to interpret the result?"):
        st.markdown("""
        A low p-value (< 0.05) means there's strong evidence that the observed difference between variants is statistically significant.
        Use one-sided tests when you only care about improvement in one direction.
        """)

    st.markdown("### 📊 A/B Testing by Cohorts")
    cohort_feature = st.selectbox("Segment by", ["gender", "age", "income"])
    for value in df[cohort_feature].unique():
        seg = df[df[cohort_feature] == value]
        a = seg[seg["variant"] == "A"]["metric"]
        b = seg[seg["variant"] == "B"]["metric"]
        if len(a) > 0 and len(b) > 0:
            stat, p = ttest_ind(a, b, equal_var=False)
            st.markdown(f"**{cohort_feature} = {value}** — p-value: `{p:.4f}`")

def run_uplift_modeling(df):
    st.markdown("## 🎯 Uplift Modeling")
    model_type = st.radio("Choose uplift model", ["T-Learner", "Logistic Regression"])

    df["treatment"] = df["variant"].apply(lambda x: 1 if x == "B" else 0)
    X = pd.get_dummies(df[["gender", "age", "income"]], drop_first=True)
    y = df["metric"]

    if model_type == "T-Learner":
        model_t = RandomForestClassifier().fit(X[df["treatment"] == 1], y[df["treatment"] == 1])
        model_c = RandomForestClassifier().fit(X[df["treatment"] == 0], y[df["treatment"] == 0])
        uplift = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]
    else:
        model_lr = LogisticRegression().fit(pd.concat([X, df["treatment"]], axis=1), y)
        uplift = model_lr.predict_proba(pd.concat([X, df["treatment"]], axis=1))[:, 1]

    df["uplift"] = uplift
    st.success("Uplift scores calculated.")
    st.dataframe(df[["gender", "age", "income", "uplift"]].head())

    fig, ax = plt.subplots()
    df["uplift"].hist(bins=20, ax=ax)
    ax.set_title("Uplift Score Distribution")
    st.pyplot(fig)

    with st.expander("📘 What is uplift modeling?"):
        st.markdown("""
        Uplift modeling estimates the difference in outcome caused **by the treatment** for each individual.
        This helps identify **who benefits the most** from your variant, allowing personalized targeting.
        - **T-Learner** fits two separate models for treatment and control groups.
        - **Logistic Regression** estimates treatment impact directly as a covariate.
        """)


if "ab_data" not in st.session_state:
    st.warning("📂 Please upload your data or load the sample dataset to continue.")
else:
    df = st.session_state["ab_data"]
    st.subheader("📋 Preview of Your Dataset")
    st.dataframe(df.head())

    st.markdown("---")
    st.header("🔍 Step 1: SRM & Normality Checks")
    check_srm_and_normality(df)

    st.markdown("---")
    st.header("📊 Step 2: A/B Testing")
    one_sided = st.radio("Use one-sided test?", [True, False])
    run_ab_test(df, one_sided)

    st.markdown("---")
    st.header("🎯 Step 3: Uplift Modeling")
    run_uplift_modeling(df)
