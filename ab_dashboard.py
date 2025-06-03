
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, chi2_contingency, shapiro, mannwhitneyu
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import seaborn as sns
import math

st.set_page_config(layout="wide")
st.title("🧪 A/B Testing Power Tool")

# --- Data Upload ---
uploaded_file = st.sidebar.file_uploader("📤 Upload your CSV", type="csv")
use_sample = st.sidebar.checkbox("Use built-in sample dataset")
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("sample_ab_test_dataset.csv")


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

    with st.expander("📘 Educational: What is SRM and Why It Matters"):
        st.markdown("""
        **Sample Ratio Mismatch (SRM)** occurs when the observed number of users in control and treatment groups
        significantly deviates from the expected ratio. This can happen due to bugs, tracking issues, or biased assignment logic.

        **Why It Matters:**
        - SRM violates the assumption of random assignment.
        - It can lead to invalid statistical inferences.

        **Test Used:** We use a **Chi-square goodness-of-fit test** to compare observed group sizes against expected ones.
        A p-value < 0.05 suggests the assignment might not be random.
        """)


    with st.expander("📘 What is SRM and Why Does It Matter?"):
        st.markdown("SRM occurs when your variant group sizes are imbalanced despite randomization. This can bias your results.")

def check_normality(df):
    st.subheader("🧪 Normality Check")
    variants = df["variant"].unique()
    for v in variants:
        p_val = shapiro(df[df["variant"] == v]["metric"])[1]
        st.write(f"Variant {v} Shapiro-Wilk p-value:", p_val)
        plt.hist(df[df["variant"] == v]["metric"], bins=10, alpha=0.5, label=str(v))
        if p_val < 0.05:
            st.warning(f"⚠️ Variant {v} data may not be normally distributed.")
        else:
            st.success(f"✅ Variant {v} passes normality test.")
    st.pyplot(plt.gcf())

    with st.expander("📘 Educational: Normality Check"):
        st.markdown("""
        **Normality tests** check whether your metric is approximately normally distributed in each variant group.

        **Why It Matters:**
        - Many statistical tests (e.g., t-tests) assume normal distribution of the data.
        - Violations can lead to inaccurate p-values or reduced test power.

        **Test Used:** We use the **Shapiro-Wilk test** to assess normality.
        - A p-value < 0.05 means the data is likely **not** normally distributed.

        **Alternative Tests:** If normality is violated, consider:
        - Non-parametric tests like Mann-Whitney U
        - Bootstrap methods
        """)


def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / math.sqrt(((nx - 1)*x.std()**2 + (ny - 1)*y.std()**2) / dof)

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    count = sum((xi > yj) - (xi < yj) for xi in x for yj in y)
    return count / (nx * ny)

def mean_confidence_interval(data1, data2, confidence=0.95):
    n1, n2 = len(data1), len(data2)
    m1, m2 = np.mean(data1), np.mean(data2)
    se = np.sqrt(np.var(data1, ddof=1)/n1 + np.var(data2, ddof=1)/n2)
    diff = m1 - m2
    z = norm.ppf(1 - (1 - confidence) / 2)
    return diff - z*se, diff + z*se

def run_ab_test(df):
    st.subheader("🎯 A/B Test")
    alternative = st.radio("Test Type", ["Two-sided", "One-sided"])
    
    if "variant" not in df.columns or "metric" not in df.columns:
        st.warning("Data must contain 'variant' and 'metric' columns.")
        return

    var = df["variant"].unique()
    if len(var) != 2:
        st.warning("A/B test requires exactly two variants.")
        return

    data1 = df[df["variant"] == var[0]]["metric"]
    data2 = df[df["variant"] == var[1]]["metric"]

    # Visualization: histograms
    st.write("### Distribution of Metrics")
    fig, ax = plt.subplots()
    sns.histplot(data1, color="blue", kde=True, label=f"Variant {var[0]}", ax=ax)
    sns.histplot(data2, color="green", kde=True, label=f"Variant {var[1]}", ax=ax)
    ax.legend()
    st.pyplot(fig)

    # Normality check
    p1 = shapiro(data1)[1]
    p2 = shapiro(data2)[1]
    normal = p1 > 0.05 and p2 > 0.05

    if normal:
        stat, p = ttest_ind(data1, data2, equal_var=False)
        if alternative == "One-sided":
            p /= 2
        st.write(f"**T-test p-value:** {p:.4f}")
        st.write(f"**Effect Size (Cohen's d):** {cohens_d(data1, data2):.3f}")
        ci_low, ci_high = mean_confidence_interval(data1, data2)
        st.write(f"**95% CI for Mean Difference:** [{ci_low:.3f}, {ci_high:.3f}]")
        if p < 0.05:
            st.success("✅ Statistically significant difference.")
        else:
            st.info("ℹ️ No significant difference found.")
    else:
        stat, p = mannwhitneyu(data1, data2, alternative="two-sided" if alternative == "Two-sided" else "less")
        if alternative == "One-sided":
            p /= 2
        st.write(f"**Mann-Whitney U p-value:** {p:.4f}")
        st.write(f"**Effect Size (Cliff's Delta):** {cliffs_delta(data1.tolist(), data2.tolist()):.3f}")
        if p < 0.05:
            st.success("✅ Statistically significant difference (non-parametric test).")
        else:
            st.info("ℹ️ No significant difference found.")
    

    # CI plot
    st.write("### Confidence Interval for Mean Difference")
    ci_low, ci_high = mean_confidence_interval(data1, data2)
    fig, ax = plt.subplots()
    ax.errorbar(x=0, y=np.mean(data1) - np.mean(data2), yerr=[[np.mean(data1) - np.mean(data2) - ci_low], [ci_high - (np.mean(data1) - np.mean(data2))]],
                fmt='o', capsize=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title("Mean Difference with 95% CI")
    ax.set_ylabel("Difference in Means")
    ax.set_xticks([])
    st.pyplot(fig)
    with st.expander("📘 Understanding Effect Sizes"):
    st.markdown("""
Effect size quantifies the **magnitude** of the difference between groups:

- **Cohen’s d**: Standardized difference in means (for t-tests). Rules of thumb:
    - 0.2 = small effect
    - 0.5 = medium
    - 0.8 = large
- **Cliff’s Delta**: Proportion of values in one group higher than in the other (non-parametric).

Use effect sizes alongside p-values to interpret practical significance, not just statistical.
    """)
    with st.expander("📘 What is a Confidence Interval?"):
    st.markdown("""
A **confidence interval (CI)** gives a range of values within which we expect the true population parameter (e.g., mean difference or uplift) to fall, with a certain level of confidence (typically 95%).

---

### ✅ Key Concepts

- A 95% confidence interval means:  
  *“If we repeated this experiment 100 times, we expect the true effect to lie within this interval in 95 of those experiments.”*

- CI = **[lower bound, upper bound]**

---

### 💡 Why It Matters

- It helps **quantify uncertainty** in your estimates.
- If a 95% CI for the mean difference **does not include zero**, the result is statistically significant at α = 0.05.
- CIs offer more information than just p-values, providing **both direction and size** of the effect.

---

### 🛠️ Interpretation Examples

- ✅ CI = [0.01, 0.05]: The treatment improves conversion by 1–5%.
- ⚠️ CI = [-0.02, 0.04]: The effect is inconclusive (it could be positive or negative).
- ❌ CI = [-0.05, -0.01]: The treatment has a negative impact.

---

### 📊 Best Practice

Always report CIs alongside p-values and effect sizes to give a **more complete picture** of your experiment results.
    """)


def run_uplift_modeling(df):
    st.subheader("📈 Uplift Modeling - T Learner")
    features = st.multiselect("Choose features", [col for col in df.columns if col not in ["variant", "metric"]])
    if not features:
        st.warning("Please select features")
        return
    df["treatment"] = (df["variant"] == df["variant"].unique()[1]).astype(int)
    X = df[features]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    y = df["metric"]
    model_t = RandomForestClassifier().fit(X[df["treatment"] == 1], y[df["treatment"] == 1])
    model_c = clone(model_t).fit(X[df["treatment"] == 0], y[df["treatment"] == 0])
    uplift = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]
    st.write("Avg uplift:", np.mean(uplift))
    st.line_chart(uplift)
    with st.expander("📘 What is Uplift Modeling?"):
    st.markdown("""
Uplift modeling estimates the **individual-level impact** of the treatment rather than just group-level averages.

We use a **T-Learner** approach:
- Train one model on the **treatment group**
- Train another on the **control group**
- Subtract predictions to get **uplift score** per user

Why use it?
- Identify **who benefits most** from the treatment.
- Target interventions more effectively.
- Go beyond “did it work?” to “for whom did it work?”
    """)

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
    with st.expander("📘 Why Check Pre/Post Trends?"):
    st.markdown("""
Pre/post trend analysis checks whether groups followed similar trends before the test started.

Why important?
- If pre-trends differ, differences after the test could be due to **baseline drift**, not the treatment.
- Ensures **parallel trend assumption**, critical for causal inference in time-series experiments.

We visualize daily metrics per group to check alignment before and divergence after launch.
    """)

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
    with st.expander("📘 Why Apply Multiple Testing Correction?"):
    st.markdown("""
If you're testing **many metrics** (or many variants), the chance of a **false positive** increases.

We apply:
- **Bonferroni Correction**: Very strict, divides significance level by number of tests.
- **Benjamini-Hochberg (FDR)**: Less conservative, controls the expected proportion of false positives.

Use when:
- Comparing multiple metrics
- Running tests on several cohorts
- Running many variants in the same experiment
    """)



def educational_toggle():
    st.subheader("📘 Explain Mode")
    mode = st.radio("Choose View", ["Beginner", "Advanced", "ELI5"])
    if mode == "Beginner":
        st.markdown("**A/B testing** compares group averages to test for significance.")
    elif mode == "Advanced":
        st.markdown("You may use t-tests, FDR corrections, or uplift modeling.")
    else:
        st.markdown("💡 *A/B tests ask: 'Is version B better than version A?'*")

def run_segmented_ab_test(df):
    st.subheader("🔍 Segmented A/B Testing")
    if "variant" not in df.columns or "metric" not in df.columns:
        st.warning("❗ Please upload data with 'variant' and 'metric' columns.")
        return

    segment_col = st.selectbox("Select a segmentation feature", [col for col in df.columns if col not in ["variant", "metric"]])
    segments = df[segment_col].unique()

    for segment in segments:
        st.markdown(f"#### Segment: {segment}")
        subset = df[df[segment_col] == segment]
        variants = subset["variant"].unique()

        if len(variants) != 2:
            st.warning(f"⚠️ Skipping segment '{segment}' (needs 2 variants).")
            continue

        group1 = subset[subset["variant"] == variants[0]]["metric"]
        group2 = subset[subset["variant"] == variants[1]]["metric"]

        stat, p = ttest_ind(group1, group2)
        st.write(f"p-value = {p:.4f}")
        with st.expander("📘 What is Segmented A/B Testing?"):
    st.markdown("""
Segmented A/B testing analyzes how different user groups (segments) respond to the treatment.

Why use it?
- Your experiment might work better for **certain user types** (e.g., mobile users vs desktop, new vs returning).
- Helps identify **heterogeneous treatment effects** and tailor product experiences.

Each segment is tested separately to reveal nuanced insights.
    """)

# --- Navigation ---
tab = st.sidebar.radio("Choose Tool", [
    "Sample Size Calculator",
    "Check Data Quality",
    "Run A/B Test",
    "Run Segmented A/B Test",
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
elif tab == "Run Segmented A/B Test":
    if df is not None:
        run_segmented_ab_test(df)
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
    st.header("📚 A/B Testing Tutorial")
    st.markdown("""
    ## 🧪 What is A/B Testing?

    A/B testing is an experiment comparing two or more variants (A, B, etc.) to determine which one performs better for a given metric.

    ---

    ## 🔍 Sample Ratio Mismatch (SRM)

    SRM occurs when the number of users in each group is not proportionate as expected under random assignment. This could signal a bug in targeting or assignment logic.
    We detect SRM using a **Chi-square goodness-of-fit test**.

    ---

    ## 📊 Normality Checks

    Statistical tests like t-tests assume normal distribution of the metric. We check this using the **Shapiro-Wilk test**. If the distribution fails this check, we advise:
    - Using non-parametric tests (e.g., Mann-Whitney U)
    - Bootstrapping

    ---

    ## 🎯 A/B Testing

    We run standard t-tests (one-sided or two-sided) to compare the means of treatment and control groups.

    ---

    ## 📈 Uplift Modeling

    Uplift modeling estimates the causal effect of an intervention per individual. We use a **T-Learner**:
    - Train one model on the treatment group
    - Train another on the control group
    - Subtract their predictions to compute uplift

    ---

    ## 🧠 Multiple Testing Correction

    - **Why Adjust?** When you test many hypotheses, even at 5% significance, you increase the chance of false positives.
    - **Bonferroni** is conservative: divide α by number of tests.
    - **Benjamini-Hochberg (FDR)** controls the expected proportion of false discoveries.

    Use corrections when:
    - You're comparing multiple metrics
    - You're slicing by cohorts
    - You're running the same test across multiple variants
    ---

    ## 📉 Pre/Post Trend Analysis

    When time-series data is present, we recommend checking parallel pre-trends to ensure experimental validity. Drift post-intervention is also visualized.

    ---

    ## 🎓 "Explain Like I'm 5" Mode

    We've added toggles throughout the tool that simplify statistical concepts for new learners!
    """)



# # --- Multiple Testing Correction ---
# if df is not None and "variant" in df.columns and "metric" in df.columns:
#     st.subheader("📊 Multiple Testing Correction")

#     st.markdown("If you're testing multiple metrics or hypotheses, it's important to adjust for multiple comparisons to avoid inflated false positive rates.")

#     metrics = df.select_dtypes(include='number').columns.tolist()
#     selected_metrics = st.multiselect("Select numeric metrics to test", metrics)

#     if len(selected_metrics) >= 2:
#         import scipy.stats as stats
#         from statsmodels.stats.multitest import multipletests

#         group_a = df[df["variant"] == "A"]
#         group_b = df[df["variant"] == "B"]

#         p_values = []
#         for metric in selected_metrics:
#             _, p = stats.ttest_ind(group_a[metric], group_b[metric], nan_policy="omit")
#             p_values.append(p)

#         st.write("Raw p-values:", p_values)

#         method = st.selectbox("Choose correction method", ["Bonferroni", "Benjamini-Hochberg (FDR)"])
#         if method == "Bonferroni":
#             corrected = multipletests(p_values, alpha=0.05, method="bonferroni")
#         else:
#             corrected = multipletests(p_values, alpha=0.05, method="fdr_bh")

#         reject, pvals_corrected, _, _ = corrected
#         results_df = pd.DataFrame({
#             "Metric": selected_metrics,
#             "Raw p-value": p_values,
#             "Corrected p-value": pvals_corrected,
#             "Reject Null?": reject
#         })

#         st.dataframe(results_df)

#         with st.expander("📘 Learn More About Multiple Testing"):
#             st.markdown("""
# - **Why Adjust?** When you test many hypotheses, even at 5% significance, you increase the chance of false positives.
# - **Bonferroni** is conservative: divide α by number of tests.
# - **Benjamini-Hochberg (FDR)** controls the expected proportion of false discoveries.

# Use corrections when:
# - You're comparing multiple metrics
# - You're slicing by cohorts
# - You're running the same test across multiple variants
#             """)
