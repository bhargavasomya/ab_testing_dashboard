
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, shapiro, chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="A/B Testing Pro", layout="wide")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Sample Size", "SRM & Normality", "A/B Testing", "Uplift Modeling", "Trend Analysis", "Tutorials"])

def load_sample_data():
    return pd.read_csv("https://raw.githubusercontent.com/streamlit/example-data/master/AB_Testing/ab_data.csv")

# 1. Sample Size
if page == "Sample Size":
    st.title("📏 Sample Size Calculator")
    effect_size = st.number_input("Minimum Detectable Effect (%)", value=5.0) / 100
    alpha = st.number_input("Significance Level (α)", value=0.05)
    power = st.number_input("Power (1 - β)", value=0.8)
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, alternative='two-sided')
    st.success("📊 You need approximately {:,} users per group.".format(int(sample_size)))
    with st.expander("ℹ️ About Sample Size"):
        st.markdown("We use two-sample z-test assumptions to compute sample size needed for a given minimum detectable effect (MDE), power, and alpha.")

# 2. SRM and Normality
elif page == "SRM & Normality":
    st.title("🔍 SRM and Normality Checks")
    df = load_sample_data()
    df = df.rename(columns={"group": "variant", "converted": "metric"})

    with st.expander("👁️ View Dataset"):
        st.dataframe(df.head())

    # SRM
    variant_counts = df["variant"].value_counts()
    observed = variant_counts.values
    expected = [len(df)/len(variant_counts)] * len(variant_counts)
    chi2, p_srm = chisquare(f_obs=observed, f_exp=expected)
    st.subheader("Sample Ratio Mismatch (SRM)")
    st.write(f"Observed Counts:\n{observed_counts}")
    st.write(f"p-value: **{p_srm:.4f}**")
    if p_srm < 0.05:
        st.warning("⚠️ Possible Sample Ratio Mismatch!")
    else:
        st.success("✅ Group assignment looks random.")

    # Visual
    fig, ax = plt.subplots()
    sns.barplot(x=variant_counts.index, y=variant_counts.values, ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Normality
    st.subheader("Normality Check (Shapiro-Wilk)")
    for variant in df["variant"].unique():
        stat, p_norm = shapiro(df[df["variant"] == variant]["metric"])
        st.write(f"{variant} group p-value: **{p_norm:.4f}**")
        if p_norm < 0.05:
            st.warning(f"⚠️ {variant} group may not be normally distributed.")
        else:
            st.success(f"✅ {variant} group appears normal.")

    fig, ax = plt.subplots()
    for v in df["variant"].unique():
        sns.histplot(df[df["variant"] == v]["metric"], kde=True, label=v, ax=ax)
    ax.legend()
    st.pyplot(fig)

# 3. A/B Testing
elif page == "A/B Testing":
    st.title("🧪 A/B Test Execution")
    df = load_sample_data()
    df = df.rename(columns={"group": "variant", "converted": "metric"})

    group_a = df[df["variant"] == "control"]["metric"]
    group_b = df[df["variant"] == "treatment"]["metric"]

    _, p_a = shapiro(group_a)
    _, p_b = shapiro(group_b)
    normal = p_a > 0.05 and p_b > 0.05

    if normal:
        stat, pval = ttest_ind(group_a, group_b)
        test_type = "Two-sample t-test"
    else:
        stat, pval = mannwhitneyu(group_a, group_b)
        test_type = "Mann-Whitney U test"

    st.write(f"Test used: **{test_type}**")
    st.metric("p-value", f"{pval:.4f}")
    if pval < 0.05:
        st.success("✅ Statistically significant difference found!")
    else:
        st.info("ℹ️ No statistically significant difference found.")

    # Visualization
    fig, ax = plt.subplots()
    sns.boxplot(x="variant", y="metric", data=df, ax=ax)
    st.pyplot(fig)

# 4. Uplift Modeling
elif page == "Uplift Modeling":
    st.title("📈 Uplift Modeling")
    df = load_sample_data()
    df = df.rename(columns={"group": "variant", "converted": "metric"})
    df["treatment"] = (df["variant"] == "treatment").astype(int)

    df["gender"] = np.random.choice(["M", "F"], size=len(df))
    df["age"] = np.random.randint(20, 60, size=len(df))
    df["income"] = np.random.normal(60000, 10000, size=len(df))

    model_choice = st.selectbox("Choose Uplift Model", ["T-Learner", "Logistic Regression"])

    features = st.multiselect("Choose Features", ["age", "gender", "income"], default=["age", "income"])

    df = pd.get_dummies(df, columns=["gender"], drop_first=True)

    X = df[features + [col for col in df.columns if "gender_" in col]]
    y = df["metric"]
    treatment = df["treatment"]

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(X, y, treatment, test_size=0.3, random_state=42)

    if model_choice == "T-Learner":
        model_t = RandomForestClassifier().fit(X_train[t_train == 1], y_train[t_train == 1])
        model_c = RandomForestClassifier().fit(X_train[t_train == 0], y_train[t_train == 0])
        uplift = model_t.predict_proba(X_test)[:, 1] - model_c.predict_proba(X_test)[:, 1]
    else:
        model = LogisticRegression().fit(pd.concat([X_train, t_train], axis=1), y_train)
        df_test = pd.concat([X_test, t_test], axis=1)
        df_test["interaction"] = df_test["treatment"] * df_test["age"]
        uplift = model.predict_proba(df_test)[:, 1]

    st.subheader("Estimated Uplift Distribution")
    fig, ax = plt.subplots()
    sns.histplot(uplift, kde=True, ax=ax)
    st.pyplot(fig)

# 5. Trend Analysis
elif page == "Trend Analysis":
    st.title("📉 Pre/Post Trend Analysis")
    st.markdown("Upload time-series data or simulate trends to validate parallel trend assumptions.")
    st.info("🔧 This feature is under active development.")

# 6. Tutorials
elif page == "Tutorials":
    st.title("📘 Educational Modules")
    with st.expander("🔍 Confidence Intervals"):
        st.markdown("A confidence interval gives a range within which we believe the true effect lies, with a given level of certainty.")
    with st.expander("🔍 P-Values vs. Bayesian Posteriors"):
        st.markdown("P-values show how extreme your result is under the null. Bayesian posteriors give direct probabilities of hypotheses.")
    with st.expander("🔍 Why SRM Matters"):
        st.markdown("Sample Ratio Mismatch suggests your users were not randomly split. This invalidates experiment assumptions.")


# --------------------------
# Decision Dashboard Section
# --------------------------
with st.expander("📊 Decision Dashboard"):
    if 'ab_results' in st.session_state:
        lift = st.session_state['ab_results'].get('lift', 0)
        p_val = st.session_state['ab_results'].get('p_value', 1.0)
        st.metric("Estimated Lift", f"{lift*100:.2f}%")
        st.metric("p-value", f"{p_val:.4f}")
        if p_val < 0.05:
            st.success("✅ Recommendation: Ship variant B!")
        elif 0.05 <= p_val <= 0.1:
            st.warning("⚠️ Recommendation: Consider rerunning with larger sample.")
        else:
            st.info("ℹ️ Recommendation: No significant difference found.")

# --------------------------
# Multiple Testing Correction
# --------------------------
with st.expander("📚 Multiple Testing Correction"):
    st.markdown("""
**Why Correct for Multiple Testing?**

When testing multiple hypotheses, the chance of a false positive increases. To control for this, we use:

- **Bonferroni Correction**: Very strict, divides alpha by number of tests.
- **Benjamini-Hochberg (BH) Procedure**: Controls False Discovery Rate (FDR).

We'll demonstrate this if multiple metrics are uploaded.
""")
    st.code("""
from statsmodels.stats.multitest import multipletests
p_vals = [0.03, 0.04, 0.01]  # Example
corrected = multipletests(p_vals, method='fdr_bh')
print(corrected)
""", language='python')

# --------------------------
# Trend Simulation Placeholder
# --------------------------
with st.expander("📈 Trend Simulation Dataset (Built-In)"):
    st.markdown("Use built-in synthetic dataset to simulate pre/post trends.")
    st.info("🧪 Feature coming soon! Will include OpenML & marketing-style datasets.")
