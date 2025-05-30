import streamlit as st
from ab_testing import simulate_users, check_srm, check_normality, run_statistical_test, calculate_sample_size
import pandas as pd

st.set_page_config(page_title="A/B Testing Tool", layout="centered")

st.title("🧪 A/B Testing Playground")
st.markdown("Welcome to the A/B Testing Playground — a tool to help you calculate sample size, evaluate experiment quality, and understand statistical testing. Select a module below to begin.")

# Landing page selector
option = st.selectbox("Choose what you want to do:", ["📐 Sample Size Calculator", "📊 A/B Test & Data Quality Checker"])

if option == "📐 Sample Size Calculator":
    st.header("Sample Size Calculator")

    with st.expander("📘 What is Power Analysis?"):
        st.markdown("""
        Power analysis helps you determine how many samples you need to reliably detect an effect in your A/B test.

        - **Baseline Conversion Rate**: Your expected conversion rate for control (variant A).
        - **Minimum Detectable Effect (MDE)**: The smallest effect you care to detect.
        - **Significance Level (α)**: Probability of Type I error (false positive).
        - **Power (1 - β)**: Probability of correctly detecting a true effect (avoiding false negative).

        A higher power requires a larger sample size. Common choices: α = 0.05, Power = 0.8
        """)

    baseline = st.number_input("Baseline Conversion Rate", value=0.10)
    mde = st.number_input("Minimum Detectable Effect (Absolute)", value=0.02)
    alpha = st.number_input("Significance Level (α)", value=0.05)
    power = st.number_input("Statistical Power (1-β)", value=0.8)

    sample_size = calculate_sample_size(baseline_rate=baseline, mde=mde, alpha=alpha, power=power)
    st.write(f"🔍 Required sample size **per group**: **{sample_size} users**")

elif option == "📊 A/B Test & Data Quality Checker":
    st.header("A/B Test with Data Quality Checks")

    with st.expander("📘 How Does Statistical Testing Work?"):
        st.markdown("""
        Once you’ve collected data from an A/B test, statistical testing helps determine whether the observed differences are due to chance or represent a true effect.

        - **Z-Test**: Used when sample size is large and distribution is normal.
        - **Mann-Whitney U Test**: A non-parametric alternative when data is not normal.
        - **One-sided test**: Checks if one group is *greater than* the other.
        - **Two-sided test**: Checks for *any* difference between groups (higher or lower).

        Choosing between one-sided and two-sided tests depends on your hypothesis.
        """)

    test_type = st.radio("Choose test type", ["Two-sided", "One-sided"])
    alt = "two-sided" if test_type == "Two-sided" else "larger"

    test_method = st.radio("Choose test method", ["Frequentist", "Bayesian"])

    st.markdown("### Upload your CSV file")
    uploaded_file = st.file_uploader("Upload a file with at least two columns: `variant` and `metric`", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            required_cols = {"variant", "metric"}
            if not required_cols.issubset(set(df.columns)):
                st.error("Uploaded file must contain at least `variant` and `metric` columns.")
            else:
                st.success("✅ File uploaded successfully.")
                st.dataframe(df.head())

                # Proceed with sample ratio check, normality, and A/B test
                df = df.rename(columns={"variant": "group", "metric": "converted"})  # reuse existing functions
                st.markdown("### Sample Ratio Mismatch (SRM) Check")

    with st.expander("📘 What is SRM and Why Does It Matter?"):
        st.markdown("""
        **Sample Ratio Mismatch (SRM)** occurs when the actual split between control and treatment groups
        is significantly different from the expected ratio (usually 50/50).

        This can indicate issues like:
        - Improper randomization
        - Data corruption
        - Tracking bugs or dropped events

        We use the **Chi-Square Test** to compare observed group sizes to expected group sizes.

        If SRM exists, any statistical conclusions drawn from the experiment could be invalid because
        the assignment is not truly random and unbiased.
        """)
                chi2_stat, p_srm = check_srm(df)
                st.write(f"Chi2 Statistic: {chi2_stat:.2f}, p-value: {p_srm:.4f}")
                if p_srm >= 0.05:
                    st.success("✅ Random assignment looks fine.")
                else:
                    st.warning("⚠️ Possible sample ratio mismatch detected!")

                st.markdown("### Normality Check")
                p_norm_a, p_norm_b = check_normality(df)
                st.write(f"Shapiro-Wilk p-values - Group A: {p_norm_a:.4f}, Group B: {p_norm_b:.4f}")
                if p_norm_a < 0.05 or p_norm_b < 0.05:
                    st.warning("⚠️ One or both groups are not normally distributed. Will use non-parametric test.")
                else:
                    st.success("✅ Both groups appear normally distributed.")

                st.markdown("### A/B Test Result")
                test_name, stat, p_val = run_statistical_test(df, alternative=alt)
                st.write(f"Test Used: **{test_name}**")
                st.write(f"Test Statistic: {stat:.4f}")
                st.write(f"p-value: {p_val:.4f}")
                if p_val < 0.05:
                    st.success("✅ Statistically significant difference (p < 0.05).")
                else:
                    st.info("ℹ️ No statistically significant difference.")


                if test_method == "Bayesian":
                    from ab_testing import run_bayesian_ab_test
                    st.markdown("### Bayesian A/B Test Result")
                    prob_b_better, lift, fig = run_bayesian_ab_test(df)
                    st.pyplot(fig)
                    st.write(f"🧠 Probability that Group B is better than A: **{prob_b_better:.2%}**")
                    st.write(f"📈 Estimated Lift: **{lift:.2%}**")
                    if prob_b_better > 0.95:
                        st.success("✅ High confidence that B is better than A.")
                    elif prob_b_better < 0.05:
                        st.success("✅ High confidence that A is better than B.")
                    else:
                        st.info("ℹ️ No strong evidence for either group.")


        
                st.markdown("### Feature Balance Check (Optional)")
                with st.expander("📘 Why Check Feature Balance?"):
                    st.markdown("""
                    If your data includes columns like `age`, `gender`, or `income`, it's important to verify that the **distribution of these features is balanced** across treatment and control groups.

                    Imbalance can introduce bias into your experiment and make it unclear whether differences are due to the treatment or the population characteristics.
                    """)

                feature_cols = [col for col in df.columns if col not in ["group", "converted"]]

                if feature_cols:
                    selected_features = st.multiselect("Select demographic features to check", feature_cols)

                    if selected_features:
                        import seaborn as sns
                        import matplotlib.pyplot as plt

                        for feature in selected_features:
                            st.markdown(f"#### Distribution of `{feature}` by Group")
                            if pd.api.types.is_numeric_dtype(df[feature]):
                                fig, ax = plt.subplots()
                                sns.kdeplot(data=df, x=feature, hue="group", fill=True, common_norm=False, ax=ax)
                                st.pyplot(fig)
                            else:
                                dist = pd.crosstab(df[feature], df["group"], normalize='columns')
                                st.bar_chart(dist)
                else:
                    st.info("No additional features found in your dataset to compare.")


        
                
                st.markdown("### Multi-Metric & Multi-Variant Support")
                with st.expander("📘 Why Support Multiple Metrics or Variants?"):
                    st.markdown("""
                    In real-world experiments, you might:
                    - Track **more than one outcome metric** (e.g., conversion, revenue, engagement)
                    - Run tests with **more than two variants** (e.g., A/B/C/D)

                    This section allows you to explore each metric individually and see how different variants compare.
                    """)

                metric_cols = [col for col in df.columns if col not in ["group"] + selected_features]
                chosen_metric = st.selectbox("Select a metric to analyze", metric_cols, index=0)

                variant_counts = df["group"].nunique()
                st.write(f"✅ Found {variant_counts} unique variants.")

                if variant_counts > 2:
                    st.warning("⚠️ More than 2 variants detected. Using ANOVA-style comparisons.")
                    import statsmodels.api as sm
                    from statsmodels.formula.api import ols

                    model = ols(f"{chosen_metric} ~ C(group)", data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.write("### ANOVA Table")
                    st.dataframe(anova_table)

                    p_val = anova_table["PR(>F)"].values[0]
                    if p_val < 0.05:
                        st.success("✅ Statistically significant differences among variants.")
                    else:
                        st.info("ℹ️ No statistically significant difference across all variants.")
                else:
                    # Run standard test on selected metric
                    df_metric_test = df.rename(columns={chosen_metric: "converted"})
                    test_name, stat, p_val = run_statistical_test(df_metric_test, alternative=alt)
                    st.write(f"Test Used: **{test_name}**")
                    st.write(f"Test Statistic: {stat:.4f}")
                    st.write(f"p-value: {p_val:.4f}")
                    if p_val < 0.05:
                        st.success("✅ Statistically significant difference (p < 0.05).")
                    else:
                        st.info("ℹ️ No statistically significant difference.")


                st.markdown("### Segmented A/B Testing")
                with st.expander("📘 What is Segmented A/B Testing?"):
                    st.markdown("""
                    Segmented A/B testing allows you to analyze whether your experiment results vary across different subgroups
                    (e.g., age groups, gender, income brackets).

                    This is useful to uncover **interaction effects**, where a treatment works better or worse for certain users.
                    """)

                if feature_cols:
                    segment_col = st.selectbox("Select a column to segment by", feature_cols)

                    segment_values = df[segment_col].dropna().unique()
                    st.write(f"Found {len(segment_values)} segments in `{segment_col}`.")

                    segment_results = []

                    for value in segment_values:
                        segment_df = df[df[segment_col] == value]
                        if segment_df['group'].nunique() < 2:
                            continue  # Skip if only one group

                        try:
                            test_name, stat, p_val = run_statistical_test(segment_df, alternative=alt)
                            segment_results.append({
                                segment_col: value,
                                "Test Used": test_name,
                                "Test Stat": round(stat, 4),
                                "p-value": round(p_val, 4),
                                "Significant": "✅" if p_val < 0.05 else "–"
                            })
                        except:
                            continue

                    if segment_results:
                        st.markdown("#### Segment-level Results")
                        st.dataframe(pd.DataFrame(segment_results))
                    else:
                        st.info("Not enough data to perform segmented testing.")


        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a CSV file to begin A/B testing.")
