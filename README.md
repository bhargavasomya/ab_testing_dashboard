
# [🧪 A/B Testing Power Tool](https://abtestingdashboard.streamlit.app/)

This interactive Streamlit app allows you to perform end-to-end A/B testing workflows, including power analysis, SRM checks, normality testing, uplift modeling, segmented A/B testing, and more. It’s designed for product analysts, data scientists, and growth teams to quickly analyze experiments with statistical rigor.

---

## 🚀 Features

### ✅ Sample Size Calculator
- Input baseline conversion rate, MDE, α, and power to compute the required users per group.
- Uses two-proportion z-test formula for binary outcomes.
- Includes educational explanation of Power Analysis. 

---

### 🔍 SRM (Sample Ratio Mismatch) Check
- Chi-square goodness-of-fit test between expected and observed group sizes.
- Detects experiment allocation bugs.

### 🧪 Normality Testing
- Shapiro-Wilk test to determine if the data is normally distributed.
- Recommends alternatives (Mann-Whitney U) if violated.

### 🎯 A/B Test Module
- One-sided and two-sided tests.
- Auto-switch between T-test (for normal data) and Mann-Whitney U test (for non-normal).
- Computes p-value, confidence interval, and effect size (Cohen’s d or Cliff’s delta).

### 📈 Confidence Intervals
- Visual plot and interpretation of 95% CI for mean difference.

### 🔬 Segmented A/B Testing
- Allows testing experiment results across user-defined segments (e.g., platform, country).
- Identifies heterogeneous treatment effects.

### 📊 Uplift Modeling (T-Learner)
- Compares model predictions across treatment and control to estimate individual uplift.
- Useful for targeting optimization.

### 📉 Pre/Post Trend Analysis
- Checks for parallel trends and detects drift over time.
- Time-series line chart.

### 🔎 Multiple Testing Corrections
- Adjusts p-values using Bonferroni or Benjamini-Hochberg (FDR).
- Helps mitigate false positives when testing multiple metrics.

### 📚 Education Tab
- Rich educational content for all modules.
- Beginner to advanced explanations.
- “Explain Like I’m 5” mode for newcomers.

---

## 📂 How to Use

1. Upload your A/B test dataset with `variant`, `metric`, and optionally `date` and segment columns.
2. Alternatively, use the built-in sample dataset.
3. Navigate via the sidebar to choose tools and view analysis.

---

## 📁 Sample Dataset Format

CSV must include:

```
variant,metric,date,platform
A,0,2023-01-01,web
B,1,2023-01-01,web
...
```

---

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas, NumPy, SciPy, Scikit-learn
- Seaborn & Matplotlib for visualizations

---

## 👩‍💻 Author

Built with ❤️ by Somya Bhargava for A/B testing enthusiasts and practitioners.
