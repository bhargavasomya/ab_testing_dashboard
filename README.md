# 🧪 A/B Testing Playground

This Streamlit app helps you:

- Calculate sample size using power analysis
- Upload A/B test data and check data quality
- Run Frequentist and Bayesian A/B tests
- Visualize feature balance and distribution
- Perform segmented and multi-variant testing
- Learn the *why* behind every step!

## 🚀 Features

- ✅ Sample size calculator with power analysis explanation
- ✅ Upload data with `variant` and `metric` columns
- ✅ Support for demographic features (age, gender, income, etc.)
- ✅ SRM detection using Chi-square test
- ✅ Normality check and adaptive test selection
- ✅ Frequentist and Bayesian A/B testing
- ✅ Multi-metric and multi-variant testing
- ✅ Segment-level comparisons
- ✅ Educational explanations at every step

## 📂 File Structure

- `ab_dashboard.py` — Streamlit app frontend
- `ab_testing.py` — Core backend functions
- `requirements.txt` — Python dependencies

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run ab_dashboard.py
```

## 🌐 Deployment

You can deploy this on [Streamlit Community Cloud](https://streamlit.io/cloud) by linking your GitHub repo and selecting `ab_dashboard.py` as the main file.
