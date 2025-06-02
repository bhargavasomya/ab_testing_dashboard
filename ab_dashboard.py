
import streamlit as st
import pandas as pd
from statsmodels.stats.power import NormalIndPower

st.set_page_config(page_title="A/B Testing Dashboard", layout="wide")

def sample_size_calculator():
    st.title("Sample Size Calculator")
    effect_size = st.number_input("Minimum Detectable Effect (%)", value=5.0) / 100
    alpha = st.number_input("Significance Level (α)", value=0.05)
    power = st.number_input("Power (1 - β)", value=0.8)
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, alternative='two-sided')
    st.success("📊 You need approximately {:,} users per group.".format(int(sample_size)))

sample_size_calculator()
