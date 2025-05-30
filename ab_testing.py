import pandas as pd
import numpy as np
from scipy.stats import chisquare, shapiro, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
import scipy.stats as stats
import matplotlib.pyplot as plt

def simulate_users(n_users=1000, conv_rate_a=0.1, conv_rate_b=0.12, seed=42):
    np.random.seed(seed)
    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'group': np.random.choice(['A', 'B'], size=n_users)
    })
    conversion_rates = {'A': conv_rate_a, 'B': conv_rate_b}
    users['converted'] = users.apply(lambda row: np.random.rand() < conversion_rates[row['group']], axis=1)
    return users

def check_srm(users):
    n_users = users.shape[0]
    expected = [n_users / 2, n_users / 2]
    observed = users['group'].value_counts().sort_index().values
    chi2_stat, p_srm = chisquare(f_obs=observed, f_exp=expected)
    return chi2_stat, p_srm

def check_normality(users):
    group_a = users[users['group'] == 'A']['converted']
    group_b = users[users['group'] == 'B']['converted']
    _, p_norm_a = shapiro(group_a)
    _, p_norm_b = shapiro(group_b)
    return p_norm_a, p_norm_b

def run_statistical_test(users, alternative='two-sided'):
    group_a = users[users['group'] == 'A']['converted']
    group_b = users[users['group'] == 'B']['converted']
    p_norm_a, p_norm_b = check_normality(users)
    if p_norm_a < 0.05 or p_norm_b < 0.05:
        test_name = "Mann-Whitney U Test"
        stat, p_val = mannwhitneyu(group_a, group_b, alternative=alternative)
    else:
        test_name = "Z-Test"
        success = users.groupby('group')['converted'].sum().values
        nobs = users.groupby('group')['converted'].count().values
        stat, p_val = proportions_ztest(count=success, nobs=nobs, alternative=alternative)
    return test_name, stat, p_val

def calculate_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.8):
    std_dev = np.sqrt(baseline_rate * (1 - baseline_rate))
    effect_size = mde / std_dev
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0, alternative='two-sided')
    return int(np.ceil(sample_size))

def run_bayesian_ab_test(df, group_col="group", metric_col="converted", group_a="A", group_b="B"):
    a_data = df[df[group_col] == group_a][metric_col]
    b_data = df[df[group_col] == group_b][metric_col]

    a_success = a_data.sum()
    a_total = a_data.count()
    b_success = b_data.sum()
    b_total = b_data.count()

    a_post = stats.beta(1 + a_success, 1 + a_total - a_success)
    b_post = stats.beta(1 + b_success, 1 + b_total - b_success)

    samples = 100_000
    a_samples = a_post.rvs(samples)
    b_samples = b_post.rvs(samples)

    prob_b_better = (b_samples > a_samples).mean()
    lift = ((b_samples - a_samples) / a_samples).mean()

    fig, ax = plt.subplots()
    ax.hist(a_samples, bins=100, alpha=0.5, label='Group A (control)', density=True)
    ax.hist(b_samples, bins=100, alpha=0.5, label='Group B (treatment)', density=True)
    ax.set_title("Posterior Distributions of Conversion Rates")
    ax.legend()

    return prob_b_better, lift, fig
