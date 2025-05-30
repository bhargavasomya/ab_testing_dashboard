import pandas as pd
import numpy as np
from scipy.stats import chisquare, shapiro, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower


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


def run_statistical_test(users):
    group_a = users[users['group'] == 'A']['converted']
    group_b = users[users['group'] == 'B']['converted']
    p_norm_a, p_norm_b = check_normality(users)
    if p_norm_a < 0.05 or p_norm_b < 0.05:
        test_name = "Mann-Whitney U Test"
        stat, p_val = mannwhitneyu(group_a, group_b, alternative='two-sided')
    else:
        test_name = "Z-Test"
        success = users.groupby('group')['converted'].sum().values
        nobs = users.groupby('group')['converted'].count().values
        stat, p_val = proportions_ztest(count=success, nobs=nobs)
    return test_name, stat, p_val


def calculate_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.8):
    std_dev = np.sqrt(baseline_rate * (1 - baseline_rate))
    effect_size = mde / std_dev
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0, alternative='two-sided')
    return int(np.ceil(sample_size))
