import numpy as np
from scipy import stats
def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x, dtype=float).flatten()
    n = len(x)
    xbar = np.mean(x)
    s = np.std(x, ddof=1)   # sample std dev (Bessel's correction)
    se = s / np.sqrt(n)     # standard error
    t_stat = (xbar - mu0) / se
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return float(t_stat)