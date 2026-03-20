import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.sort(np.array(x))
    n = len(x)
    q_arr = np.atleast_1d(np.array(q)) / 100.0

    indices = q_arr * (n - 1)
    lower = np.floor(indices).astype(int)
    upper = np.minimum(lower + 1, n - 1)
    frac = indices - lower

    return x[lower] + frac * (x[upper] - x[lower])