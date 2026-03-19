import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if len(x) != len(p):
        raise ValueError(f"x and p must have the same length, got {len(x)} and {len(p)}")

    if np.any(p < 0):
        raise ValueError(f"All probabilities must be non-negative, got {p}")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError(f"Probabilities must sum to 1, got {np.sum(p):.6f}")

    return float(np.dot(x, p))
