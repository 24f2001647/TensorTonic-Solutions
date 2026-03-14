import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.array(param, dtype=float)
    grad  = np.array(grad,  dtype=float)
    m     = np.array(m,     dtype=float)
    v     = np.array(v,     dtype=float)
    """"
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    m_new = beta1 * m + (1 - beta1) * grad           # 1st moment (mean)
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)    # 2nd moment (variance)
    m_hat = m_new / (1 - beta1 ** t)                 # bias-corrected 1st moment
    v_hat = v_new / (1 - beta2 ** t)                 # bias-corrected 2nd moment
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param_new, m_new, v_new