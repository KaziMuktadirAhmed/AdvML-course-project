"""Initializations compared in Zhang et al., NeurIPS 2022.

gaussian : theta ~ N(0, gamma^2), gamma^2 = 1/(4 S (L+2))  (Theorem 4.1); 'var_mult' scales the VARIANCE (supp. A.1)
uniform  : theta ~ U[0, 2 pi]   (variance (2 pi)^2 / 12 ~= 3.29)
zero     : theta = 0
"""
import numpy as np

UNIFORM_VAR = (2 * np.pi) ** 2 / 12


def gamma2(S, L):
    return 1.0 / (4 * S * (L + 2))


def init_params(kind, shape, rng, var=None, var_mult=1.0):
    if kind == "gaussian":
        if var is None:
            raise ValueError("gaussian init needs var (use gamma2(S, L))")
        return rng.normal(0.0, np.sqrt(var * var_mult), size=shape)
    if kind == "uniform":
        return rng.uniform(0.0, 2 * np.pi, size=shape)
    if kind == "zero":
        return np.zeros(shape)
    raise ValueError(f"unknown init {kind!r}")
