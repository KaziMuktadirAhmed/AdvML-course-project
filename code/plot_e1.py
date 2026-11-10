"""Summarize and plot E1 (results/e1_gradnorm_raw.csv -> results/e1_summary.csv, figures/e1_gradnorm.{png,pdf}).

Theorem 4.1 bound (local observable, input |0...0>, Tr[sigma_j rho]^2 = 1): E||grad f||^2 >= L / (S^S (L+2)^(S+1)), S = 2.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
S = 2


def bound(L):
    return L / (S ** S * (L + 2) ** (S + 1))


df = pd.read_csv(os.path.join(RES, "e1_gradnorm_raw.csv"))
df["per_param"] = df["sq_norm"] / df["params"]
g = df.groupby(["observable", "blocks", "L", "qubits", "params", "init"])
summ = g.agg(draws=("sq_norm", "size"),
             mean_sq_norm=("sq_norm", "mean"), sem_sq_norm=("sq_norm", "sem"),
             mean_per_param=("per_param", "mean"), sem_per_param=("per_param", "sem"),
             mean_g000_sq=("g000_sq", "mean"), sem_g000_sq=("g000_sq", "sem"),
             mean_meas_sq=("meas_sq", "mean"), min_meas_sq=("meas_sq", "min"),
             sec_per_grad=("sec_per_grad", "mean")).reset_index()
summ["thm41_bound"] = np.where(summ["observable"] == "Z1Z2", bound(summ["L"]), np.nan)
summ.to_csv(os.path.join(RES, "e1_summary.csv"), index=False)

# exponential-decay fits: log2 E[g^2] = a - b N  (b = halvings per extra qubit)
fits = []
for (obs, blocks, init), d in summ[summ.init != "zero"].groupby(["observable", "blocks", "init"]):
    for col in ("mean_per_param", "mean_g000_sq"):
        d2 = d[d[col] > 0]
        if len(d2) >= 3:
            slope = np.polyfit(d2["qubits"], np.log2(d2[col]), 1)[0]
            fits.append({"observable": obs, "blocks": blocks, "init": init, "metric": col,
                         "log2_slope_per_qubit": slope, "factor_per_qubit": 2 ** slope})
pd.DataFrame(fits).to_csv(os.path.join(RES, "e1_decay_fits.csv"), index=False)

# ---- figure ------------------------------------------------------------------------------------------------
colors = dict(zip(sorted(df.blocks.unique()), ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))
style = {"gaussian": dict(ls="-", marker="o"), "uniform": dict(ls="--", marker="s", mfc="white")}
panels = [("H", "mean_per_param", "sem_per_param", "Heisenberg H: $\\mathbb{E}\\|\\nabla f\\|^2$ / #params"),
          ("Z1Z2", "mean_sq_norm", "sem_sq_norm", "$Z_1Z_2$: $\\mathbb{E}\\|\\nabla f\\|^2$ (Thm 4.1 bound dotted)"),
          ("H", "mean_g000_sq", "sem_g000_sq", "Heisenberg H: $\\mathbb{E}(\\partial f/\\partial\\theta_{\\rm first})^2$"),
          ("Z1Z2", "mean_g000_sq", "sem_g000_sq", "$Z_1Z_2$: $\\mathbb{E}(\\partial f/\\partial\\theta_{\\rm first})^2$")]
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, (obs, col, err, title) in zip(axes.flat, panels):
    for blocks, c in colors.items():
        for init in ("gaussian", "uniform"):
            d = summ[(summ.observable == obs) & (summ.blocks == blocks) & (summ.init == init)].sort_values("qubits")
            d = d[d[col] > 0]
            ax.errorbar(d.qubits, d[col], yerr=d[err], color=c, capsize=2, ms=4, **style[init],
                        label=f"{init}, {blocks} blocks (L={2 * blocks - 2})")
        if obs == "Z1Z2" and col == "mean_sq_norm":
            ax.axhline(bound(2 * blocks - 2), color=c, ls=":", lw=1.2)
    ax.set_yscale("log"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3, which="both")
for ax in axes[1]:
    ax.set_xlabel("qubits N")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("E1: gradient at initialization, Gaussian $\\gamma^2=1/(4S(L+2))$ (solid) vs uniform $[0,2\\pi]$ (dashed); "
             "mean $\\pm$ s.e.m. over draws", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"e1_gradnorm.{ext}"), dpi=160)
print("saved figures/e1_gradnorm.png/.pdf, results/e1_summary.csv, results/e1_decay_fits.csv")
