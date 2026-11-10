"""Summarize and plot S2.
results/s2_gradsweep_raw.csv -> results/s2_gradsweep_summary.csv, figures/s2_gradsweep.{png,pdf}
results/s2_train.csv         -> results/s2_train_summary.csv, results/s2_paired.csv, figures/s2_training.{png,pdf}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
COL = {"gaussian": "#d62728", "uniform": "#1f77b4", "zero": "#2ca02c"}
MSE_TARGET = 0.30

# ---------------- gradient sweep ----------------
g = pd.read_csv(os.path.join(RES, "s2_gradsweep_raw.csv"))
gs = g.groupby(["qubits", "init"]).agg(draws=("sq_norm", "size"), mean_sq=("sq_norm", "mean"),
                                       sem_sq=("sq_norm", "sem"), median_sq=("sq_norm", "median"),
                                       sec_per_grad=("sec_per_grad", "mean")).reset_index()
piv = gs.pivot(index="qubits", columns="init", values="mean_sq")
piv["gauss_over_unif"] = piv["gaussian"] / piv["uniform"]
gs.to_csv(os.path.join(RES, "s2_gradsweep_summary.csv"), index=False)
fits = {i: 2 ** np.polyfit(piv.index, np.log2(piv[i]), 1)[0] for i in ("gaussian", "uniform", "zero")}
print("S2 sweep (mean ||grad L||^2):\n", piv.round(4).to_string())
print("per-qubit factor (log-linear fit, N=4-14):", {k: round(v, 3) for k, v in fits.items()})

fig, ax = plt.subplots(figsize=(5.5, 4))
for init, c in COL.items():
    d = gs[gs.init == init]
    ax.errorbar(d.qubits, d.mean_sq, yerr=d.sem_sq, color=c, marker="o", capsize=2, label=init)
ax.set_yscale("log"); ax.set_xlabel("qubits N"); ax.set_ylabel(r"$\mathbb{E}\|\nabla_\theta \mathrm{MSE}\|^2$")
ax.set_title("S2: classifier loss gradient at init (10 blocks, batch 16)", fontsize=9)
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"s2_gradsweep.{ext}"), dpi=160)

# ---------------- training ----------------
t = pd.read_csv(os.path.join(RES, "s2_train.csv"))
last = t.step.max()
ev = t.dropna(subset=["test_mse"])
runs = []
for (opt, init, seed), d in ev.groupby(["opt", "init", "seed"]):
    d = d.sort_values("step")
    hit = d[d.test_mse <= MSE_TARGET]
    runs.append({"opt": opt, "init": init, "seed": seed,
                 "test_acc0": d.test_acc.iloc[0], "test_mse0": d.test_mse.iloc[0],
                 "test_acc50": d[d.step == 50].test_acc.iloc[0], "test_mse50": d[d.step == 50].test_mse.iloc[0],
                 "test_acc": d.test_acc.iloc[-1], "test_mse": d.test_mse.iloc[-1],
                 "train_acc": d.train_acc.iloc[-1], "train_mse": d.train_mse.iloc[-1],
                 "steps_to_mse_target": hit.step.iloc[0] if len(hit) else np.nan,
                 "sec_per_run": t[(t.opt == opt) & (t.init == init) & (t.seed == seed)].wall_s.max()})
runs = pd.DataFrame(runs)
summ = runs.groupby(["opt", "init"]).agg(
    seeds=("seed", "size"),
    **{f"{c}_{s}": (c, s) for c in ("test_acc0", "test_mse0", "test_acc50", "test_mse50", "test_acc", "test_mse",
                                     "train_acc", "train_mse") for s in ("mean", "std")},
    steps_to_mse_target_mean=("steps_to_mse_target", "mean"),
    runs_reaching_target=("steps_to_mse_target", "count"),
    sec_per_run=("sec_per_run", "mean")).reset_index()
summ.to_csv(os.path.join(RES, "s2_train_summary.csv"), index=False)

# paired Gaussian - uniform (and Gaussian - zero) differences per seed
paired = []
for opt, d in runs.groupby("opt"):
    w = d.pivot(index="seed", columns="init", values="test_mse")
    for other in ("uniform", "zero"):
        if other in w and "gaussian" in w:
            diff = (w["gaussian"] - w[other]).dropna()
            paired.append({"opt": opt, "comparison": f"gaussian - {other} (final test MSE)", "n": len(diff),
                           "mean_diff": diff.mean(), "std_diff": diff.std(), "gaussian_better_in": int((diff < 0).sum())})
pd.DataFrame(paired).to_csv(os.path.join(RES, "s2_paired.csv"), index=False)

opts = [o for o in ("adam", "gd") if o in t.opt.unique()]
fig, axes = plt.subplots(len(opts), 3, figsize=(14, 3.6 * len(opts)), squeeze=False)
for i, opt in enumerate(opts):
    for j, (col, lab, src) in enumerate([("test_mse", "test MSE", ev), ("test_acc", "test accuracy", ev),
                                         ("batch_grad_sq", r"batch $\|\nabla\|^2$", t.dropna(subset=["batch_grad_sq"]))]):
        ax = axes[i, j]
        for init, c in COL.items():
            d = src[(src.opt == opt) & (src.init == init)].groupby("step")[col].agg(["mean", "std"]).reset_index()
            if d.empty:
                continue
            if col == "batch_grad_sq":
                d["mean"] = d["mean"].rolling(5, min_periods=1).mean()
            ax.plot(d.step, d["mean"], color=c, label=init)
            if col != "batch_grad_sq":
                ax.fill_between(d.step, d["mean"] - d["std"].fillna(0), d["mean"] + d["std"].fillna(0), color=c, alpha=0.2)
        if col == "batch_grad_sq":
            ax.set_yscale("log")
        ax.set_title(f"{opt.upper()}: {lab}", fontsize=9); ax.grid(alpha=0.3); ax.set_xlabel("step")
axes[0, 0].legend(fontsize=8)
fig.suptitle("S2: QNN classifier, digits 3 vs 8, N = 12 (240 params), lr 0.01, batch 16; mean $\\pm$ 1 std over seeds "
             "(gradient panel: 5-step moving average)", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"s2_training.{ext}"), dpi=150)
print("saved S2 figures and summaries")
