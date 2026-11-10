"""Summarize and plot E2 (results/e2_runs.csv -> results/e2_summary.csv, figures/e2_training.{png,pdf}).

Layout mirrors the paper's Fig. 2: rows = GD, Adam; columns = loss (exact grads), loss (noisy grads),
gradient norm (exact), gradient norm (noisy). Lines = mean over seeds, bands = +- 1 std (we state this explicitly,
the paper does not say what its bands are).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
TARGET = -20.0   # energy threshold for "iterations to target" (E0 = -25.67)

df = pd.read_csv(os.path.join(RES, "e2_runs.csv"))
df["grad_norm"] = np.sqrt(df["grad_sq_norm"])
E0 = df["E0"].iloc[0]
steps = df["step"].max()
keys = ["opt", "noise_var", "init"]

# ---- per-run metrics, then mean/std over seeds ----
runs = []
for k, d in df.groupby(keys + ["seed"]):
    d = d.sort_values("step")
    hit = d[d.energy <= TARGET]
    within1 = d[d.energy <= E0 + 1.0]
    runs.append(dict(zip(keys + ["seed"], k),
                     f0=d.energy.iloc[0], f10=d.energy.iloc[10], f30=d.energy.iloc[30], f_final=d.energy.iloc[-1],
                     gnorm0=d.grad_norm.iloc[0], iters_to_target=hit.step.iloc[0] if len(hit) else np.nan,
                     iters_within_1=within1.step.iloc[0] if len(within1) else np.nan,
                     sec_per_iter=d.wall_s.iloc[-1] / steps))
runs = pd.DataFrame(runs)
agg = runs.groupby(keys).agg(seeds=("seed", "size"),
                             f0=("f0", "mean"), f10_mean=("f10", "mean"), f10_std=("f10", "std"),
                             f30_mean=("f30", "mean"), f_final_mean=("f_final", "mean"), f_final_std=("f_final", "std"),
                             gnorm0_mean=("gnorm0", "mean"),
                             iters_to_target_mean=("iters_to_target", "mean"),
                             runs_reaching_target=("iters_to_target", "count"),
                             runs_within_1_of_E0=("iters_within_1", "count"),
                             sec_per_iter=("sec_per_iter", "mean")).reset_index()
agg["gap_final"] = agg["f_final_mean"] - E0
agg.to_csv(os.path.join(RES, "e2_summary.csv"), index=False)
runs.to_csv(os.path.join(RES, "e2_per_run.csv"), index=False)

# ---- figure ----
colors = {"gaussian": "#d62728", "uniform": "#1f77b4", "zero": "#2ca02c"}
curves = df.groupby(keys + ["step"]).agg(e_m=("energy", "mean"), e_s=("energy", "std"),
                                         g_m=("grad_norm", "mean"), g_s=("grad_norm", "std")).reset_index()
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for i, opt in enumerate(("gd", "adam")):
    for j, (nv, what) in enumerate([(0.0, "e"), (0.01, "e"), (0.0, "g"), (0.01, "g")]):
        ax = axes[i, j]
        for init, c in colors.items():
            d = curves[(curves.opt == opt) & (curves.noise_var == nv) & (curves.init == init)]
            if d.empty:
                continue
            m, s = d[f"{what}_m"], d[f"{what}_s"].fillna(0)
            ax.plot(d.step, m, color=c, label=init)
            ax.fill_between(d.step, m - s, m + s, color=c, alpha=0.2)
        if what == "e":
            ax.axhline(E0, color="k", ls=":", lw=1, label=f"$E_0$ = {E0:.2f}")
        ax.set_title(f"{opt.upper()}, {'exact' if nv == 0 else 'noisy (var 0.01)'} gradients: "
                     f"{'loss f' if what == 'e' else 'gradient norm'}", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 1:
            ax.set_xlabel("iteration")
axes[0, 0].legend(fontsize=8)
fig.suptitle("E2: Heisenberg model, N = 15, 10 blocks (300 params), lr 0.01; mean $\\pm$ 1 std over 5 seeds", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"e2_training.{ext}"), dpi=150)
print("saved figures/e2_training.png/.pdf, results/e2_summary.csv, results/e2_per_run.csv")
