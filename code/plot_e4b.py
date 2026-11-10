"""Summarize and plot E4b (results/e4b_runs*.csv -> results/e4b_summary.csv, figures/e4b_noise.{png,pdf}).

Metrics per (backend, N, blocks, init, noise), mean +- std over seeds (zero init: 1 seed, deterministic):
  E_final        energy of the training objective after the last step (noisy energy if noise on)
  E_clean_final  NOISELESS energy of the final parameters (quality of what was learned)
  grad_sq0       squared gradient norm at step 0
  noise_penalty  E_clean_final(noise on) - E_final(noise off): how much worse the learned parameters are
                 when training sees device noise
NOTE: zero init with noise OFF should never move (exact zero gradient). In the transpiled Qiskit circuit the
gradient is ~1e-16 round-off instead of 0, and GD can amplify it away from the saddle at larger depth; those runs
are flagged as numerical artifacts (column zero_off_artifact) and drawn hollow.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
COL = {"gaussian": "#d62728", "uniform": "#1f77b4", "zero": "#2ca02c"}

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "e4b_runs*.csv")))], ignore_index=True)
df = df.drop_duplicates(subset=["backend", "qubits", "blocks", "init", "noise", "seed", "step"])
last = df.step.max()
keys = ["backend", "qubits", "blocks", "init", "noise"]
runs = []
for k, d in df.groupby(keys + ["seed"]):
    d = d.sort_values("step")
    clean = d.dropna(subset=["energy_noiseless"])
    runs.append(dict(zip(keys + ["seed"], k), E_start=d.energy.iloc[0], E_final=d.energy.iloc[-1],
                     E_clean_final=clean.energy_noiseless.iloc[-1], grad_sq0=d.grad_sq.iloc[0],
                     two_qubit_gates=d.two_qubit_gates.iloc[0], depth=d.depth.iloc[0], params=d.params.iloc[0],
                     circuit_evals=d.circuit_evals.iloc[-1], sec_per_step=d.wall_s.iloc[-1] / last, E0=d.E0.iloc[0]))
runs = pd.DataFrame(runs)
runs["zero_off_artifact"] = (runs.init == "zero") & (runs.noise == "off") & (runs.E_final < runs.E_start - 1e-6)
runs.to_csv(os.path.join(RES, "e4b_per_run.csv"), index=False)
summ = runs.groupby(keys).agg(seeds=("seed", "size"), E_start=("E_start", "mean"),
                              E_final_mean=("E_final", "mean"), E_final_std=("E_final", "std"),
                              E_clean_mean=("E_clean_final", "mean"), E_clean_std=("E_clean_final", "std"),
                              grad_sq0=("grad_sq0", "mean"), two_qubit_gates=("two_qubit_gates", "first"),
                              depth=("depth", "first"), params=("params", "first"),
                              circuit_evals=("circuit_evals", "first"), sec_per_step=("sec_per_step", "mean"),
                              zero_off_artifact=("zero_off_artifact", "any"), E0=("E0", "first")).reset_index()
off = summ[summ.noise == "off"].set_index(["backend", "qubits", "blocks", "init"])
on = summ[summ.noise == "on"].set_index(["backend", "qubits", "blocks", "init"])
pen = (on["E_clean_mean"] - off["E_final_mean"]).rename("noise_penalty")
summ = summ.merge(pen.reset_index(), on=["backend", "qubits", "blocks", "init"], how="left")
summ.loc[(summ.noise == "off") | (summ.init == "zero"), "noise_penalty"] = np.nan   # zero/off is the round-off artifact
summ.to_csv(os.path.join(RES, "e4b_summary.csv"), index=False)

for (backend, n), S in summ.groupby(["backend", "qubits"]):
    E0 = S.E0.iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    # (a) training curves at the deepest circuit
    bmax = S.blocks.max()
    ax = axes[0, 0]
    for init, c in COL.items():
        for noise, ls in (("on", "-"), ("off", "--")):
            d = df[(df.backend == backend) & (df.qubits == n) & (df.blocks == bmax) & (df.init == init) & (df.noise == noise)]
            if d.empty or (init == "zero" and noise == "off"):
                continue
            m = d.groupby("step").energy.agg(["mean", "std"])
            ax.plot(m.index, m["mean"], color=c, ls=ls, label=f"{init}, noise {noise}")
            ax.fill_between(m.index, m["mean"] - m["std"].fillna(0), m["mean"] + m["std"].fillna(0), color=c, alpha=0.15)
    ax.axhline(E0, color="k", ls=":", lw=1)
    ax.set_title(f"(a) training loss at {bmax} blocks (noise on: noisy energy)", fontsize=9)
    ax.set_xlabel("GD step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # (b) observed (noisy) final loss vs depth;  (c) noiseless energy of learned params vs depth
    for ax, col, err, title in ((axes[0, 1], "E_final_mean", "E_final_std", "(b) final training loss vs depth"),
                                (axes[1, 0], "E_clean_mean", "E_clean_std",
                                 "(c) NOISELESS energy of the learned parameters vs depth")):
        for init, c in COL.items():
            for noise, ls, mk in (("on", "-", "o"), ("off", "--", "s")):
                d = S[(S.init == init) & (S.noise == noise)].sort_values("blocks")
                if d.empty or (init == "zero" and noise == "off"):
                    continue   # zero/off: should stay at f = N-1; any movement is the round-off artifact
                art = d.zero_off_artifact.fillna(False).values
                ax.errorbar(d.blocks, d[col], yerr=d[err].fillna(0), color=c, ls=ls, marker=mk, capsize=2,
                            mfc="white" if (init == "zero" and noise == "off") else c, label=f"{init}, noise {noise}")
        ax.axhline(E0, color="k", ls=":", lw=1, label=f"$E_0$ = {E0:.3f}")
        ax.set_title(title, fontsize=9); ax.set_xlabel("blocks"); ax.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=7)
    # (d) gradient at init vs depth
    ax = axes[1, 1]
    for init in ("gaussian", "uniform"):
        for noise, ls, mk in (("on", "-", "o"), ("off", "--", "s")):
            d = S[(S.init == init) & (S.noise == noise)].sort_values("blocks")
            ax.plot(d.blocks, d.grad_sq0, color=COL[init], ls=ls, marker=mk, label=f"{init}, noise {noise}")
    ax.set_yscale("log"); ax.set_title("(d) $\\|\\nabla f\\|^2$ at initialization vs depth", fontsize=9)
    ax.set_xlabel("blocks"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    gates = ", ".join(f"{b}:{g}" for b, g in S.groupby("blocks").two_qubit_gates.first().items())
    fig.suptitle(f"E4b: {backend} noise model, N = {n}, GD lr 0.01, {last} steps, mean $\\pm$ std over seeds; "
                 f"2-qubit gates after transpile (blocks:count) {gates}", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"e4b_noise_{backend}_{n}q.{ext}"), dpi=150)
print("saved figures/e4b_noise_*.png/.pdf, results/e4b_summary.csv, results/e4b_per_run.csv")
