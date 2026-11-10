"""Sanity check of all project outputs (writes results/sanity_check.txt).

1. Every expected data file and figure exists and is non-empty; figures are newer than their data.
2. Raw CSVs are complete: row/run counts match the experiment grids, no duplicate keys, no NaNs where there
   should be values, every run has steps 0..100 exactly once, no unfinished E4b checkpoints.
3. Headline numbers quoted in claude/results-log.md are recomputed from the RAW data.
4. All scripts compile.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.normpath(os.path.join(HERE, ".."))
R, F = os.path.join(P, "results"), os.path.join(P, "figures")
out, nfail = [], 0


def check(name, ok, detail=""):
    global nfail
    nfail += 0 if ok else 1
    s = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else "")
    print(s, flush=True)
    out.append(s)


def close(a, b, tol):
    return abs(a - b) <= tol


# ---- 1. files -------------------------------------------------------------------------------------------------
data_fig = {
    "e1_gradnorm_raw.csv": ["e1_gradnorm.png", "e1_gradnorm.pdf"],
    "e2_runs.csv": ["e2_training.png", "e2_training.pdf"],
    "s2_gradsweep_raw.csv": ["s2_gradsweep.png", "s2_gradsweep.pdf"],
    "s2_train.csv": ["s2_training.png", "s2_training.pdf"],
    "e4b_runs.csv": ["e4b_noise_FakeManilaV2_5q.png", "e4b_noise_FakeManilaV2_5q.pdf",
                     "e4b_noise_FakeNairobiV2_7q.png", "e4b_noise_FakeNairobiV2_7q.pdf"],
}
derived = ["e0_validation.txt", "e1_summary.csv", "e1_decay_fits.csv", "e2_summary.csv", "e2_per_run.csv",
           "s2_gradsweep_summary.csv", "s2_train_summary.csv", "s2_paired.csv", "e4b_summary.csv", "e4b_per_run.csv",
           "noise_depth_sweep_FakeManilaV2_5q_scale2.csv"]
for d, figs in data_fig.items():
    dp = os.path.join(R, d)
    check(f"data file {d}", os.path.isfile(dp) and os.path.getsize(dp) > 0,
          f"{os.path.getsize(dp) / 1e6:.2f} MB" if os.path.isfile(dp) else "MISSING")
    newest_data = max(os.path.getmtime(f) for f in glob.glob(os.path.join(R, d.replace(".csv", "*.csv"))))
    for fg in figs:
        fp = os.path.join(F, fg)
        ok = os.path.isfile(fp) and os.path.getsize(fp) > 1000 and os.path.getmtime(fp) >= newest_data
        check(f"figure {fg} exists and is newer than its data", ok,
              f"{os.path.getsize(fp) / 1e3:.0f} kB" if os.path.isfile(fp) else "MISSING")
for d in derived:
    dp = os.path.join(R, d)
    check(f"derived file {d}", os.path.isfile(dp) and os.path.getsize(dp) > 0)

# ---- 2. completeness ------------------------------------------------------------------------------------------
e1 = pd.read_csv(os.path.join(R, "e1_gradnorm_raw.csv"))
cells = e1.groupby(["qubits", "blocks", "observable", "init"]).size()
exp_cells = 6 * 4 * 2 * 3
check("E1 rows / cells", len(e1) == 6 * 4 * 2 * 201 and len(cells) == exp_cells
      and set(cells[cells.index.get_level_values("init") != "zero"]) == {100}
      and set(cells[cells.index.get_level_values("init") == "zero"]) == {1},
      f"{len(e1)} rows (expect 9648), {len(cells)} cells (expect {exp_cells})")
check("E1 no duplicate draws / NaNs", not e1.duplicated(["qubits", "blocks", "observable", "init", "draw"]).any()
      and e1.sq_norm.notna().all())
check("E1 zero-init gradient exactly 0", e1[e1.init == "zero"].sq_norm.max() == 0.0)


def runs_ok(df, keys, steps=100):
    g = df.groupby(keys).step.agg(["count", "min", "max", "nunique"])
    return g, bool(((g["count"] == steps + 1) & (g["min"] == 0) & (g["max"] == steps) & (g["nunique"] == steps + 1)).all())


e2 = pd.read_csv(os.path.join(R, "e2_runs.csv"))
g, ok = runs_ok(e2, ["init", "opt", "noise_var", "seed"])
check("E2 60 complete runs, steps 0..100 once each", ok and len(g) == 60, f"{len(g)} runs, {len(e2)} rows")
check("E2 no NaN energies", e2.energy.notna().all() and e2.grad_sq_norm.notna().all())

s2g = pd.read_csv(os.path.join(R, "s2_gradsweep_raw.csv"))
c = s2g.groupby(["qubits", "init"]).size()
check("S2 sweep rows", len(s2g) == 6 * 210 and len(c) == 18 and not s2g.duplicated(["qubits", "init", "draw"]).any(),
      f"{len(s2g)} rows (expect 1260)")

s2 = pd.read_csv(os.path.join(R, "s2_train.csv"))
g, ok = runs_ok(s2, ["opt", "init", "seed"])
ev = s2.dropna(subset=["test_mse"])
check("S2 30 complete runs (Adam + GD), steps 0..100 once each", ok and len(g) == 30, f"{len(g)} runs")
check("S2 eval every 10 steps (11 evals per run)", set(ev.groupby(["opt", "init", "seed"]).size()) == {11})

e4 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "e4b_runs*.csv")))], ignore_index=True)
key = ["backend", "qubits", "blocks", "init", "noise", "seed"]
g, ok = runs_ok(e4, key)
check("E4b all runs complete, steps 0..100 once each (across all e4b_runs*.csv files)", ok and len(g) == 116,
      f"{len(g)} runs in {len(glob.glob(os.path.join(R, 'e4b_runs*.csv')))} files (expect 116)")
grid = e4.drop_duplicates(key).groupby(["qubits", "blocks", "noise"]).size()
exp = {(5, 1): 9, (5, 2): 7, (5, 3): 7, (5, 4): 7, (5, 6): 7, (5, 8): 7, (7, 2): 7, (7, 3): 7}
check("E4b grid: G/U x 3 seeds + zero per (N, blocks, noise)",
      all(grid.get((q, b, nz), 0) == n for (q, b), n in exp.items() for nz in ("on", "off")),
      "5q blocks 1 has 3 identical zero-init seeds from before the zero-seed-0 rule; harmless")
z1 = e4[(e4.qubits == 5) & (e4.blocks == 1) & (e4.init == "zero")]
check("E4b duplicated zero-init seeds at 5q/1 block are identical", z1.groupby(["noise", "step"]).energy.nunique().max() == 1)
pending = [f for f in glob.glob(os.path.join(R, "e4b_ckpt", "*.json"))]
check("E4b no unfinished checkpoints", len(pending) == 0, f"{len(pending)} pending .json")
check("E4b noise-on runs have noiseless evaluations every 10 steps",
      set(e4[e4.noise == "on"].dropna(subset=["energy_noiseless"]).groupby(key).size()) == {11})

# ---- 3. headline numbers from raw data ------------------------------------------------------------------------
m = e1.assign(pp=e1.sq_norm / e1.params).groupby(["observable", "blocks", "qubits", "init"]).pp.mean()
rH = m["H", 20, 14, "gaussian"] / m["H", 20, 14, "uniform"]
rZ = m["Z1Z2", 20, 14, "gaussian"] / m["Z1Z2", 20, 14, "uniform"]
check("E1 headline: Gaussian/uniform at N=14, 20 blocks = 586x (H), 610x (Z1Z2)",
      close(rH, 586, 2) and close(rZ, 610, 2), f"{rH:.1f}x, {rZ:.1f}x")
fin = e2[e2.step == 100].groupby(["opt", "noise_var", "init"]).energy.mean()
hit = e2[e2.energy <= -20].groupby(["opt", "noise_var", "init", "seed"]).step.min().groupby(["opt", "noise_var", "init"]).mean()
check("E2 headline: GD exact f(100) G -22.85 / U -21.77; iters to -20: 16.8 / 71.4",
      close(fin["gd", 0.0, "gaussian"], -22.85, 0.01) and close(fin["gd", 0.0, "uniform"], -21.77, 0.01)
      and close(hit["gd", 0.0, "gaussian"], 16.8, 0.01) and close(hit["gd", 0.0, "uniform"], 71.4, 0.01),
      f"{fin['gd', 0.0, 'gaussian']:.3f} / {fin['gd', 0.0, 'uniform']:.3f}; "
      f"{hit['gd', 0.0, 'gaussian']:.1f} / {hit['gd', 0.0, 'uniform']:.1f}")
check("E2 headline: Adam exact f(100) G -22.97 / U -23.50",
      close(fin["adam", 0.0, "gaussian"], -22.97, 0.01) and close(fin["adam", 0.0, "uniform"], -23.50, 0.01),
      f"{fin['adam', 0.0, 'gaussian']:.3f} / {fin['adam', 0.0, 'uniform']:.3f}")
s2m = s2g.groupby(["qubits", "init"]).sq_norm.mean()
check("S2 sweep headline: Gaussian/uniform = 71x (N=12), 100x (N=14)",
      close(s2m[12, "gaussian"] / s2m[12, "uniform"], 71.1, 0.2) and close(s2m[14, "gaussian"] / s2m[14, "uniform"], 100.3, 0.2),
      f"{s2m[12, 'gaussian'] / s2m[12, 'uniform']:.1f}x, {s2m[14, 'gaussian'] / s2m[14, 'uniform']:.1f}x")
tm = ev[ev.step == 100].groupby(["opt", "init"]).test_mse.mean()
check("S2 headline: final test MSE Adam G 0.251 / U 0.273; GD G 0.281 / U 0.921",
      close(tm["adam", "gaussian"], 0.251, 0.001) and close(tm["adam", "uniform"], 0.273, 0.001)
      and close(tm["gd", "gaussian"], 0.281, 0.001) and close(tm["gd", "uniform"], 0.921, 0.001),
      ", ".join(f"{k[0]} {k[1]} {v:.3f}" for k, v in tm.items()))
f4 = e4[e4.step == 100].drop_duplicates(key).groupby(["qubits", "blocks", "init", "noise"]).energy.mean()
check("E4b headline: 5q 8 blocks noisy loss G -3.19 / U -1.00; noiseless G -7.23 / U -7.54",
      close(f4[5, 8, "gaussian", "on"], -3.19, 0.005) and close(f4[5, 8, "uniform", "on"], -1.00, 0.005)
      and close(f4[5, 8, "gaussian", "off"], -7.23, 0.005) and close(f4[5, 8, "uniform", "off"], -7.54, 0.005),
      f"{f4[5, 8, 'gaussian', 'on']:.3f} / {f4[5, 8, 'uniform', 'on']:.3f}; "
      f"{f4[5, 8, 'gaussian', 'off']:.3f} / {f4[5, 8, 'uniform', 'off']:.3f}")
check("E4b headline: 7q noisy loss G -6.56 / U -5.94 (2 blocks), -6.50 / -5.70 (3 blocks)",
      all(close(f4[7, b, i, "on"], v, 0.005) for (b, i), v in
          {(2, "gaussian"): -6.56, (2, "uniform"): -5.94, (3, "gaussian"): -6.50, (3, "uniform"): -5.70}.items()))
gap = {b: f4[5, b, "gaussian", "on"] - f4[5, b, "uniform", "on"] for b in (1, 2, 3, 4, 6, 8)}
check("E4b headline: noisy G-U gap grows with depth (0.0 -> 2.19), monotone after 1 block",
      close(-gap[8], 2.19, 0.01) and all(-gap[a] < -gap[b] for a, b in ((2, 3), (3, 4), (4, 6), (6, 8))),
      ", ".join(f"{b}:{-v:.2f}" for b, v in gap.items()))

# ---- 4. scripts compile ---------------------------------------------------------------------------------------
bad = []
for f in glob.glob(os.path.join(HERE, "*.py")):
    try:
        compile(open(f).read(), f, "exec")
    except SyntaxError:
        bad.append(os.path.basename(f))
check("all scripts in code/ compile", not bad, f"{len(glob.glob(os.path.join(HERE, '*.py')))} files" + (f"; bad: {bad}" if bad else ""))

out.append(f"\n{'ALL CHECKS PASS' if nfail == 0 else f'{nfail} CHECK(S) FAILED'}")
print(out[-1])
open(os.path.join(R, "sanity_check.txt"), "w").write("\n".join(out) + "\n")
