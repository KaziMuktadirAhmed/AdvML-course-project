"""E2: Heisenberg VQE training, reproducing Fig. 2 of Zhang et al. (plan Sec. 5, E2).

Grid: N = 15, 10 blocks (300 params), init {gaussian 1/160, uniform, zero} x opt {gd, adam} (lr 0.01)
      x gradient noise {0 (exact), 0.01} x seeds 0-4, 100 iterations each -> 60 runs.
Seeding: the initial parameters depend only on (seed, init), so GD/Adam and exact/noisy runs of the same seed start
from the same point; the gradient noise has its own stream per run.
Output: results/e2_runs.csv (one row per run per iteration). Resumable: finished runs are skipped, and a run is
written only once it completes. Use --max-seconds to stop cleanly before a time limit.

Usage: python e2_heisenberg.py [--steps 100] [--max-seconds 160]
"""
import argparse
import csv
import os
import time

import numpy as np

from ansatz import expval_qnode, param_shape, theorem_L
from hamiltonians import heisenberg_open_chain, ground_energy
from inits import gamma2, init_params
from train import make_optimizer, train

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e2_runs.csv")
INITS, OPTS, NOISES, SEEDS = ("gaussian", "uniform", "zero"), ("gd", "adam"), (0.0, 0.01), range(5)
FIELDS = ["qubits", "blocks", "init", "opt", "lr", "noise_var", "seed", "angle_scale", "E0",
          "step", "energy", "grad_sq_norm", "wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=15)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--angle-scale", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    t_start = time.perf_counter()

    n, blocks = a.qubits, a.blocks
    H = heisenberg_open_chain(n)
    e0 = ground_energy(H, n)
    f = expval_qnode(n, blocks, H, a.angle_scale)
    var = gamma2(2, theorem_L(blocks))

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for r in csv.DictReader(fh):
                if int(r["step"]) == a.steps:
                    done.add((r["init"], r["opt"], float(r["noise_var"]), int(r["seed"])))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    last = 0.0
    for seed in SEEDS:
        for init in INITS:
            for opt in OPTS:
                for nv in NOISES:
                    if (init, opt, nv, seed) in done:
                        continue
                    if time.perf_counter() - t_start + 1.3 * last > a.max_seconds:
                        print("Stopping before the time limit; re-run to resume.", flush=True)
                        fh.close()
                        return
                    t0 = time.perf_counter()
                    i_init = INITS.index(init)
                    p0 = init_params(init, param_shape(n, blocks), np.random.default_rng([seed, i_init]), var=var)
                    noise_rng = np.random.default_rng([seed, i_init, OPTS.index(opt), int(nv > 0), 99])
                    recs, _ = train(f, p0, make_optimizer(opt, a.lr), a.steps, nv, noise_rng)
                    w.writerows({"qubits": n, "blocks": blocks, "init": init, "opt": opt, "lr": a.lr,
                                 "noise_var": nv, "seed": seed, "angle_scale": a.angle_scale, "E0": e0, **r}
                                for r in recs)
                    fh.flush()
                    last = time.perf_counter() - t0
                    print(f"seed {seed} {init:8s} {opt:4s} noise {nv:<4g}: f {recs[0]['energy']:7.2f} -> "
                          f"{recs[-1]['energy']:7.2f}  ({last:.0f} s)", flush=True)
    fh.close()
    print("E2 done.")


if __name__ == "__main__":
    main()
