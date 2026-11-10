"""E1: squared gradient norm at initialization vs qubits N and depth (plan Sec. 5, E1).

Grid: N in {4,6,8,10,12,14} x blocks in {2,5,10,20} x init {gaussian, uniform, zero} x draws (default 100),
for two observables: the full Heisenberg H (Eq. 10) and a single Z_1 Z_2 term (Theorem 4.1 applies directly).
Gaussian variance follows the depth: gamma^2 = 1/(4 S (L+2)) with S = 2, L = 2*blocks - 2.
Zero init is deterministic, so it gets 1 draw.

One row per draw in results/e1_gradnorm_raw.csv. Re-running resumes: finished (N, blocks, obs, init) cells are skipped.
Columns: sq_norm = ||grad f||^2; g000_sq = (df/d theta[block 0, RX, qubit 0])^2 (the parameter farthest from the
measurement, the classic barren-plateau probe); meas_sq = squared gradient restricted to the measured qubits 0, 1.

Usage: python e1_gradnorm.py [--draws 100] [--qubits 4 6 8] [--blocks 2 5 10 20]
"""
import argparse
import csv
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz import expval_qnode, param_shape, n_params, theorem_L
from hamiltonians import heisenberg_open_chain, zz
from inits import gamma2, init_params

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e1_gradnorm_raw.csv")
FIELDS = ["qubits", "blocks", "L", "params", "observable", "init", "var", "draw", "sq_norm", "g000_sq", "meas_sq",
          "sec_per_grad", "angle_scale"]


def done_cells():
    if not os.path.exists(OUT):
        return set()
    cells = {}
    with open(OUT) as fh:
        for r in csv.DictReader(fh):
            k = (int(r["qubits"]), int(r["blocks"]), r["observable"], r["init"])
            cells[k] = cells.get(k, 0) + 1
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--blocks", type=int, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--angle-scale", type=float, default=2.0)
    a = ap.parse_args()

    done = done_cells()
    new_file = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for n in a.qubits:
        observables = {"H": heisenberg_open_chain(n), "Z1Z2": zz(0, 1)}
        for blocks in a.blocks:
            L, shape = theorem_L(blocks), param_shape(n, blocks)
            var = gamma2(2, L)
            for oname, obs in observables.items():
                grad = qml.grad(expval_qnode(n, blocks, obs, a.angle_scale))
                for init in ("gaussian", "uniform", "zero"):
                    want = 1 if init == "zero" else a.draws
                    have = done.get((n, blocks, oname, init), 0) if isinstance(done, dict) else 0
                    if have >= want:
                        continue
                    # seed depends only on the cell, so a resumed cell restarts from draw 0 reproducibly
                    rng = np.random.default_rng([n, blocks, 0 if oname == "H" else 1,
                                                 ("gaussian", "uniform", "zero").index(init)])
                    t_cell = time.perf_counter()
                    rows = []
                    for d in range(want):
                        p = pnp.array(init_params(init, shape, rng, var=var), requires_grad=True)
                        t0 = time.perf_counter()
                        g = np.asarray(grad(p))
                        dt = time.perf_counter() - t0
                        rows.append({"qubits": n, "blocks": blocks, "L": L, "params": n_params(n, blocks),
                                     "observable": oname, "init": init, "var": var if init == "gaussian" else "",
                                     "draw": d, "sq_norm": float(np.sum(g ** 2)), "g000_sq": float(g[0, 0, 0] ** 2),
                                     "meas_sq": float(np.sum(g[:, :, :2] ** 2)), "sec_per_grad": dt,
                                     "angle_scale": a.angle_scale})
                    w.writerows(rows)   # write whole cells only, so a crash never leaves a partial cell
                    fh.flush()
                    m = np.mean([r["sq_norm"] for r in rows])
                    print(f"N={n:2d} blocks={blocks:2d} {oname:4s} {init:8s} E||g||^2={m:10.4e} "
                          f"({time.perf_counter() - t_cell:6.1f} s)", flush=True)
    fh.close()
    print("E1 done.")


if __name__ == "__main__":
    main()
