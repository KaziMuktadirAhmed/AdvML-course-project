"""S2 gradient sweep: squared gradient norm of the classifier's MSE loss at initialization vs qubits N.

N in {4,...,14}, 10 blocks, init {gaussian (S = 1: gamma^2 = 1/(4(L+2))), uniform, zero}, 100 draws each
(zero: 10 draws, since only the batch varies). Each draw uses a fresh random batch of 16 training images.
Output: results/s2_gradsweep_raw.csv (one row per draw); resumable per (N, init) cell.
"""
import argparse
import csv
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz import param_shape, n_params, theorem_L
from inits import gamma2, init_params
from qnn import S, load_data, make_model, mse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "s2_gradsweep_raw.csv")
FIELDS = ["qubits", "blocks", "params", "init", "var", "draw", "sq_norm", "g000_sq", "sec_per_grad", "angle_scale"]
INITS = ("gaussian", "uniform", "zero")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--angle-scale", type=float, default=2.0)
    a = ap.parse_args()

    done = {}
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            k = (int(r["qubits"]), int(r["blocks"]), r["init"])
            done[k] = done.get(k, 0) + 1
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    blocks = a.blocks
    var = gamma2(S, theorem_L(blocks))
    for n in a.qubits:
        Xtr, ytr, _, _ = load_data(n)
        circuit = make_model(n, blocks, a.angle_scale)
        for init in INITS:
            want = 10 if init == "zero" else a.draws
            if done.get((n, blocks, init), 0) >= want:
                continue
            rng = np.random.default_rng([n, blocks, INITS.index(init), 7])
            rows, t_cell = [], time.perf_counter()
            for d in range(want):
                p = pnp.array(init_params(init, param_shape(n, blocks), rng, var=var), requires_grad=True)
                idx = rng.choice(len(Xtr), a.batch, replace=False)
                Xb, yb = Xtr[idx], ytr[idx]
                t0 = time.perf_counter()
                g = np.asarray(qml.grad(lambda pp: mse(circuit, pp, Xb, yb))(p))
                rows.append({"qubits": n, "blocks": blocks, "params": n_params(n, blocks), "init": init,
                             "var": var if init == "gaussian" else "", "draw": d, "sq_norm": float(np.sum(g ** 2)),
                             "g000_sq": float(g[0, 0, 0] ** 2), "sec_per_grad": time.perf_counter() - t0,
                             "angle_scale": a.angle_scale})
            w.writerows(rows)
            fh.flush()
            print(f"N={n:2d} {init:8s} E||grad L||^2 = {np.mean([r['sq_norm'] for r in rows]):.4e} "
                  f"({time.perf_counter() - t_cell:.0f} s)", flush=True)
    fh.close()
    print("S2 sweep done.")


if __name__ == "__main__":
    main()
