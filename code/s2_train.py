"""S2 training: QNN classifier (digits 3 vs 8) at N = 12 qubits, 10 blocks (240 params).

Grid: init {gaussian 1/80, uniform, zero} x seeds 0-4, optimizer Adam (plan) and optionally GD, lr 0.01, 100 steps,
mini-batch 16. Pairing: within a seed, all inits see the SAME mini-batch sequence, so the comparison is paired.
Logged every step: exact batch-gradient squared norm. Every 10 steps: full train and test MSE / accuracy.
Output: results/s2_train.csv (one row per logged step); resumable per run (a run is written when it completes).

Usage: python s2_train.py [--opts adam gd] [--max-seconds 160]
"""
import argparse
import csv
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz import param_shape, theorem_L
from inits import gamma2, init_params
from qnn import S, load_data, make_model, metrics, mse
from train import make_optimizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "s2_train.csv")
INITS = ("gaussian", "uniform", "zero")
FIELDS = ["qubits", "blocks", "init", "opt", "lr", "seed", "batch", "step", "batch_grad_sq", "train_mse", "train_acc",
          "test_mse", "test_acc", "wall_s"]


def run(circuit, data, init, opt_name, seed, a):
    Xtr, ytr, Xte, yte = data
    n, blocks = a.qubits, a.blocks
    p = init_params(init, param_shape(n, blocks), np.random.default_rng([seed, INITS.index(init)]),
                    var=gamma2(S, theorem_L(blocks)))
    batch_rng = np.random.default_rng([seed, 11])       # same batches for every init
    opt = make_optimizer(opt_name, a.lr)
    recs, t0 = [], time.perf_counter()
    for t in range(a.steps + 1):
        idx = batch_rng.choice(len(Xtr), a.batch, replace=False)
        rec = {"step": t}
        if t % a.eval_every == 0 or t == a.steps:
            rec["train_mse"], rec["train_acc"] = metrics(circuit, p, Xtr, ytr)
            rec["test_mse"], rec["test_acc"] = metrics(circuit, p, Xte, yte)
        if t < a.steps:
            Xb, yb = Xtr[idx], ytr[idx]
            g = np.asarray(qml.grad(lambda pp: mse(circuit, pp, Xb, yb))(pnp.array(p, requires_grad=True)))
            rec["batch_grad_sq"] = float(np.sum(g ** 2))
            p = opt.step(p, g)
        rec["wall_s"] = time.perf_counter() - t0
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=12)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--opts", nargs="+", default=["adam"])
    ap.add_argument("--max-seconds", type=float, default=1e9)
    a = ap.parse_args()
    t_start = time.perf_counter()

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            if int(r["step"]) == a.steps and int(r["qubits"]) == a.qubits:
                done.add((r["init"], r["opt"], int(r["seed"])))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    data = load_data(a.qubits)
    circuit = make_model(a.qubits, a.blocks)
    last = 0.0
    for opt_name in a.opts:
        for seed in a.seeds:
            for init in INITS:
                if (init, opt_name, seed) in done:
                    continue
                if time.perf_counter() - t_start + 1.2 * last > a.max_seconds:
                    print("Stopping before the time limit; re-run to resume.", flush=True)
                    return
                t0 = time.perf_counter()
                recs = run(circuit, data, init, opt_name, seed, a)
                w.writerows({"qubits": a.qubits, "blocks": a.blocks, "init": init, "opt": opt_name, "lr": a.lr,
                             "seed": seed, "batch": a.batch, **r} for r in recs)
                fh.flush()
                last = time.perf_counter() - t0
                print(f"{opt_name:4s} seed {seed} {init:8s}: test acc {recs[0]['test_acc']:.3f} -> "
                      f"{recs[-1]['test_acc']:.3f}, test MSE {recs[0]['test_mse']:.3f} -> {recs[-1]['test_mse']:.3f} "
                      f"({last:.0f} s)", flush=True)
    fh.close()
    print("S2 training done.")


if __name__ == "__main__":
    main()
