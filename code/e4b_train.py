"""E4b: Heisenberg VQE training in a noisy hardware simulation (Qiskit Aer + IBM fake-backend noise model).

Grid (default): FakeManilaV2, N = 5, blocks {1,2,3,4} (plan core) + {6,8} (extension), init {gaussian, uniform, zero},
device noise {off, on}, seeds 0-2, GD lr 0.01, 100 steps. Exact density-matrix expectations, no shot noise.
The noise-on and noise-off runs of a (blocks, init, seed) start from the same parameters.
Logged per step: training-objective energy (noisy if noise on), squared norm of the gradient used, cumulative
circuit evaluations. Every 10 steps (noise-on runs): the NOISELESS energy of the current parameters, i.e. how good
the learned parameters are once device noise is removed.
Output: results/e4b_runs.csv. Resumable at step level: an unfinished run is checkpointed to results/e4b_ckpt/.

Zero init is run for seed 0 only (the noise model is deterministic, so all seeds give the same run).
Usage: python e4b_train.py [--backend FakeNairobiV2 --qubits 7 --blocks 2 3] [--max-seconds 160]
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from ansatz import param_shape, theorem_L
from hamiltonians import heisenberg_open_chain, ground_energy
from inits import gamma2, init_params
from noisy import NoisyVQE

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT, CKPT = os.path.join(RES, "e4b_runs.csv"), os.path.join(RES, "e4b_ckpt")
INITS = ("gaussian", "uniform", "zero")
FIELDS = ["backend", "qubits", "blocks", "params", "two_qubit_gates", "depth", "init", "noise", "seed", "lr", "E0",
          "step", "energy", "energy_noiseless", "grad_sq", "circuit_evals", "wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="FakeManilaV2")
    ap.add_argument("--qubits", type=int, default=5)
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--noise", nargs="+", default=["off", "on"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--inits", nargs="+", default=list(INITS), help="subset of inits (to split work across processes)")
    ap.add_argument("--tag", default="", help="suffix for the output CSV, so parallel processes never share a file")
    a = ap.parse_args()
    global OUT
    OUT = os.path.join(RES, f"e4b_runs{a.tag}.csv")
    t_start = time.perf_counter()
    os.makedirs(CKPT, exist_ok=True)

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            if int(r["step"]) == a.steps:
                done.add((r["backend"], int(r["qubits"]), int(r["blocks"]), r["init"], r["noise"], int(r["seed"])))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    n = a.qubits
    E0 = ground_energy(heisenberg_open_chain(n), n)
    for blocks in a.blocks:
        model = None
        for seed in a.seeds:
            for init in INITS:
                if init not in a.inits:
                    continue
                for noise in a.noise:
                    key = (a.backend, n, blocks, init, noise, seed)
                    if key in done:
                        continue
                    if init == "zero" and seed != 0:
                        continue    # zero init + deterministic noise model -> every seed is identical
                    if time.perf_counter() - t_start > a.max_seconds:
                        print("Time budget used; re-run to resume.", flush=True)
                        return
                    if model is None:
                        model = NoisyVQE(a.backend, n, blocks)
                    ck = os.path.join(CKPT, "_".join(map(str, key)) + ".json")
                    if os.path.exists(ck):
                        st = json.load(open(ck))
                        p, recs, t0_wall = np.array(st["p"]), st["recs"], st["wall"]
                    else:
                        p = init_params(init, param_shape(n, blocks),
                                        np.random.default_rng([seed, INITS.index(init), blocks]),
                                        var=gamma2(2, theorem_L(blocks))).reshape(-1)
                        recs, t0_wall = [], 0.0
                    t_run = time.perf_counter()
                    evals = recs[-1]["circuit_evals"] if recs else 0
                    t = len(recs)
                    while t <= a.steps:
                        if time.perf_counter() - t_start > a.max_seconds:
                            json.dump({"p": p.tolist(), "recs": recs,
                                       "wall": t0_wall + time.perf_counter() - t_run}, open(ck, "w"))
                            print(f"checkpoint {key} at step {t}; re-run to resume.", flush=True)
                            fh.close()
                            return
                        e, g = model.value_and_grad(p, noise)
                        evals += 2 * model.P + 1
                        e_clean = ""
                        if noise == "on" and (t % 10 == 0 or t == a.steps):
                            e_clean = float(model.values(p, "off")[0])
                        elif noise == "off":
                            e_clean = e
                        recs.append({"step": t, "energy": e, "energy_noiseless": e_clean, "grad_sq": float(g @ g),
                                     "circuit_evals": evals, "wall_s": t0_wall + time.perf_counter() - t_run})
                        if t < a.steps:
                            p = p - a.lr * g
                        t += 1
                    meta = {"backend": a.backend, "qubits": n, "blocks": blocks, "params": model.P,
                            "two_qubit_gates": model.two_qubit_gates, "depth": model.depth, "init": init,
                            "noise": noise, "seed": seed, "lr": a.lr, "E0": E0}
                    w.writerows({**meta, **r} for r in recs)
                    fh.flush()
                    if os.path.exists(ck):
                        os.replace(ck, ck + ".done")
                    print(f"{a.backend} N={n} blocks={blocks} seed {seed} {init:8s} noise {noise:3s}: "
                          f"E {recs[0]['energy']:7.3f} -> {recs[-1]['energy']:7.3f} "
                          f"(noiseless {recs[-1]['energy_noiseless']:7.3f}; E0 {E0:.3f})  "
                          f"[{recs[-1]['wall_s']:.0f} s]", flush=True)
    fh.close()
    print("E4b grid done.")


if __name__ == "__main__":
    main()
