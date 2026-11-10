"""Training loop for the Heisenberg VQE (E2) and reusable optimizers.

Gradient noise model (plan Sec. 8, item 2; the paper gives no formula): independent N(0, noise_var) noise added to
every gradient component at every step, in paper units. The logged grad norm is the EXACT (noiseless) gradient.
Optimizers mirror PennyLane's GradientDescentOptimizer / AdamOptimizer (beta1 0.9, beta2 0.999, eps 1e-8).

CLI example (one E2 run):
  python train.py --qubits 15 --blocks 10 --init gaussian --opt gd --lr 0.01 --steps 100 --noise-var 0.01 --seed 0
"""
import argparse
import csv
import os
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz import expval_qnode, param_shape, theorem_L
from hamiltonians import heisenberg_open_chain, ground_energy
from inits import gamma2, init_params


class GD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, p, g):
        return p - self.lr * g


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, p, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(g), np.zeros_like(g)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        lr_t = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        return p - lr_t * self.m / (np.sqrt(self.v) + self.eps)


def make_optimizer(name, lr):
    return {"gd": GD, "adam": Adam}[name](lr)


def value_and_grad(qnode):
    g = qml.grad(qnode)

    def vg(p):
        pp = pnp.array(p, requires_grad=True)
        return float(qnode(pp)), np.asarray(g(pp), dtype=float)

    return vg


def train(qnode, params0, opt, steps, noise_var=0.0, rng=None):
    """Returns a list of per-step records: step, energy, exact grad_sq_norm, cumulative wall-clock seconds."""
    vg = value_and_grad(qnode)
    p = np.array(params0, dtype=float)
    recs, t_start = [], time.perf_counter()
    for t in range(steps + 1):
        e, g = vg(p)
        recs.append({"step": t, "energy": e, "grad_sq_norm": float(np.sum(g ** 2)),
                     "wall_s": time.perf_counter() - t_start})
        if t == steps:
            break
        if noise_var > 0:
            g = g + rng.normal(0.0, np.sqrt(noise_var), size=g.shape)
        p = opt.step(p, g)
    return recs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=15)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--init", choices=["gaussian", "uniform", "zero"], default="gaussian")
    ap.add_argument("--opt", choices=["gd", "adam"], default="gd")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--noise-var", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angle-scale", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n, blocks = a.qubits, a.blocks
    H = heisenberg_open_chain(n)
    f = expval_qnode(n, blocks, H, a.angle_scale)
    rng = np.random.default_rng(a.seed)
    p0 = init_params(a.init, param_shape(n, blocks), rng, var=gamma2(2, theorem_L(blocks)))
    recs, _ = train(f, p0, make_optimizer(a.opt, a.lr), a.steps, a.noise_var, rng)
    e0 = ground_energy(H, n)
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "results",
                                f"train_{n}q_{a.init}_{a.opt}_nv{a.noise_var:g}_s{a.seed}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["qubits", "blocks", "init", "opt", "lr", "noise_var", "seed",
                                           "angle_scale", "E0"] + list(recs[0].keys()))
        w.writeheader()
        for r in recs:
            w.writerow({"qubits": n, "blocks": blocks, "init": a.init, "opt": a.opt, "lr": a.lr,
                        "noise_var": a.noise_var, "seed": a.seed, "angle_scale": a.angle_scale, "E0": e0, **r})
    print(f"E0 = {e0:.4f}; energy {recs[0]['energy']:.3f} -> {recs[-1]['energy']:.3f} in {a.steps} steps "
          f"({recs[-1]['wall_s']:.1f} s). Saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
