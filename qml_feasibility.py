"""
Feasibility test: Gaussian initialization (Zhang et al., NeurIPS 2022) on a QML classification task.

Task: binary classification of handwritten digits (sklearn's built-in 8x8 digits, no download),
two classes, PCA down to N features, one feature per qubit.
Model: angle encoding RY(x_i) on qubit i, then the paper's ansatz (blocks of CZ ring -> RX -> RY),
prediction = <Z> on qubit 0 (a LOCAL observable, S = 1), loss = mean squared error to labels +-1.
Paper angle convention for trainable gates (gate angle = 2*theta), so gamma^2 = 1/(4*S*(L+2)), L = 2*blocks - 2.

Part 1: mean squared gradient norm at initialization vs qubit count (does the task show a barren plateau?).
Part 2: short Adam training (lr 0.01) for Gaussian / uniform / zero init; test accuracy.

Usage:  python qml_feasibility.py            (about 2-5 minutes on a laptop CPU)
Requires: pennylane, pennylane-lightning, scikit-learn
"""
import time

import numpy as onp
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

CLASSES = (3, 8)       # a harder pair than 0 vs 1
BLOCKS = 10
SCALE = 2.0            # paper convention for trainable rotations
S = 1                  # observable Z on one qubit


def data(n_qubits, seed=0):
    X, y = load_digits(return_X_y=True)
    m = onp.isin(y, CLASSES)
    X, y = X[m], onp.where(y[m] == CLASSES[0], 1.0, -1.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    lo, hi = Ztr.min(0), Ztr.max(0)
    f = lambda Z: onp.clip((Z - lo) / (hi - lo), 0, 1) * onp.pi  # features in [0, pi]
    return f(Ztr), ytr, f(Zte), yte


def model(n, blocks):
    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, x):
        for q in range(n):
            qml.RY(x[q], wires=q)                       # data encoding (not trainable)
        for b in range(blocks):
            for q in range(n):
                qml.CZ(wires=[q, (q + 1) % n])
            for q in range(n):
                qml.RX(SCALE * params[b, 0, q], wires=q)
            for q in range(n):
                qml.RY(SCALE * params[b, 1, q], wires=q)
        return qml.expval(qml.PauliZ(0))
    return circuit


def init(kind, shape, rng, gamma2):
    if kind == "gaussian":
        return rng.normal(0, onp.sqrt(gamma2), shape)
    if kind == "uniform":
        return rng.uniform(0, 2 * onp.pi, shape)
    return onp.zeros(shape)


def loss_fn(circuit, params, X, y):
    preds = pnp.stack([circuit(params, x) for x in X])
    return pnp.mean((preds - y) ** 2)


def main():
    L = 2 * BLOCKS - 2
    gamma2 = 1.0 / (4 * S * (L + 2))
    print(f"Digits {CLASSES[0]} vs {CLASSES[1]}, {BLOCKS} blocks (L = {L}), S = {S}, gamma^2 = 1/{4*S*(L+2)}")

    # ---- Part 1: gradient norm at initialization vs qubits ----
    print("\nPart 1: mean ||grad loss||^2 at init (5 random inits, batch of 16 training images)")
    print(f"{'N':>3} {'params':>7} {'Gaussian':>10} {'uniform':>10} {'zero':>10} {'G/U':>7}  [s]")
    for n in (4, 6, 8, 10, 12):
        Xtr, ytr, _, _ = data(n)
        circuit = model(n, BLOCKS)
        shape = (BLOCKS, 2, n)
        rng = onp.random.default_rng(0)
        Xb, yb = Xtr[:16], ytr[:16]
        t0 = time.perf_counter()
        res = {}
        for kind in ("gaussian", "uniform", "zero"):
            vals = []
            for _ in range(1 if kind == "zero" else 5):
                p = pnp.array(init(kind, shape, rng, gamma2), requires_grad=True)
                g = qml.grad(lambda pp: loss_fn(circuit, pp, Xb, yb))(p)
                vals.append(float(onp.sum(onp.asarray(g) ** 2)))
            res[kind] = onp.mean(vals)
        print(f"{n:>3} {2*BLOCKS*n:>7} {res['gaussian']:>10.3e} {res['uniform']:>10.3e} {res['zero']:>10.3e} "
              f"{res['gaussian']/res['uniform']:>7.1f}  [{time.perf_counter()-t0:.0f}]", flush=True)

    # ---- Part 2: short training run ----
    n, steps, batch = 8, 60, 16
    Xtr, ytr, Xte, yte = data(n)
    circuit = model(n, BLOCKS)
    print(f"\nPart 2: training, N = {n}, {2*BLOCKS*n} params, Adam lr 0.01, {steps} steps, batch {batch}, "
          f"{len(Xtr)} train / {len(Xte)} test images")
    for kind in ("gaussian", "uniform", "zero"):
        accs, losses = [], []
        t0 = time.perf_counter()
        for seed in (0, 1):
            rng = onp.random.default_rng(seed)
            p = pnp.array(init(kind, (BLOCKS, 2, n), rng, gamma2), requires_grad=True)
            opt = qml.AdamOptimizer(0.01)
            for _ in range(steps):
                idx = rng.choice(len(Xtr), batch, replace=False)
                p = opt.step(lambda pp: loss_fn(circuit, pp, Xtr[idx], ytr[idx]), p)
            pred = onp.sign([float(circuit(p, x)) for x in Xte])
            accs.append(onp.mean(pred == yte))
            losses.append(float(loss_fn(circuit, p, Xte, yte)))
        print(f"{kind:>9}: test accuracy {onp.mean(accs):.3f} (seeds: {', '.join(f'{a:.3f}' for a in accs)}), "
              f"test MSE {onp.mean(losses):.3f}  [{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
