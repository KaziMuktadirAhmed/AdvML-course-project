"""
Smoke test for reproducing:
  K. Zhang et al., "Escaping from the Barren Plateau via Gaussian Initializations
  in Deep Variational Quantum Circuits," NeurIPS 2022 (arXiv:2203.09376).

What it does (default: 10 qubits, 10 blocks -> 200 parameters):
  1. Builds the paper's Heisenberg-model setup: open chain H = sum_i X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}
     and a hardware-efficient ansatz of (RY layer, RX layer, CZ ring) blocks.
  2. Compares three initializations: Gaussian N(0, 1/(4S(L+2))), uniform [0, 2*pi], and zeros.
  3. Measures the squared gradient norm at initialization (averaged over random draws).
  4. Runs a few Adam steps (lr = 0.01, as in the paper) and records energy and time per step.
  5. Prints a summary table and writes smoke_results_<N>q_scale<k>.csv (no more overwriting).

Angles: by default parameters are in the paper's convention RX(theta) = exp(-i theta X), i.e. the gate
gets PennyLane angle 2*theta (see build_circuit). Use --angle-scale 1 for raw PennyLane angles.

Usage:
  python smoke_test.py                      # default 10 qubits, 10 blocks
  python smoke_test.py --qubits 15 --blocks 10 --steps 20   # the paper's full size

Requires: pip install pennylane pennylane-lightning scipy
"""
import argparse
import csv
import time

import numpy as onp
import pennylane as qml
from pennylane import numpy as pnp
from scipy.sparse.linalg import eigsh


def heisenberg_open_chain(n):
    """Paper Eq. (10): H = sum_{i=1}^{N-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}), open boundary."""
    coeffs, ops = [], []
    for i in range(n - 1):
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            coeffs.append(1.0)
            ops.append(P(i) @ P(i + 1))
    return qml.Hamiltonian(coeffs, ops)


def build_circuit(n, blocks, dev, diff_method, angle_scale=2.0):
    """
    Paper, p. 7: "In each block, we first employ the CZ gate to neighboring qubits pairs
    {(1,2), ..., (N,1)}, followed by RX and RY rotations for all qubits."
    So each block = CZ ring -> RX layer -> RY layer. Parameters = 2 * blocks * n
    (15 qubits x 10 blocks = 300, as in the paper). The circuit therefore ends with an RX layer
    then an RY layer, matching Fig. 1 (R_X(theta_{L+1}), R_Y(theta_{L+2})) and the proof of
    Theorem 4.1 (supplement Eqs. 110-114). The first CZ ring acts on |0...0> and does nothing.
    Theorem counting: 2*blocks rotation layers = L + 2, so L = 2*blocks - 2 (10 blocks -> L = 18).

    ANGLE CONVENTION: the paper (Sec. 2) defines RX(theta) = exp(-i theta X), but PennyLane's
    qml.RX(t) = exp(-i t X / 2). So paper angle theta == PennyLane angle 2*theta. With
    angle_scale = 2 (default) the parameters are in the PAPER's units: the variance 1/160, the
    learning rates, the gradient norms and the gradient-noise variance all match the paper.
    angle_scale = 1 treats PennyLane angles as the parameters (what the first smoke tests did).
    """
    H = heisenberg_open_chain(n)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):  # params shape: (blocks, 2, n); [:, 0, :] = RX angles, [:, 1, :] = RY angles
        for b in range(blocks):
            for q in range(n):  # CZ ring: (0,1), (1,2), ..., (n-1,0)
                qml.CZ(wires=[q, (q + 1) % n])
            for q in range(n):
                qml.RX(angle_scale * params[b, 0, q], wires=q)
            for q in range(n):
                qml.RY(angle_scale * params[b, 1, q], wires=q)
        return qml.expval(H)

    return circuit, H


def init_params(kind, blocks, n, sigma2, rng):
    shape = (blocks, 2, n)
    if kind == "gaussian":
        return rng.normal(0.0, onp.sqrt(sigma2), size=shape)
    if kind == "uniform":
        return rng.uniform(0.0, 2 * onp.pi, size=shape)
    if kind == "zero":
        return onp.zeros(shape)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--draws", type=int, default=10, help="random inits for the gradient-norm estimate")
    ap.add_argument("--steps", type=int, default=10, help="Adam steps per init (lr = 0.01)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angle-scale", type=float, default=2.0,
                    help="2 = paper convention exp(-i theta G) (default); 1 = PennyLane angles")
    args = ap.parse_args()

    n, blocks = args.qubits, args.blocks
    S = 2                           # nearest-neighbour terms act on 2 qubits
    L = 2 * blocks - 2              # theorem's L (rotation layers before the final RX, RY)
    sigma2 = 1.0 / (4 * S * (L + 2))
    n_params = 2 * blocks * n

    try:
        dev = qml.device("lightning.qubit", wires=n)
        diff_method = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n)
        diff_method = "backprop"

    circuit, H = build_circuit(n, blocks, dev, diff_method, args.angle_scale)
    grad_fn = qml.grad(circuit)

    # Exact ground energy for reference (sparse diagonalization; fine up to ~16 qubits)
    t0 = time.perf_counter()
    Hs = H.sparse_matrix() if hasattr(H, "sparse_matrix") else qml.matrix(H, wire_order=range(n))
    e0 = float(eigsh(Hs, k=1, which="SA")[0][0])
    t_eig = time.perf_counter() - t0

    print(f"Qubits N = {n}, blocks = {blocks}, parameters = {n_params}, theorem L = {L}, S = {S}")
    print(f"Gaussian variance 1/(4S(L+2)) = 1/{4*S*(L+2)} = {sigma2:.5f}  (sigma = {onp.sqrt(sigma2):.4f} rad)")
    print(f"Uniform [0, 2pi] variance = {(2*onp.pi)**2/12:.3f}  ->  {((2*onp.pi)**2/12)/sigma2:.0f}x larger")
    print(f"Device: {dev.name}, diff_method = {diff_method}, angle scale = {args.angle_scale:g} "
          f"({'paper convention exp(-i theta G)' if args.angle_scale == 2 else 'PennyLane angles'})")
    print(f"Exact ground energy E0 = {e0:.5f}  (computed in {t_eig:.2f} s)")
    print(f"Parameter-shift cost per gradient on hardware = 2 x {n_params} = {2*n_params} circuit runs\n")

    rng = onp.random.default_rng(args.seed)
    rows = []
    for kind in ("gaussian", "uniform", "zero"):
        # 1) squared gradient norm at initialization
        sq_norms, g_times = [], []
        draws = 1 if kind == "zero" else args.draws
        for _ in range(draws):
            p = pnp.array(init_params(kind, blocks, n, sigma2, rng), requires_grad=True)
            t0 = time.perf_counter()
            g = grad_fn(p)
            g_times.append(time.perf_counter() - t0)
            sq_norms.append(float(onp.sum(onp.asarray(g) ** 2)))

        # 2) a few Adam steps (lr 0.01 as in the paper)
        p = pnp.array(init_params(kind, blocks, n, sigma2, rng), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.01)
        e_start = float(circuit(p))
        t0 = time.perf_counter()
        for _ in range(args.steps):
            p = opt.step(circuit, p)
        t_steps = (time.perf_counter() - t0) / max(args.steps, 1)
        e_end = float(circuit(p))

        row = {
            "init": kind,
            "angle_scale": args.angle_scale,
            "qubits": n,
            "params": n_params,
            "mean_sq_grad_norm_at_init": onp.mean(sq_norms),
            "sec_per_gradient": onp.mean(g_times),
            "energy_start": e_start,
            f"energy_after_{args.steps}_adam_steps": e_end,
            "sec_per_adam_step": t_steps,
            "exact_E0": e0,
        }
        rows.append(row)

    hdr = f"{'init':<9}{'E||grad||^2':>14}{'s/grad':>9}{'E start':>10}{'E end':>10}{'s/step':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        e_end = r[f"energy_after_{args.steps}_adam_steps"]
        print(f"{r['init']:<9}{r['mean_sq_grad_norm_at_init']:>14.4e}{r['sec_per_gradient']:>9.3f}"
              f"{r['energy_start']:>10.4f}{e_end:>10.4f}{r['sec_per_adam_step']:>9.3f}")

    out = f"smoke_results_{n}q_scale{args.angle_scale:g}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")
    print("Note: zero init gives an exactly zero gradient for this H and ansatz, so noiseless training cannot move;")
    print("the paper's zero-init runs only move because of gradient noise.")


if __name__ == "__main__":
    main()
