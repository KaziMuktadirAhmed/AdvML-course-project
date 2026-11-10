"""
Qiskit-native noisy simulation for the Gaussian-initialization reproduction.

- Builds the paper's ansatz (blocks of CZ ring -> RX -> RY) as ONE parameterized circuit.
- Transpiles it ONCE to an IBM fake backend (real device layout, native gates, SWAPs),
  and uses that backend's noise model (gate errors, T1/T2, readout) in Aer.
- Computes parameter-shift gradients by sending all 2P shifted parameter sets to the Aer
  EstimatorV2 in a single batched call (density-matrix method -> exact noisy expectation values).

Usage:
  python qiskit_noise_test.py                                   # FakeManilaV2, 5 qubits, 2 blocks
  python qiskit_noise_test.py --backend FakeNairobiV2 --qubits 7 --blocks 3
  python qiskit_noise_test.py --noiseless                       # same circuit, no noise (control)
  python qiskit_noise_test.py --steps 100                       # also run 100 GD steps (lr 0.01)

Requires: pip install qiskit qiskit-aer qiskit-ibm-runtime
"""
import argparse
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime import fake_provider


def heisenberg(n):
    terms = []
    for i in range(n - 1):
        for P in "XYZ":
            terms.append((P + P, [i, i + 1], 1.0))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def ansatz(n, blocks, angle_scale=2.0):
    """Paper ansatz. angle_scale = 2: parameters in the paper's convention RX(theta) = exp(-i theta X)
    (Qiskit's rx(t) = exp(-i t X / 2), so the gate gets 2*theta). angle_scale = 1: raw Qiskit angles."""
    theta = ParameterVector("t", 2 * blocks * n)  # order: block, (RX then RY), qubit
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(blocks):
        for q in range(n):
            qc.cz(q, (q + 1) % n)
        for q in range(n):
            qc.rx(angle_scale * theta[k], q); k += 1
        for q in range(n):
            qc.ry(angle_scale * theta[k], q); k += 1
    return qc, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="FakeManilaV2")
    ap.add_argument("--qubits", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--noiseless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angle-scale", type=float, default=2.0,
                    help="2 = paper convention exp(-i theta G) (default); 1 = raw Qiskit angles")
    args = ap.parse_args()

    n, blocks = args.qubits, args.blocks
    S, L = 2, 2 * blocks - 2
    sigma2 = 1.0 / (4 * S * (L + 2))
    P = 2 * blocks * n

    fake = getattr(fake_provider, args.backend)()
    qc, theta = ansatz(n, blocks, args.angle_scale)
    isa = transpile(qc, backend=fake, optimization_level=1, seed_transpiler=args.seed)
    obs = heisenberg(n).apply_layout(isa.layout)
    ops = isa.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "swap"))

    sim_opts = {"method": "density_matrix"}
    if not args.noiseless:
        sim_opts["noise_model"] = NoiseModel.from_backend(fake)
    est = EstimatorV2(options={"backend_options": sim_opts, "default_precision": 0.0})

    def energies(param_rows):
        return np.asarray(est.run([(isa, obs, param_rows)]).result()[0].data.evs)

    # Parameter-shift for a gate exp(-i c theta G/2) with c = angle_scale:
    # df/dtheta = (c/2) * [f(theta + pi/(2c)) - f(theta - pi/(2c))]; for c = 2 this is the paper's Eq. (2):
    # df/dtheta = f(theta + pi/4) - f(theta - pi/4).
    c = args.angle_scale
    shift, factor = np.pi / (2 * c), c / 2

    def value_and_grad(p):
        shifts = np.vstack([p, p + shift * np.eye(P), p - shift * np.eye(P)])
        ev = energies(shifts)
        return ev[0], factor * (ev[1:P + 1] - ev[P + 1:])

    print(f"Backend {args.backend} ({fake.num_qubits} q) | N = {n}, blocks = {blocks}, P = {P}, "
          f"gamma^2 = 1/{4*S*(L+2)} | noise = {'OFF' if args.noiseless else 'ON'} | angle scale = {args.angle_scale:g}")
    print(f"Transpiled: depth {isa.depth()}, two-qubit gates {two_q}, ops {dict(ops)}")
    rng = np.random.default_rng(args.seed)
    print(f"{'init':<9}{'E||grad||^2':>14}{'E at init':>11}{'s/grad':>9}"
          + (f"{'E after GD':>12}" if args.steps else ""))
    for kind in ("gaussian", "uniform", "zero"):
        norms, e0s, ts = [], [], []
        for _ in range(1 if kind == "zero" else args.draws):
            if kind == "gaussian":
                p = rng.normal(0, np.sqrt(sigma2), P)
            elif kind == "uniform":
                p = rng.uniform(0, 2 * np.pi, P)
            else:
                p = np.zeros(P)
            t0 = time.perf_counter()
            e, g = value_and_grad(p)
            ts.append(time.perf_counter() - t0)
            norms.append(float(g @ g)); e0s.append(float(e))
        line = f"{kind:<9}{np.mean(norms):>14.4e}{np.mean(e0s):>11.4f}{np.mean(ts):>9.3f}"
        if args.steps:
            for _ in range(args.steps):
                e, g = value_and_grad(p)
                p = p - 0.01 * g
            line += f"{energies(p[None, :])[0]:>12.4f}"
        print(line)
    print(f"\nParameter-shift: {2*P} circuit evaluations per gradient (batched into one Aer call).")


if __name__ == "__main__":
    main()
