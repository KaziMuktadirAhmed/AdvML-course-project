"""
Depth sweep under realistic device noise (Experiment E4b, preliminary version).

For N qubits on an IBM fake backend, and blocks = 1, 2, 3, 4, 6, 8:
  - squared gradient norm at initialization, noise OFF vs ON, for Gaussian and uniform init
    (mean over --draws random draws; same parameter draws used for OFF and ON)
  - zero-init energy, noise OFF vs ON (how much the device noise shrinks the signal)
Exact density-matrix expectation values (no shot noise); parameter-shift gradients batched in one call.
Writes noise_depth_sweep_<backend>_<N>q_scale<k>.csv.
Angles: default --angle-scale 2 = paper convention RX(theta) = exp(-i theta X); 1 = raw Qiskit angles.

Usage:
  python noise_depth_sweep.py                                  # 5 qubits, FakeManilaV2
  python noise_depth_sweep.py --backend FakeNairobiV2 --qubits 7 --blocks 1 2 3 4

Requires: qiskit, qiskit-aer, qiskit-ibm-runtime, and qiskit_noise_test.py in the same folder.
"""
import argparse
import csv
import time

import numpy as np
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime import fake_provider

from qiskit_noise_test import ansatz, heisenberg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="FakeManilaV2")
    ap.add_argument("--qubits", type=int, default=5)
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angle-scale", type=float, default=2.0,
                    help="2 = paper convention exp(-i theta G) (default); 1 = raw Qiskit angles")
    args = ap.parse_args()
    c = args.angle_scale
    shift, factor = np.pi / (2 * c), c / 2  # parameter-shift; c = 2 gives the paper's Eq. (2)

    n = args.qubits
    fake = getattr(fake_provider, args.backend)()
    nm = NoiseModel.from_backend(fake)
    ests = {
        "off": EstimatorV2(options={"backend_options": {"method": "density_matrix"}, "default_precision": 0.0}),
        "on": EstimatorV2(options={"backend_options": {"method": "density_matrix", "noise_model": nm},
                                   "default_precision": 0.0}),
    }
    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"{args.backend}, N = {n}, {args.draws} draws per point, angle scale = {c:g}")
    print("blocks 2q-gates | grad^2: gauss_off gauss_on unif_off unif_on | G/U off  G/U on | zero-init E off -> on")
    t0 = time.time()
    for blocks in args.blocks:
        qc, _ = ansatz(n, blocks, c)
        P = 2 * blocks * n
        isa = transpile(qc, backend=fake, optimization_level=1, seed_transpiler=args.seed)
        obs = heisenberg(n).apply_layout(isa.layout)
        two_q = sum(v for k, v in isa.count_ops().items() if k in ("cx", "ecr", "cz"))
        L = 2 * blocks - 2
        s2 = 1.0 / (4 * 2 * (L + 2))

        def grad_sq_and_energy(p, mode):
            shifts = np.vstack([p, p + shift * np.eye(P), p - shift * np.eye(P)])
            ev = np.asarray(ests[mode].run([(isa, obs, shifts)]).result()[0].data.evs)
            g = factor * (ev[1:P + 1] - ev[P + 1:])
            return float(g @ g), float(ev[0])

        res = {}
        for kind in ("gauss", "unif"):
            draws = [rng.normal(0, np.sqrt(s2), P) if kind == "gauss" else rng.uniform(0, 2 * np.pi, P)
                     for _ in range(args.draws)]
            for mode in ("off", "on"):
                res[f"{kind}_{mode}"] = float(np.mean([grad_sq_and_energy(p, mode)[0] for p in draws]))
        z_off = grad_sq_and_energy(np.zeros(P), "off")[1]
        z_on = grad_sq_and_energy(np.zeros(P), "on")[1]
        row = {"backend": args.backend, "angle_scale": c, "qubits": n, "blocks": blocks, "params": P, "two_qubit_gates": two_q,
               **res, "zero_E_off": z_off, "zero_E_on": z_on}
        rows.append(row)
        print(f"{blocks:>6} {two_q:>8} | {res['gauss_off']:>9.3f} {res['gauss_on']:>8.3f} {res['unif_off']:>8.3f} "
              f"{res['unif_on']:>7.3f} | {res['gauss_off']/res['unif_off']:>7.2f} {res['gauss_on']/res['unif_on']:>7.2f} | "
              f"{z_off:.3f} -> {z_on:.3f} ({100*(z_on/z_off-1):+.0f}%)  [{time.time()-t0:.0f}s]", flush=True)

    out = f"noise_depth_sweep_{args.backend}_{n}q_scale{c:g}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
