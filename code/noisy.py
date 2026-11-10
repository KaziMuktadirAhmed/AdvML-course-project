"""E4b / S3: noisy hardware simulation with Qiskit Aer and IBM fake-backend noise models.
Ported from Project/qiskit_noise_test.py and noise_depth_sweep.py (Nov 5).

The ansatz is built ONCE as a parameterized circuit (same gate order and parameter order as ansatz.py:
flat index = block*2n + (0 for RX, 1 for RY)*n + qubit, i.e. params.reshape(-1) of shape (blocks, 2, n)),
transpiled ONCE to the fake backend (real layout, native gates, SWAPs for the CZ ring), and evaluated with the
Aer EstimatorV2 using the density-matrix method -> exact expectation values with device noise, no shot noise.
Gradients: parameter-shift with all 2P shifted circuits in one batched call. In paper units (angle_scale 2) the
rule is the paper's Eq. (2): df/dtheta = f(theta + pi/4) - f(theta - pi/4).
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime import fake_provider


def heisenberg_op(n):
    return SparsePauliOp.from_sparse_list([(P + P, [i, i + 1], 1.0) for i in range(n - 1) for P in "XYZ"],
                                          num_qubits=n)


def ansatz_circuit(n, blocks, angle_scale=2.0):
    theta = ParameterVector("t", 2 * blocks * n)
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(blocks):
        for q in range(n):
            qc.cz(q, (q + 1) % n)
        for q in range(n):
            qc.rx(angle_scale * theta[k], q); k += 1
        for q in range(n):
            qc.ry(angle_scale * theta[k], q); k += 1
    return qc


class NoisyVQE:
    """f(params) and its parameter-shift gradient on a transpiled circuit, with device noise on or off."""

    def __init__(self, backend, n, blocks, observable=None, angle_scale=2.0, seed_transpiler=0):
        self.fake = getattr(fake_provider, backend)()
        self.n, self.blocks, self.P = n, blocks, 2 * blocks * n
        self.isa = transpile(ansatz_circuit(n, blocks, angle_scale), backend=self.fake, optimization_level=1,
                             seed_transpiler=seed_transpiler)
        self.obs = (observable if observable is not None else heisenberg_op(n)).apply_layout(self.isa.layout)
        ops = self.isa.count_ops()
        self.two_qubit_gates = int(sum(v for k, v in ops.items() if k in ("cx", "ecr", "cz", "swap")))
        self.depth = self.isa.depth()
        nm = NoiseModel.from_backend(self.fake)
        self.est = {
            # noise off: a pure state, so the exact statevector method gives the same values much faster
            "off": EstimatorV2(options={"backend_options": {"method": "statevector"}, "default_precision": 0.0}),
            "on": EstimatorV2(options={"backend_options": {"method": "density_matrix", "noise_model": nm},
                                       "default_precision": 0.0}),
        }
        self.shift, self.factor = np.pi / (2 * angle_scale), angle_scale / 2
        self.circuit_evals = 0

    def values(self, rows, noise):
        rows = np.atleast_2d(rows)
        self.circuit_evals += len(rows)
        return np.asarray(self.est[noise].run([(self.isa, self.obs, rows)]).result()[0].data.evs, dtype=float)

    def value_and_grad(self, p, noise):
        p = np.asarray(p, dtype=float).reshape(-1)
        E = np.eye(self.P)
        ev = self.values(np.vstack([p, p + self.shift * E, p - self.shift * E]), noise)
        return float(ev[0]), self.factor * (ev[1:self.P + 1] - ev[self.P + 1:])
