"""Observables for the reproduction of Zhang et al., NeurIPS 2022.

heisenberg_open_chain : paper Eq. (10), H = sum_{i=1}^{N-1} X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}
zz                    : a single 2-local Pauli string Z_i Z_j (used to test Theorem 4.1 directly)
ground_energy         : exact smallest eigenvalue by sparse diagonalization (fine up to ~16 qubits)
"""
import pennylane as qml
from scipy.sparse.linalg import eigsh


def heisenberg_open_chain(n):
    coeffs, ops = [], []
    for i in range(n - 1):
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            coeffs.append(1.0)
            ops.append(P(i) @ P(i + 1))
    return qml.Hamiltonian(coeffs, ops)


def zz(i=0, j=1):
    return qml.PauliZ(i) @ qml.PauliZ(j)


def ground_energy(H, n):
    try:
        M = H.sparse_matrix(wire_order=range(n))
    except Exception:
        M = qml.matrix(H, wire_order=range(n))
    return float(eigsh(M, k=1, which="SA")[0][0])
