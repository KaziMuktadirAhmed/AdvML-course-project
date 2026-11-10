"""S2: quantum neural network classifier (plan Sec. 5, decision of Nov 5, 20:26). Refactored from qml_feasibility.py.

Task : sklearn 8x8 digits, classes 3 vs 8 (labels +1 / -1), stratified 70/30 split (random_state 0),
       PCA to N components fitted on the training set, each feature min-max scaled to [0, pi] using training ranges.
Model: angle encoding RY(x_q) on qubit q (fixed, raw PennyLane angle), then the paper's ansatz
       (blocks x (CZ ring -> RX -> RY), paper angle convention), output <Z_0>.
Loss : mean squared error to the +-1 labels; prediction = sign(<Z_0>).
Init : the observable is Z on ONE qubit, so S = 1 and gamma^2 = 1/(4 (L+2)) = 1/80 at 10 blocks (L = 18).
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from ansatz import apply_blocks, make_device

CLASSES = (3, 8)
S = 1


def load_data(n_qubits, seed=0):
    X, y = load_digits(return_X_y=True)
    m = np.isin(y, CLASSES)
    X, y = X[m], np.where(y[m] == CLASSES[0], 1.0, -1.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    lo, hi = Ztr.min(0), Ztr.max(0)

    def scale(Z):
        return np.clip((Z - lo) / (hi - lo), 0, 1) * np.pi

    return scale(Ztr), ytr, scale(Zte), yte


def make_model(n, blocks, angle_scale=2.0, dev=None):
    """Returns circuit(params, X) -> <Z_0> for every row of X (parameter broadcasting over the batch)."""
    dev = dev if dev is not None else make_device(n)
    diff = "adjoint" if dev.name.startswith("lightning") else "backprop"

    @qml.qnode(dev, diff_method=diff)
    def circuit(params, X):
        for q in range(n):
            qml.RY(X[..., q], wires=q)          # data encoding, not trainable
        apply_blocks(params, n, blocks, angle_scale)
        return qml.expval(qml.PauliZ(0))

    return circuit


def mse(circuit, params, X, y):
    return pnp.mean((circuit(params, X) - y) ** 2)


def predict(circuit, params, X):
    return np.asarray(circuit(np.asarray(params), X), dtype=float)


def metrics(circuit, params, X, y):
    out = predict(circuit, params, X)
    return float(np.mean((out - y) ** 2)), float(np.mean(np.sign(out) == y))
