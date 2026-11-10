"""Hardware-efficient ansatz of Zhang et al., NeurIPS 2022 (Sec. 5, Fig. 1).

One block = CZ ring (0,1),(1,2),...,(n-1,0) -> RX on every qubit -> RY on every qubit.
Parameters have shape (blocks, 2, n): [:, 0, :] are RX angles, [:, 1, :] are RY angles.
Rotation layers = 2*blocks = L + 2, so the theorem's L = 2*blocks - 2 (10 blocks -> L = 18, 300 params at n = 15).

ANGLE CONVENTION (plan Sec. 8, item 8): the paper defines R_X(theta) = exp(-i theta X); PennyLane/Qiskit use
exp(-i t X / 2). With angle_scale = 2 (default) parameters are in the PAPER's units, so variances, learning
rates, gradient norms and gradient-noise variances all match the paper. angle_scale = 1 = raw PennyLane angles.
"""
import pennylane as qml


def theorem_L(blocks):
    return 2 * blocks - 2


def n_params(n, blocks):
    return 2 * blocks * n


def param_shape(n, blocks):
    return (blocks, 2, n)


def apply_blocks(params, n, blocks, angle_scale=2.0):
    for b in range(blocks):
        for q in range(n):
            qml.CZ(wires=[q, (q + 1) % n])
        for q in range(n):
            qml.RX(angle_scale * params[b, 0, q], wires=q)
        for q in range(n):
            qml.RY(angle_scale * params[b, 1, q], wires=q)


def make_device(n, prefer="lightning.qubit", shots=None):
    kw = {} if shots is None else {"shots": shots}
    try:
        return qml.device(prefer, wires=n, **kw)
    except Exception:
        return qml.device("default.qubit", wires=n, **kw)


def expval_qnode(n, blocks, observable, angle_scale=2.0, dev=None, diff_method=None):
    """QNode f(params) = <0|V(params)^dag O V(params)|0>, paper Eq. (1) with rho_in = |0...0>."""
    dev = dev if dev is not None else make_device(n)
    if diff_method is None:
        diff_method = "adjoint" if dev.name.startswith("lightning") else "backprop"

    @qml.qnode(dev, diff_method=diff_method)
    def f(params):
        apply_blocks(params, n, blocks, angle_scale)
        return qml.expval(observable)

    return f
