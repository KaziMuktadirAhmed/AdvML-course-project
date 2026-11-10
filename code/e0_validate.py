"""E0: validation checks before any real experiment (plan Sec. 5, E0).

1. Supplement Lemma B.1 single-gate check: paper R_Y(theta) on |0>, theta ~ N(0, 1/160), E[<Z>^2] = (1 + e^{-8 g2})/2.
   The wrong (PennyLane-literal) convention would give (1 + e^{-2 g2})/2, so this also catches the angle bug.
2. Exact Heisenberg ground energies (N = 10, 15) against the plan's values.
3. Parameter count (300 at N = 15, 10 blocks) and the zero-init state: energy 14 = N - 1, gradient exactly 0.
4. Gradient correctness on 4 qubits / 2 blocks: adjoint vs central finite differences vs the paper's
   parameter-shift rule Eq. (2), df/dtheta = f(theta + pi/4) - f(theta - pi/4)  (paper units).
5. Regression check against the Nov 5 smoke test (10 qubits, seed 0, 10 Gaussian draws: E||grad||^2 = 385.6).
Writes results/e0_validation.txt.
"""
import os
import sys
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz import expval_qnode, param_shape, n_params, theorem_L
from hamiltonians import heisenberg_open_chain, ground_energy
from inits import gamma2, init_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e0_validation.txt")
lines, ok_all = [], True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    s = f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"
    print(s, flush=True)
    lines.append(s)


# ---- 1. Lemma B.1 -------------------------------------------------------------------------------------------
g2 = 1 / 160
rng = np.random.default_rng(0)
theta = rng.normal(0, np.sqrt(g2), 100_000)
dev1 = qml.device("default.qubit", wires=1)


@qml.qnode(dev1)
def one_gate(t, scale):
    qml.RY(scale * t, wires=0)
    return qml.expval(qml.PauliZ(0))


z = np.asarray(one_gate(theta, 2.0))           # parameter broadcasting: one call, 100k angles
est, se = np.mean(z ** 2), np.std(z ** 2) / np.sqrt(len(z))
exact = (1 + np.exp(-8 * g2)) / 2
wrong = (1 + np.exp(-2 * g2)) / 2
report("Lemma B.1, paper convention", abs(est - exact) < 4 * se,
       f"E[<Z>^2] = {est:.5f} +- {se:.5f}; exact {exact:.5f}; lemma lower bound 1 - 4g2 = {1 - 4 * g2:.4f}")
z1 = np.asarray(one_gate(theta, 1.0))
report("Lemma B.1, wrong convention is distinguishable", abs(np.mean(z1 ** 2) - wrong) < 4 * se
       and abs(np.mean(z1 ** 2) - exact) > 20 * se,
       f"scale 1 gives {np.mean(z1 ** 2):.5f} (predicted {wrong:.5f}), far from {exact:.5f}")

# ---- 2. Ground energies ---------------------------------------------------------------------------------------
for n, ref in ((10, -17.032), (15, -25.668)):
    e0 = ground_energy(heisenberg_open_chain(n), n)
    report(f"Ground energy N = {n}", abs(e0 - ref) < 1e-3, f"E0 = {e0:.5f} (plan: {ref})")

# ---- 3. Parameter count and zero init at N = 15 --------------------------------------------------------------
n, blocks = 15, 10
H = heisenberg_open_chain(n)
f = expval_qnode(n, blocks, H)
p0 = pnp.array(np.zeros(param_shape(n, blocks)), requires_grad=True)
t0 = time.perf_counter()
g0 = np.asarray(qml.grad(f)(p0))
tg = time.perf_counter() - t0
report("Parameter count", n_params(n, blocks) == 300 == g0.size and theorem_L(blocks) == 18,
       f"{g0.size} parameters, L = {theorem_L(blocks)}")
report("Zero init energy", abs(float(f(p0)) - (n - 1)) < 1e-10, f"<H> = {float(f(p0)):.10f} (expect {n - 1})")
report("Zero init gradient", np.max(np.abs(g0)) < 1e-10,
       f"max |grad| = {np.max(np.abs(g0)):.2e} (time per gradient at 15 qubits: {tg:.3f} s)")

# ---- 4. Gradient correctness, 4 qubits ----------------------------------------------------------------------
n, blocks = 4, 2
f4 = expval_qnode(n, blocks, heisenberg_open_chain(n))
fb = expval_qnode(n, blocks, heisenberg_open_chain(n), dev=qml.device("default.qubit", wires=n))
p = init_params("uniform", param_shape(n, blocks), np.random.default_rng(1))
g_adj = np.asarray(qml.grad(f4)(pnp.array(p, requires_grad=True)))
g_fd, g_ps = np.zeros_like(p), np.zeros_like(p)
h = 1e-6
for idx in np.ndindex(p.shape):
    e = np.zeros_like(p); e[idx] = 1.0
    g_fd[idx] = (float(f4(p + h * e)) - float(f4(p - h * e))) / (2 * h)
    g_ps[idx] = float(f4(p + np.pi / 4 * e)) - float(f4(p - np.pi / 4 * e))
report("Adjoint vs finite differences", np.max(np.abs(g_adj - g_fd)) < 1e-6,
       f"max diff {np.max(np.abs(g_adj - g_fd)):.2e} over {p.size} params")
report("Adjoint vs paper Eq. (2) shift rule", np.max(np.abs(g_adj - g_ps)) < 1e-10,
       f"max diff {np.max(np.abs(g_adj - g_ps)):.2e}")
report("lightning.qubit vs default.qubit energy", abs(float(f4(p)) - float(fb(p))) < 1e-10,
       f"{float(f4(p)):.12f} vs {float(fb(p)):.12f}")

# ---- 5. Regression vs smoke test ------------------------------------------------------------------------------
n, blocks = 10, 10
f10 = expval_qnode(n, blocks, heisenberg_open_chain(n))
gr = qml.grad(f10)
rng = np.random.default_rng(0)
vals = [float(np.sum(np.asarray(gr(pnp.array(init_params("gaussian", param_shape(n, blocks), rng,
                                                              var=gamma2(2, theorem_L(blocks))),
                                                  requires_grad=True))) ** 2)) for _ in range(10)]
report("Regression vs smoke test (10q Gaussian)", abs(np.mean(vals) - 385.6) < 0.1,
       f"E||grad||^2 = {np.mean(vals):.2f} (smoke test, Nov 5: 385.6)")

lines.append(f"\nOverall: {'ALL PASS' if ok_all else 'SOME CHECKS FAILED'}  (PennyLane {qml.__version__}, "
             f"Python {sys.version.split()[0]}, {time.strftime('%Y-%m-%d %H:%M')})")
print(lines[-1])
with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
