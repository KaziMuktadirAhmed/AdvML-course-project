import numpy as onp, pennylane as qml
from pennylane import numpy as pnp
import smoke_test as st
n,blocks=15,10; P=(blocks,2,n); s2=1/160
dev=qml.device("lightning.qubit",wires=n)
for scale in (1.0,2.0):
    c,_=st.build_circuit(n,blocks,dev,"adjoint",scale); g=qml.grad(c)
    rng=onp.random.default_rng(0)
    for kind in ("gaussian","uniform"):
        f0,gn=[],[]
        for _ in range(5):
            p=pnp.array(rng.normal(0,onp.sqrt(s2),P) if kind=="gaussian" else rng.uniform(0,2*onp.pi,P),requires_grad=True)
            f0.append(float(c(p))); gn.append(float(onp.linalg.norm(onp.asarray(g(p)))))
        # one GD run, lr 0.01, 100 steps, from the last draw
        opt=qml.GradientDescentOptimizer(0.01); traj=[]
        for t in range(101):
            if t in (10,30,100): traj.append((t,float(c(p))))
            p=opt.step(c,p)
        print(f"scale {scale:g} {kind:9s} init f = {onp.mean(f0):6.2f}  init |grad| = {onp.mean(gn):6.2f}   GD f(10,30,100) = {[round(v,2) for _,v in traj]}",flush=True)
