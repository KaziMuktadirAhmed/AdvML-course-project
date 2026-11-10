# ENGI 9882 Course Project Plan
## Reproducing "Escaping from the Barren Plateau via Gaussian Initializations in Deep Variational Quantum Circuits"

*Kazi · Prepared Nov 5, 2026*

**Paper:** K. Zhang, L. Liu, M.-H. Hsieh, D. Tao, "Escaping from the Barren Plateau via Gaussian Initializations in Deep Variational Quantum Circuits," *NeurIPS 2022*.
**Download:** [arXiv DOI 10.48550/arXiv.2203.09376](https://doi.org/10.48550/arXiv.2203.09376) · [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2022/file/7611a3cb5d733e628081431445cb01fd-Paper-Conference.pdf) · [NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7611a3cb5d733e628081431445cb01fd-Abstract.html) (check its **Supplemental** link for the authors' code)
**Supplement:** `3600270.3601622_supp.pdf` in this folder (ACM DL). It has extra experiments and proofs, no code; §8 lists what it resolves.
**Project type:** Reproduction (template 9 in `project guidance.pdf`). Negative results are acceptable.
*Updated Nov 5 (18:30): angle-convention fix (paper RX(θ) = e^(−iθX) vs PennyLane/Qiskit e^(−iθX/2)); scripts take `--angle-scale` (default 2 = paper units); §3 and E4b numbers re-run in paper units; ambiguity 6 resolved, ambiguity 8 added.*
*Updated Nov 5 (02:10) after reading the supplement: circuit gate order corrected, iteration count = 100, variance-scaling meaning, N = 18 experiment added (E3b), compute estimates revised.*

---

## 1. Deadlines

| Date | Deliverable | Weight | Status |
|---|---|---|---|
| **Thu Nov 5, 23:59** | Proposal (IEEE template, 1–2 pages) | 5% | ☐ |
| Thu Nov 19 | Assignment 3 (not the project, but it takes that week) | — | ☐ |
| **Mon Nov 23** | Slides (hard deadline) → **results needed by ~Nov 20** | part of 10% | ☐ |
| ~Nov 24 – Dec 3 | 15-min presentation + 3-min Q&A; feedback forms | 10% + 5% | ☐ |
| **Mon Dec 7** | Final report (IEEE template, 6 pages) | 15% | ☐ |

⚠ **Template conflict to resolve:** the guidance says IEEE Transactions; the course outline says ICML 2025 style. Ask on Brightspace.

---

## 2. What the paper claims (in numbers)

### The problem
In a deep variational circuit with random starting angles (uniform on [0, 2π]), gradients shrink **exponentially with the number of qubits**. This is the *barren plateau*: training can't find a direction to move in.

### The fix
Start every angle from a **narrow Gaussian**, θ ~ N(0, γ²), with the variance tied to the circuit depth L:

> γ² = 1 / (4·S·(L+2))
>
> - **S** = number of qubits each Hamiltonian term acts on. For nearest-neighbour terms like X₁X₂, S = 2.
> - **L** = number of rotation layers before the final R_X, R_Y pair.

**Worked example, the paper's main setting:** S = 2, L = 18 → γ² = 1/(4·2·20) = **1/160 = 0.00625**, so γ ≈ 0.079 rad ≈ 4.5°.
Compare uniform [0, 2π]: variance = (2π)²/12 ≈ **3.29**, which is **526× larger**. With Gaussian init, about 95% of angles start within ±0.16 rad (±9°), so the circuit starts close to the identity.

### Theorem 4.1 (local observables)
For an observable that is a product of Paulis acting on S qubits:

> E‖∇f‖² ≥ L / (S^S · (L+2)^(S+1)) · Tr[σ_j ρ_in]²

The bound decays only **polynomially** in L, not exponentially.

**Worked example:** take one ZZ term with input |0…0⟩, so Tr[·]² = 1. With S = 2 and L = 18:
bound = 18 / (2² · 20³) = 18 / 32,000 ≈ **5.6 × 10⁻⁴**. Your measured E‖∇f‖² should sit **above** this.

### Theorem 4.2 and Corollary 4.3
- **Theorem 4.2** gives a similar guarantee for global observables. It only helps when the gradient at θ = 0 is already large, which the authors themselves flag.
- **Corollary 4.3** says O(L/ε²) measurements are enough to estimate a useful gradient under shot noise.

### The paper's experiments

| | Heisenberg model (main) | LiH molecule |
|---|---|---|
| Hamiltonian | H = Σᵢ₌₁^{N−1} (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁), open chain | 2 electrons, 10 spin orbitals |
| Qubits | 15 | 10 |
| Ansatz | 10 blocks of (CZ ring → R_X layer → R_Y layer), 300 parameters (p. 7; Fig. 1) | 24 Givens rotations (8 single + 16 double excitations) |
| Iterations | **100** (every supplement figure runs 0–100) | 100 |
| Input state | \|0⟩^⊗15 | Hartree–Fock \|11 0000 0000⟩ |
| Inits compared | Gaussian 1/160 vs uniform [0, 2π] vs zero | Gaussian 1/147,456 vs uniform vs zero |
| Optimizers | GD, Adam; learning rate 0.01 | GD (learning rate 0.1), Adam (learning rate 0.01) |
| Gradient noise | none, or Gaussian with variance 0.01 | none, constant 0.001, or "adaptive" (Eq. 12) |
| Seeds | 5 | 5 |
| Supplement (Sec. A) | L ∈ {14, 18, 22} × variances {0.01γ, 0.1γ, γ, 10γ, 100γ}, GD, exact + noisy (5 runs); momentum/NAG/AdaGrad at N = 15 (5 runs) and **N = 18, L = 38** (3 runs) | depths 24, 48, 72; GD (learning rate 0.1) + Adam (0.01), adaptive noise; **Gaussian vs zero only**, loss shown as f − f* (3 runs) |

**What the paper does *not* show:** a plot of gradient norm vs **number of qubits**. That is the barren-plateau claim itself, so it's a natural **extension** for you (Experiment E1 below).

---

## 3. What we already know from the smoke test

> ⚠ **Angle convention (found Nov 5, 18:20).** The paper (Sec. 2) defines R_X(θ) = e^(−iθX), but PennyLane and Qiskit define RX(t) = e^(−itX/2). So **paper angle θ = PennyLane angle 2θ**, and the variance, learning rates, gradient norms and gradient-noise variance are all in the paper's units. Since Nov 5 the scripts take `--angle-scale 2` (the default, paper units). `--angle-scale 1` reproduces the earlier runs, which effectively used a Gaussian variance of 1/640 in paper units, 4× narrower than the theorem. Both are kept: scale 2 is the faithful reproduction, scale 1 is a sensitivity check. See §8, item 8.

**Paper-convention results (scale 2; seeded, so identical on any machine; times from my 2-core machine):**

| Qubits | Params | E‖∇E‖² at init: **Gaussian** | E‖∇E‖² at init: **uniform** | Ratio | Energy after Adam (lr 0.01): Gaussian / uniform | Time per gradient |
|---|---|---|---|---|---|---|
| 10 | 200 | 385.6 | 10.34 | **37×** | 6.27 → −6.63 / −0.07 → −3.10 (10 steps; E₀ = −17.03) | 0.024 s |
| 15 | 300 | 740.0 | 8.51 | **87×** | 9.37 → −17.98 / −0.20 → −10.33 (20 steps; E₀ = −25.67) | 0.073 s |

At 15 qubits, Gaussian covers (9.37 + 17.98)/(9.37 + 25.67) = **78%** of the way to E₀ in 20 steps, vs (10.33 − 0.20)/(25.67 − 0.20) = **40%** for uniform.

**Earlier PennyLane-angle results (scale 1), kept for the sensitivity comparison:** 10 qubits: 46.44 vs 2.56 (18×); 15 qubits: 92.11 vs 2.66 (35×).

What these numbers show:
- **Per parameter**, the uniform-init gradient falls as qubits grow: 10.34/200 ≈ 0.052 → 8.51/300 ≈ 0.028. That's the barren plateau starting to show. The Gaussian init doesn't fall: 1.93 → 2.47.
- **Zero init** gives an **exactly zero gradient** for this Hamiltonian and ansatz, so noiseless training never moves. The paper's zero-init runs only move because of gradient noise. Explain this in the report.
- **Compute is tiny.** At 0.08 s per gradient, 500 training steps take ~40 s, and a 60-run experiment takes ~40 min.

---

## 4. Requirements → how this project meets them

| Course requirement | How it's met |
|---|---|
| Recent paper at NeurIPS/ICML/ICLR/AISTATS/JMLR | NeurIPS 2022 |
| Implement or adapt an advanced ML method, not off-the-shelf | You implement the ansatz, the three initializations, the noisy-gradient training and the measurement-cost study yourself in PennyLane |
| Baseline comparison | Uniform [0, 2π] (the standard barren-plateau setting) and zero init |
| **Statistical** performance metric | Iterations to reach a target energy; final energy error; gradient norm at init and during training |
| **Hardware** performance metric | Wall-clock time per gradient vs qubits; **circuit executions and shots to reach the target energy** under parameter-shift (2 circuits per parameter → 600 per gradient at 300 parameters); **behaviour in the noisy hardware simulation** (E4b: transpiled depth, two-qubit gate count, noisy vs noiseless gradients) |

---

## 5. Experiments

> **Decision, Nov 5 (20:26): the project is framed as an ML project.** The proposal describes three studies:
> - **S1, reproduction / validation:** E0, E1 and E2 below.
> - **S2, QNN classifier (the main ML study):** this replaces the optional LiH experiment (E5).
>   - **Task:** digits 3 vs 8 from scikit-learn, PCA to N features, angle encoding, 10 layers of CZ → RX → RY, output ⟨Z⟩ on qubit 0, MSE loss on labels ±1.
>   - **Initialization:** S = 1, so γ² = 1/80.
>   - **Runs:** gradient sweep over N = 4–14; training at N = 12 (240 parameters), 3 inits × 5 seeds, Adam with lr 0.01, 100 steps, batch 16.
>   - **Starting code:** `qml_feasibility.py`.
>   - **Preliminary result:** with uniform init the squared gradient falls 65× from 4 to 12 qubits (3.98 → 0.061); Gaussian init is 147× larger at 12 qubits. A short 8-qubit run reaches ≈ 0.90 test accuracy for every init (Gaussian test MSE 0.27 vs uniform 0.46).
>   - **Background:** barren plateaus in QML classifiers come from deep circuits and from the data encoding (Thanasilp *et al.*, *Quantum Machine Intelligence* 2023).
>   - **Closest prior work:** Rahman *et al.*, *J. Imaging* 2026 tested Gaussian init on 2-layer QNN classifiers, found no severe barren plateaus, and used no noise.
> - **S3, noisy hardware simulation:** E4b, using the classifier as well as Heisenberg if time allows.
>
> E3, E3b and E4 remain optional. **Go/no-go for the full S2 training runs: Nov 14.**

Priority: **E0 → E1 → E2 → E4** form the core project, and they're enough for the slides. E3 and E5 are optional extras.

### E0. Validation (Nov 6, ~2 h of your time)
- **Single-gate check of supplement Lemma B.1.** For one qubit in state |0⟩, apply the paper's R_Y(θ) = e^(−iθY) (in PennyLane: `qml.RY(2*theta)`) with θ ~ N(0, γ²), and measure Z, which gives ⟨Z⟩ = cos 2θ. The supplement's Eq. (3) gives E[⟨Z⟩²] = (1 + e^(−8γ²))/2 exactly. Forgetting the factor 2 gives (1 + e^(−2γ²))/2 = 0.9938 instead, so this check also catches the convention bug.
  - With γ² = 1/160: e^(−0.05) = 0.9512, so E[⟨Z⟩²] = **0.9756**. The lemma's lower bound is 1 − 4γ² = 0.975.
  - Estimate it from 100,000 samples; you should get 0.9756 ± ~0.0001. This 5-line test confirms your sampling and gate conventions before anything bigger.
- Check the Hamiltonian's exact ground energy against sparse diagonalization (the smoke test does this): E₀ = −25.668 for N = 15.
- Check the parameter count (300) and that zero init gives a zero gradient.
- Check gradients against finite differences on a 4-qubit case.
- Write down every assumption (§8) as you make it.

### E1. Gradient norm vs qubits and depth: the core claim, plus an extension the paper lacks
- **Grid:** N ∈ {4, 6, 8, 10, 12, 14} qubits × blocks ∈ {2, 5, 10, 20} × 3 inits × 100 random draws.
- **Observable:** both the full H and a single Z₁Z₂ term. The single term lets you test Theorem 4.1's bound directly.
- **Plot:** log₁₀ E‖∇f‖² vs N, one line per init; add the theorem's bound as a dashed line.
- **Hypothesis:** for uniform init, the per-parameter gradient variance falls roughly exponentially in N, i.e., a straight line on the log plot. For Gaussian init it stays above the Theorem 4.1 bound, e.g., ≈ 5.6 × 10⁻⁴ for one ZZ term at L = 18.
- **Compute:** about 7,000 gradient evaluations at ≤ 0.1 s each, so **under 15 min**.

### E2. Heisenberg training: reproduces the paper's Figure 2
- **Setup:** N = 15, 10 blocks, 300 parameters, inits {Gaussian 1/160, uniform, zero}.
- **Grid:** optimizers {GD, Adam} at learning rate 0.01 × gradients {exact, noisy with variance 0.01} × 5 seeds = **60 runs**, **100 iterations each**, matching the supplement's figures (x-axis 0–100).
  - Optionally extend to 500 to show where each init ends up.
  - **Target to compare against** (supplement Figs. 1–2): with GD, the Gaussian-init loss falls from about +15 toward about −20 within 100 iterations (E₀ = −25.67).
- **Record per step:** energy, ‖∇‖ and wall-clock time.
- **Statistical metric:** iterations to reach within 1.0 of E₀, plus final energy error.
- **Hypothesis:** Gaussian init reaches the target in fewer iterations than uniform. Zero init moves only in the noisy runs.
- **Compute:** 60 × 100 × 0.06 s (your laptop) ≈ **6 min**; about 30 min if extended to 500 iterations.

### E3 (optional). Depth and variance sensitivity: reproduces the paper's Figures 4–5
- **Grid:** blocks ∈ {8, 10, 12}, giving L ∈ {14, 18, 22} and γ² ∈ {1/128, 1/160, 1/192}.
  - × **variance** {0.01γ², 0.1γ², γ², 10γ², 100γ²}. The supplement calls these "variances", so scale the variance, not the standard deviation.
  - × GD exact/noisy × 5 seeds = 150 runs × 100 iterations.
- **What the supplement reports:**
  - γ² converges fastest;
  - 0.01γ² and 0.1γ² behave like zero init in the noisy case;
  - 10γ² and 100γ² behave like uniform init, and slow down as depth grows.
- **Compute:** 150 × 100 × ~0.06 s ≈ **15 min**.

### E3b (optional). Bigger circuit and other optimizers: reproduces supplement Figs. 3–4
- **Setup:** N = 18 qubits with L = 38, i.e., 20 blocks and 18 × 40 = 720 parameters.
- **Grid:** {GD + momentum, NAG, AdaGrad} × {Gaussian, uniform, zero} × 3 seeds, noisy gradients, 100 iterations.
- **What the supplement reports:**
  - momentum and NAG behave like Adam, and AdaGrad like GD;
  - uniform init converges more slowly as N and L grow.
- **Compute (my estimate):** a 2¹⁸ statevector is 8× bigger than at 15 qubits, with 2.4× the parameters, so roughly 1 s per gradient → 27 runs × 100 ≈ **45 min**. Measure one gradient first.

### E4. Measurement cost: the hardware axis, an extension
- **Setup:** 8 qubits, 4 blocks (64 parameters). Use **parameter-shift** gradients with finite shots ∈ {100, 1,000, 10,000}, 3 inits × 3 seeds.
- **Metric:** total circuit executions and total shots to reach within 0.5 of E₀.
- **Worked example of why this matters:** 64 parameters × 2 circuits × 1,000 shots = 128,000 shots per step. If Gaussian init needs 100 fewer steps, that's 12.8 million fewer shots.
- **Also:** wall-clock time per exact gradient vs N (4 → 16 qubits) for the simulator.
- **Compute:** ~30–60 min.

### E4b. Noisy hardware simulation (Qiskit + IBM device noise models): ⭐ the project's main contribution

**Terminology:** throughout the project, *noisy hardware simulation* means running the circuits in Qiskit Aer with a noise model taken from an IBM "fake backend" (a calibration snapshot of a real device: gate errors, T1/T2, readout error, and the device's qubit connectivity). A run on an actual IBM QPU, if time and access allow, is called a *real hardware run*.
**Preliminary result (Nov 5, `noise_depth_sweep.py`, paper angle convention = scale 2):** 5 qubits, FakeManilaV2, mean of 8 random draws, gradient² at initialization, exact density-matrix values. Output file: `noise_depth_sweep_FakeManilaV2_5q_scale2.csv`.

| Blocks | 2-qubit gates after transpile | Gaussian, noise off | Gaussian, noise on | Uniform, noise off | Uniform, noise on | **Gauss/Unif, off** | **Gauss/Unif, on** | Zero-init energy, off → on |
|---|---|---|---|---|---|---|---|---|
| 1 | 17 | 18.10 | 15.37 | 15.02 | 12.39 | 1.20 | 1.24 | 4.000 → 3.770 (−6%) |
| 2 | 34 | 48.52 | 32.95 | 12.64 | 7.78 | 3.84 | 4.24 | 4.000 → 3.553 (−11%) |
| 3 | 57 | 59.01 | 33.17 | 17.83 | 7.42 | 3.31 | 4.47 | 4.000 → 3.357 (−16%) |
| 4 | 74 | 87.36 | 38.48 | 24.23 | 5.90 | 3.61 | 6.52 | 4.000 → 3.168 (−21%) |
| 6 | 114 | 112.11 | 35.10 | 34.47 | 2.89 | 3.25 | **12.13** | 4.000 → 2.802 (−30%) |
| 8 | 154 | 148.90 | 29.53 | 45.46 | 1.35 | 3.28 | **21.82** | 4.000 → 2.512 (−37%) |

The same pattern appeared with PennyLane-angle scaling (scale 1): the noisy ratio rose from 2.2× to 8.2×. So the finding doesn't depend on the convention.

**How to read it:**
- **Noise shrinks the signal exponentially with depth.** The zero-init energy keeps about 94% per block: 4 × 0.943⁸ ≈ 2.50, vs 2.512 measured. That's the noise-induced decay Wang *et al.* (2021) predict.
- **Uniform-init gradients collapse under noise:** they keep 12.39/15.02 = 82% at 1 block but only 1.35/45.46 = **3.0%** at 8 blocks.
- **Gaussian-init gradients hold up much better:** they keep 85% at 1 block and still **19.8%** at 8 blocks (29.53/148.90), staying around 30–38 throughout.
- **So noise *amplifies* the Gaussian advantage.** Without noise, Gaussian/uniform stays between 1.2× and 3.8× at this size. With noise it grows from 4.2× (2 blocks) to **21.8× (8 blocks)**.
- **Likely mechanism (hypothesis to test):** near-identity starts keep each term's "light cone" small, so fewer noisy gates affect it. Uniform starts spread operators across the whole circuit, so every gate's noise counts. This fits the "bounded transverse amplitude → low fragility" finding of Shen *et al.* (2026).
- **Caveats:** one backend, N = 5, 8 draws, gradients at initialization only (no training yet), no shot noise. These are my numbers from a cloud machine; **re-run on your laptop** for the proposal and report.

**Novelty check (Nov 5):**
- A 2024 survey of barren-plateau mitigation lists **no study testing initialization strategies in noisy hardware simulation**.
- The closest work is a preprint posted **Nov 3, 2026** (Shen, Jin, Guo, arXiv:2611.04198). It uses *idealized depolarizing* noise and a uniform [−0.2, 0.2] small-angle start on a periodic Heisenberg chain (N = 8). It does **not** test Zhang *et al.*'s variance rule γ² = 1/(4S(L+2)), and it doesn't use calibrated device noise (non-unital T1/T2, readout, SWAP overhead from real connectivity).
- So "Zhang *et al.*'s Gaussian init under calibrated IBM device noise" looks open in a noisy hardware simulation. **Do your own Google Scholar check** before writing "first", and phrase it as "to our knowledge".

**Setup details:**
- **Why:** the paper models noise only as Gaussian noise added to the gradient, i.e., shot noise. Real devices also have gate errors, T1/T2 decay and readout error.
- **Theory link:** Wang *et al.* (2021) show that local Pauli noise causes **noise-induced barren plateaus**, where gradients decay exponentially with *depth* whatever the initialization.
- **The question this answers:** does the Gaussian-init advantage survive the noisy hardware simulation, and how fast does it shrink with depth? A "no" here is a legitimate, interesting negative result.
- **Tooling:** `qiskit_noise_test.py` (in this folder). It:
  - builds the ansatz once as a parameterized circuit;
  - transpiles it to an IBM fake backend (real layout, native gates, SWAPs);
  - uses that backend's noise model in Aer with the **density-matrix** method, so expectation values are exact with noise and there's no shot noise;
  - gets parameter-shift gradients by sending all 2P shifted parameter sets in **one batched call**.
- **Precedent:** QuACK (NeurIPS 2023) used the same kind of fake backends (FakeLima, FakeManila).
- **Size limit:** a density matrix has 4^N entries.

  | Qubits | Entries | Memory (16 bytes each) | Feasible? |
  |---|---|---|---|
  | 5 | 1,024 | 16 KB | yes |
  | 7 | 16,384 | 256 KB | yes |
  | 10 | 1,048,576 | 16 MB | slow (1,000+ s per gradient by extrapolation) |
  | 15 | ~1.07 × 10⁹ | ~17 GB | **impossible on a laptop** |

  So run this at **5 qubits (FakeManilaV2)** and **7 qubits (FakeNairobiV2)**, not at the paper's 15.
- **Measured in my test** (2-core cloud machine; your laptop should be similar or faster):

  | Setting | Transpiled circuit | Time per gradient | Zero-init energy, noise off → on | Zero-init ‖∇‖² with noise |
  |---|---|---|---|---|
  | 5 qubits, 2 blocks, 20 params, FakeManilaV2 | 10 logical CZ → **34 CX** (linear chip, ring needs SWAPs), depth 84 | **0.2 s** | 4.000 → **3.553** (−11%) | 8 × 10⁻⁵ (noise breaks the zero-gradient symmetry) |
  | 7 qubits, 3 blocks, 42 params, FakeNairobiV2 | 63 CX, depth 176 | **5.3 s** | 6.000 → 5.195 (−13%) | 1.4 × 10⁻³ |

  - Noiseless control check: the zero-init gradient = 3.5 × 10⁻³⁰ and the energy = 4.0000 exactly, as it should be.
  - Doing the same through the PennyLane–Qiskit plugin took **~8 s per gradient at 5 qubits (40× slower)**, because it re-transpiles every shifted circuit, and it adds shot noise by default. Use Qiskit directly.
- **Protocol:**
  - Blocks ∈ {1, 2, 3, 4} at N = 5, plus blocks ∈ {2, 3} at N = 7.
  - × 3 inits × noise on/off × 3 seeds, 100 GD steps at learning rate 0.01.
- **Metrics:**
  - Statistical: ‖∇‖² at init vs depth, and energy after 100 steps, noise on vs off.
  - Hardware: transpiled depth, two-qubit gate count, circuit evaluations per gradient.
- **Caveat:** at 5–7 qubits, uniform init still has *larger* gradients than Gaussian init. My 5-qubit test gave ‖∇‖² = 3.3 vs 1.5, because barren plateaus haven't set in at that size. So E4b tests **how noise changes things as depth grows**, not the large-N barren-plateau claim; E1/E2 cover that noiselessly.
- **Compute:** 5 qubits ≈ 4 depths × 3 × 2 × 3 × 100 × 0.2 s ≈ **25 min**; 7 qubits ≈ 2 × 3 × 2 × 3 × 100 × 5.3 s ≈ **2.7 h** (my estimate). Cut seeds or steps at 7 qubits if needed.
- **Install:** `pip install qiskit qiskit-aer qiskit-ibm-runtime`. All have Python 3.14 Windows builds (checked Nov 5: qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0). My test ran on qiskit 2.6.1, qiskit-aer 0.17.2 and qiskit-ibm-runtime 0.45.1; the API used is the same.

### E5 (optional). LiH
- **Hamiltonian:** `qml.qchem.molecular_hamiltonian` with `active_electrons=2, active_orbitals=5`. I tested this: it gives **10 qubits** and `qml.qchem.excitations(2, 10)` gives **8 singles + 16 doubles = 24 Givens rotations**, matching the paper.
- **Bond length:** not given in the paper. Use ~1.6 Å and state it as an assumption.
- **Grid:** inits {Gaussian 1/147,456, uniform, zero} × {GD at learning rate 0.1, Adam at 0.01} × {exact, constant noise 0.001} × 5 seeds.
- **Compute:** under 1 h at 10 qubits.

**Total machine time for E0–E5 (+E3b):** roughly **2–3 h** on your CPU at 100 iterations per run, based on your smoke-test timing (0.057 s per gradient at 15 qubits). Compute is not the bottleneck; your coding and writing time is.

---

## 6. Implementation

### Software
- Python 3.11 or 3.12 on Windows: `pip install pennylane pennylane-lightning scipy matplotlib pandas`.
- Use the `lightning.qubit` simulator with `diff_method="adjoint"` for exact gradients (fast), and `diff_method="parameter-shift"` with `shots=` for E4.
- **No GPU needed.** At ≤ 16 qubits, a statevector is 2¹⁶ × 16 bytes ≈ 1 MB.

### Code layout (inside your course folder)
```
Project/
├── papers/                 # Zhang2022.pdf + related work
├── code/
│   ├── smoke_test.py       # already written: run tonight
│   ├── hamiltonians.py     # Heisenberg open chain; LiH via qml.qchem
│   ├── ansatz.py           # blocks of (CZ ring → RX → RY); Givens ansatz for LiH
│   ├── inits.py            # gaussian(γ² = 1/(4S(L+2))), uniform, zero
│   ├── train.py            # GD/Adam loop; optional Gaussian gradient noise; logs to CSV
│   ├── e1_gradnorm.py  e2_heisenberg.py  e3_sweep.py  e4_shots.py  e5_lih.py
│   └── plots.py
├── results/                # one CSV per run: seed, init, step, energy, grad_norm, time
├── figures/
├── proposal/  slides/  report/
```

### Noise model
The paper doesn't give a formula, so this is an assumption to state: add independent N(0, σ²) noise to every gradient component at each step, with σ² = 0.01 for Heisenberg and 0.001 for LiH.

---

## 7. Timeline

| Dates | Work | Output |
|---|---|---|
| **Thu Nov 5** | Run `smoke_test.py` (10 and 15 qubits); write and submit the proposal | Proposal ✔ |
| Fri Nov 6 – Sun Nov 8 | E0 validation; write `ansatz/inits/train.py`; **run E1** | Gradient-vs-qubits figure |
| Mon Nov 9 – Fri Nov 13 | **Run E2** (60 runs); start E4 | Training-curve figures |
| Sat Nov 14 – Mon Nov 16 | **Finish E4**; **E4b at 5 qubits** (start the 7-qubit runs overnight); sanity-check the uniform baseline against the paper | Shot-cost table; noisy-vs-noiseless figure |
| Tue Nov 17 – Thu Nov 19 | Assignment 3; let E3/E5 run in the background if you want them | (optional extras) |
| Fri Nov 20 – Sun Nov 22 | Final figures; build the slides | |
| **Mon Nov 23** | **Submit slides** | Slides ✔ |
| Nov 24 – Dec 3 | Rehearse to 15 ± 1 min; present; feedback forms | Presentation ✔ |
| Nov 27 – Dec 6 | Write the report; extra seeds or ablations if useful | |
| **Mon Dec 7** | **Submit report** | Report ✔ |

---

## 8. Ambiguities in the paper and how they were resolved

These are normal in a reproduction. Documenting how you resolved them is part of your contribution. The supplementary material (`3600270.3601622_supp.pdf`, from the ACM Digital Library) resolved three of them. It contains extra experiments and proofs, **but no code**.

| # | Ambiguity | Status | Resolution and source |
|---|---|---|---|
| 1 | Gate order inside a block, and layer counting | ✅ **Resolved** | Main text p. 7: "In each block, we first employ the CZ gate … followed by RX and RY rotations." Fig. 1 ends with R_X(θ_{L+1}) then R_Y(θ_{L+2}), and the proof of Thm 4.1 (supplement Eqs. 110–114) uses that order. So: **10 × (CZ ring → RX → RY)**, which gives 20 rotation layers = L + 2, so L = 18 and 300 parameters. `smoke_test.py` was corrected on Nov 5. (For the full Heisenberg H, both orders give identical energies, so the first smoke-test results stand. Individual XX terms do differ, by up to 0.64 in a test, so use the corrected order for E1.) |
| 2 | Noise model | ⚠ Still open | The supplement only repeats "the variance of measurement noises to be 0.01". Assume additive N(0, σ²) noise per gradient component, and say so. |
| 3 | Number of iterations | ✅ **Resolved** | Every supplement figure runs from iteration 0 to 100. Use **100**. |
| 4 | "0.01γ … 100γ" | ✅ **Resolved** (mostly) | Supplement Sec. A.1 calls them "Gaussian initialized VQAs with **variances** {0.01γ, 0.1γ, γ, 10γ, 100γ}", so scale the **variance**. (Its notation reuses γ for what Thm 4.1 calls γ²; note this in the report.) |
| 5 | LiH bond length and basis | ⚠ Still open | Not in the supplement either. Use STO-3G, ~1.6 Å, and an active space of 2 electrons in 5 orbitals, stated as assumptions. |
| 6 | LiH variance 1/147,456 vs Thm 4.2 | ✅ **Resolved** (main text p. 8) | The paper sets γ² = (8² × ½)/(48 × 8⁴ × 24) = 32/4,718,592 = **1/147,456**, "the (L, h, a, ε) = (24, 8, 8, ½) case of Theorem 4.2". It uses 48h⁴ as an upper bound on 16h²(3h(h−1)+1) = 173,056 < 48 × 8⁴ = 196,608, and treats ‖O‖ and the gradient factor as ≈ 1. |
| 7 | Typo in supplement | Note | Sec. A.1 says "(N, L) = (18.38)", meaning (18, 38). Sec. A.2's first sentence says "Heisenberg model" but describes LiH. |
| 8 | **Angle convention** (theory vs code) | ✅ **Resolved Nov 5 (18:50) by matching Fig. 2.** 15 qubits, GD, learning rate 0.01, one run each. **Scale 2** gives Gaussian f = −19.2 at iteration 10 and −22.8 at 100, uniform −21.3 at 100, uniform ‖∇‖ = 2.67 at the start. **Fig. 2(a,c)** shows ≈ −19, ≈ −22.8, ≈ −20 and ≈ 2.7. **Scale 1** gives −8.4, −20.2, −3.1 and 1.58, which doesn't match. So the authors used the theorem convention, and scale 2 is the reproduction setting. Scale 1 stays only as a sensitivity check. Original analysis follows. | Sec. 2 defines R_X(θ) = e^(−iθX), and Eq. (2) uses shifts of ±π/4 with no ½ factor, which matches that convention. But Sec. 5 says "all numerical experiments are provided using the Pennylane package", whose RX(t) = e^(−itX/2). If the authors used γ² = 1/160 directly on PennyLane angles, their effective variance was 1/640 in the theorem's units. **Main runs: `--angle-scale 2`** (theorem-faithful: γ² = 1/160 in paper units, i.e., 1/40 in PennyLane angles; learning rates and noise variance in paper units). **Sensitivity: `--angle-scale 1`** (the code-literal reading). Checked: in paper units, PennyLane's autodiff gradient equals the paper's Eq. (2), f(θ + π/4) − f(θ − π/4), to 2 × 10⁻¹⁵. |

### Useful facts from the supplement's proofs
- **Theorem 4.1 is proved per gradient component.** Eq. (108): each ∂f/∂θ_{q,n} on the S measured qubits has E(∂f/∂θ)² ≥ Tr[σ_j ρ_in]² / (S^(S+1)(L+2)^(S+1)). With S = 2 and L = 18 that's 1/(2³ · 20³) = **1/64,000 ≈ 1.6 × 10⁻⁵ per parameter**. Summing over L · S = 36 parameters gives the 5.6 × 10⁻⁴ total in §2. E1 can test the per-parameter version directly.
- **Lemma B.1 / B.2** give exact one-gate formulas, used in the E0 check above.
- **Theorem 4.2's proof needs γ² ≤ 1/(16h³)** (Lemma B.4). For the LiH 4-qubit Givens gates (h = 8): 1/(16 · 512) = 1/8,192. The paper's 1/147,456 satisfies this.

---

## 9. Proposal (due tonight): content checklist

Write it in your own words in the IEEE template. Covers guidance + outline requirements.

**Audience rule (project guidance.pdf):** the proposal is for the instructor and TAs.
- **Don't** introduce Adv. ML background: gradient descent, Adam, learning rates, initialization schemes, vanishing gradients.
- **Do** introduce the non-Adv. ML topics, which here means the quantum-computing terms. The skeleton has a dedicated "Quantum Computing Background" section (about 1 paragraph, one sentence per term):
  - qubits and the $2^N$ state;
  - rotation and CZ gates;
  - measurement, expectation values and shots;
  - Hamiltonian, ground energy, and locality $S$;
  - VQC and the parameter-shift rule;
  - device noise and fake backends.

  Define each through an ML analogy the readers already know (angles ≈ weights, energy ≈ loss, barren plateau ≈ vanishing gradients that are exponential in qubit count).

1. **Problem:** barren plateaus make deep VQCs untrainable from uniform random starts. Assume ML knowledge; 2–3 sentences.
1b. **Quantum Computing Background:** the terms listed above, nothing more.
2. **Paper and claims to reproduce:**
   - (i) Gaussian init with γ² = 1/(4S(L+2)) keeps gradients polynomially large;
   - (ii) it trains the 15-qubit Heisenberg model faster than uniform/zero init, with exact and noisy gradients.
3. **Plan:** E1, E2 and E4 core, with E3 and E5 if time allows; the PennyLane implementation; the timeline from §7.
4. **Contribution (lead with the first point):**
   - **Testing Zhang *et al.*'s Gaussian initialization in a noisy hardware simulation** (Qiskit Aer with IBM device noise models: gate errors, T1/T2, readout, real connectivity). The paper only models noise as Gaussian noise on gradients. Preliminary result: noise *amplifies* the Gaussian-vs-uniform gradient advantage from ~1.2–2× to 8.2× at 8 blocks (5 qubits), while the noise shrinks the signal ~6% per block.
   - An independent re-implementation, including the gradient-vs-qubits scaling plot the paper doesn't include.
   - A measurement-cost (shots) analysis.
   - Documented ambiguities and how they were resolved.
5. **Preliminary work:** your smoke-test table, i.e., gradient norms for 3 inits at 10 and 15 qubits plus seconds per gradient.
6. **Experiment plan (½ page):**
   - *Hypothesis:* Gaussian init reaches within 1.0 of the Heisenberg ground energy in fewer iterations than uniform init, and its gradient norm doesn't decay exponentially with qubit count.
   - *Proxy metrics:* iterations to the target energy, E‖∇f‖² vs N (statistical); wall-clock per gradient, circuit executions and shots to the target (hardware).
   - *Protocol:* N = 15, 300 parameters, 3 inits × {GD, Adam} × {exact, noisy σ² = 0.01} × 5 seeds; gradient-norm sweep over N = 4–14.
   - *Expected results:* the paper's ordering (Gaussian ≫ uniform; zero init moves only with noise). Negative results are acceptable.
7. **References (IEEE style):** the paper, McClean 2018, Cerezo 2021, Grant 2019, PennyLane (see §12).

---

## 10. Slides (15 min, rubric: structure, content, communication, media, time 15 ± 1)

| # | Slide | Time |
|---|---|---|
| 1 | Title | 0:20 |
| 2–3 | Motivation: VQCs; the barren plateau, with a numeric example of shrinking gradients | 1:30 |
| 4–5 | The paper's idea: γ² = 1/(4S(L+2)); the 526× variance contrast; Theorem 4.1 in one line | 2:30 |
| 6 | Reproduction setup: Heisenberg model, ansatz, baselines, metrics | 1:30 |
| 7 | E1: gradient vs qubits (log plot) | 1:30 |
| 8–9 | E2: training curves (GD/Adam, exact/noisy) | 2:30 |
| 10 | E4: shots and circuits to target (hardware axis) | 1:00 |
| 10b | **E4b: Gaussian vs uniform in the noisy hardware simulation (main contribution)** | 1:30 |
| 11 | What didn't match / ambiguities | 1:00 |
| 12 | Challenges: classical simulability critique (Cerezo 2023); zero-gradient start | 1:00 |
| 13 | Conclusion and future work | 0:45 |

Prepare answers for likely questions: "Why not just use zero init?", "Is a near-identity start classically simulable?", "Would this hold in a real hardware run, beyond the noisy hardware simulation?"

---

## 11. Report outline (6 pages, IEEE)

| Section | ~Length |
|---|---|
| I. Introduction: barren plateaus, the paper, your contributions list | 0.6 p |
| II. Background: VQCs, parameter-shift rule, barren plateaus, Theorem 4.1 | 0.8 p |
| III. Reproduction setup: Hamiltonians, ansatz, inits, noise model, the assumptions in §8 | 0.8 p |
| IV. Results: E1, E2, E4 (+ E3, E5) with figures and mean ± std over seeds | 2.0 p |
| V. Discussion: agreement with or departure from the paper; the zero-gradient start; classical simulability; limits of a 15-qubit study | 0.8 p |
| VI. Conclusion and future work | 0.3 p |
| References | ~0.5 p |

---

## 12. Reading list (download links)

1. **Main paper:** Zhang *et al.*, NeurIPS 2022 · [DOI 10.48550/arXiv.2203.09376](https://doi.org/10.48550/arXiv.2203.09376)
2. **Barren plateaus, original:** J. R. McClean *et al.*, "Barren plateaus in quantum neural network training landscapes," *Nature Communications* 9, 4812 (2018) · [DOI 10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
3. **Cost-function-dependent barren plateaus** (local vs global observables, i.e., why S matters): M. Cerezo *et al.*, *Nature Communications* 12, 1791 (2021) · [DOI 10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)
4. **An earlier initialization fix** (identity blocks), a good related-work contrast: E. Grant *et al.*, "An initialization strategy for addressing barren plateaus in parametrized quantum circuits," *Quantum* 3, 214 (2019) · [DOI 10.22331/q-2019-12-09-214](https://doi.org/10.22331/q-2019-12-09-214)
5. **Critique for your discussion section:** M. Cerezo *et al.*, "Does provable absence of barren plateaus imply classical simulability?" (2023) · [DOI 10.48550/arXiv.2312.09121](https://doi.org/10.48550/arXiv.2312.09121)
6. **Tool:** V. Bergholm *et al.*, "PennyLane: Automatic differentiation of hybrid quantum-classical computations" (2018) · [DOI 10.48550/arXiv.1811.04968](https://doi.org/10.48550/arXiv.1811.04968)
7. **Noise-induced barren plateaus** (theory for E4b): S. Wang *et al.*, "Noise-induced barren plateaus in variational quantum algorithms," *Nature Communications* 12, 6961 (2021) · [DOI 10.1038/s41467-021-27045-6](https://doi.org/10.1038/s41467-021-27045-6)
7b. **Closest related work for E4b (cite and contrast):** L. Shen, F. Jin, B. Guo, "Initialization-controlled fault sensitivity and force-balance plateaus in noisy variational quantum optimization," arXiv preprint, Nov 3, 2026 · [DOI 10.48550/arXiv.2611.04198](https://doi.org/10.48550/arXiv.2611.04198). It uses idealized depolarizing noise and a small-angle uniform start, not Zhang's Gaussian rule or device noise. It's two days old and not peer-reviewed.
7c. **Survey showing the gap:** "Investigating and Mitigating Barren Plateaus in Variational Quantum Circuits: A Survey" (2024) · [DOI 10.48550/arXiv.2407.17706](https://doi.org/10.48550/arXiv.2407.17706)
8. **Quantum-network motivation (intro only):** eQMARL, ICLR 2025 · [DOI 10.48550/arXiv.2405.17486](https://doi.org/10.48550/arXiv.2405.17486)

---

## 13. Risks and fallbacks

| Risk | Fallback |
|---|---|
| The paper's layer counting or noise model doesn't match your reading | Document the assumption (§8); it's still a valid reproduction |
| Uniform init isn't clearly worse at 15 qubits | Show the trend over N = 4–14 (E1); a negative or weaker result is acceptable |
| Running short on time near Nov 19 | Drop E3/E5, and keep E4b to 5 qubits. E0 + E1 + E2 + E4 are enough for slides and report |
| Noisy simulation too slow | Stay at 5 qubits (0.2 s per gradient); never try density-matrix noise above ~8 qubits |
| Overlap with your own research | Keep the project to this paper's experiments; check with your supervisor if unsure |
| Template conflict (IEEE vs ICML) | Ask on Brightspace before writing the proposal |
