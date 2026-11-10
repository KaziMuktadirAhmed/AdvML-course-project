# ENGI 9882 Course Project: Papers to Reproduce on This PC

*Prepared Nov 3, 2026; updated Nov 5 with a third search, a quantum-ML search (§1c), a combined risk ranking of all 28 papers (§1d) and download links. This replaces `Project_LitReview_Plan.md`. The project is a **reproduction** (template 9 in `project guidance.pdf`), not a literature review.*

---

## 1. What a candidate paper has to satisfy

From `project guidance.pdf` and the course outline:

1. **Venue:** a recent paper from NeurIPS, ICML, ICLR, AISTATS or JMLR (guidance, template 9).
2. **Not off-the-shelf:** you implement an advanced ML method or adapt an existing implementation. Running scikit-learn on a dataset is not enough (outline).
3. **Compared with a baseline on two axes** (outline):
   - **statistical** performance, e.g., iterations to reach an accuracy threshold;
   - **hardware** performance, e.g., throughput or wall-clock time.
4. **Runs on your PC.** You have an Intel i7-14700HX and an **NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM** (99 W cap), confirmed with `nvidia-smi`.
   - **Usable VRAM is about 6.4 GB, not 8.** The GPU also drives your display, and Windows plus your open apps were already using ~1.7 GB. Close Steam, Edge and Acrobat before long runs.
   - **The RTX 5070 is a Blackwell GPU (sm_120), and older PyTorch builds don't support it.** You need **PyTorch 2.7 or newer with a CUDA 12.8+ build**, e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu128` or a newer CUDA build. Repos that pin old torch/transformers versions will fail with "no kernel image is available" errors until you upgrade those versions. This mainly affects **C (MeZO)**.
   - Check that it works: `python -c "import torch; print(torch.cuda.get_device_name(0), torch.randn(2,2).cuda().sum())"`

Proposal (due **Thu Nov 5**) needs, per the outline: a ~1-page description, ≥1 citation, some preliminary work, and a ½-page experiment plan (hypothesis, proxy metric, protocol, expected results). Each candidate below lists those pieces.

---

## 1b. How risk is scored

Each paper gets **1 (low), 2 (medium) or 3 (high)** on four factors, which add up to a **total from 4 (safest) to 12 (riskiest)**:

| Factor | 1 = low | 2 = medium | 3 = high |
|---|---|---|---|
| **Compute** | Under ~5 h total on your RTX 5070 (6.4 GB usable) or CPU | ~5–25 h, or needs a reduced scale | Over ~25 h even when reduced, or memory is tight |
| **Setup** | Maintained pip package that works with PyTorch ≥ 2.7 | No official code (you implement it), or old pins / dataset prep / Windows quirks | Old pinned versions **and** custom kernels, unsupported frameworks or other blockers |
| **Result** | Headline claim checkable at laptop scale; well replicated | Absolute numbers depend on a checkpoint, small gains need many seeds, best-of-N reporting, or the claim is fuzzy | The headline metric needs something you don't have (paid API, cluster, paper-scale steps), or the paper is mostly theory |
| **Course fit** | Venue + recent (2022+), and both the statistical **and** hardware axes are natural | One weak point: 2021, weak hardware axis, very small method, or D&B track | 2019 or older (the guidance says "recent"), or the reproduction target is vague |

Compute figures are my estimates from model size and the papers' own reported runtimes, not measurements. Ties are broken by course fit, then by the worst single factor. ⚛ marks quantum papers.

---

## 1c. Quantum ML search (Nov 5): ranked by risk

I searched NeurIPS, ICML, ICLR, AISTATS and JMLR for quantum machine learning, quantum networks and quantum communication. Two papers from the earlier searches (I and Q) are quantum, so this ranking covers 10 papers.

**About quantum networks:** the machine-learning work on quantum *networks* that I found mostly appears in networking and quantum-engineering venues, not the five ML venues the guidance accepts. That includes RL for entanglement routing and entanglement distribution, e.g., an ACM 2025 paper "Reinforcement Learning for Entanglement Distribution in Quantum Networks" ([DOI 10.1145/3749096.3750028](https://doi.org/10.1145/3749096.3750028)). Two accepted-venue papers touch the theme:
- **AA. eQMARL** (ICLR 2025): multi-agent RL where agents' quantum critics are coupled through **entangled quantum channels**.
- **AB. Exponential Quantum Communication Advantage in Distributed Inference and Learning** (NeurIPS 2024): distributed learning over a **quantum network**. It is mostly theory.

If you want a pure quantum-network routing paper, you'd need the instructor to accept a non-listed venue.

### Quantum ranking (lowest risk first)

| Q-rank | # | Paper (venue) | Compute | Setup | Result | Fit | **Total** | Main risk in one line |
|---|---|---|---|---|---|---|---|---|
| 1 | ⚛ **I** | Escaping the Barren Plateau via Gaussian Initializations (NeurIPS 2022) | 1 | 2 | 1 | 1 | **5** | No code release; you write it in PennyLane |
| 2 | ⚛ **U** | QuACK: Koopman-accelerated quantum optimization (NeurIPS 2023) | 1 | 2 | 2 | 1 | **6** | No code link found; speedup size depends on the regime |
| 3 | ⚛ **W** | Quantum Implicit Neural Representations, QIREN (ICML 2024) | 1 | 2 | 2 | 1 | **6** | Paper reports the best of 5 runs, so mean gains may be smaller |
| 4 | ⚛ Z | Power and Limitations of Single-Qubit Native QNNs (NeurIPS 2022) | 1 | 2 | 1 | 2 | **6** | Mostly theory; hardware axis is weak |
| 5 | ⚛ Y | Equivariant Quantum Graph Circuits (ICML 2022 spotlight) | 1 | 2 | 1 | 2 | **6** | No code; no empirical GNN baseline, so you add one |
| 6 | ⚛ Q | Exponentially Many Local Minima in QNNs (ICML 2021 spotlight) | 1 | 2 | 1 | 2 | **6** | 2021; hardware axis is weak |
| 7 | ⚛ **V** | Quantum Deep Equilibrium Models, QDEQ (NeurIPS 2024) | 2 | 2 | 2 | 1 | **7** | ~45 min–1.5 h per run; paper reports single runs with small margins |
| 8 | ⚛ X | Parametrized Quantum Policies for RL (NeurIPS 2021) | 2 | 2 | 1 | 2 | **7** | Paper used Cirq/Qulacs + a TensorFlow Quantum tutorial; 20 agents per curve |
| 9 | ⚛ AA | eQMARL: Entangled Quantum Multi-Agent RL over Quantum Channels (ICLR 2025) | 3 | 3 | 2 | 1 | **9** | 3.5–8.5 h per run on a 128-core cluster × 10 runs; TFQ/Cirq; no code link found |
| 10 | ⚛ AB | Exponential Quantum Communication Advantage in Distributed Inference and Learning (NeurIPS 2024) | 2 | 3 | 3 | 2 | **10** | Mostly theory; experiment details and code unclear |

### Notes on the new quantum papers

**U. QuACK** (Luo, Shen, Dangovski, Soljačić; NeurIPS 2023). This is the strongest optimization fit after I.
- **Claim:** learning a Koopman operator from a short window of the optimizer's parameter trajectory predicts where the parameters are heading. That cuts the number of quantum gradient evaluations.
- **Reported speedups:**
  - more than 200× in the overparameterized regime (2–5 qubits);
  - more than 10× in the smooth regime (2–12 qubits);
  - 3.24× (Ising), 4.63× (LiH) and 4.38× (QML, binary MNIST "1" vs "9", 10 qubits) in the non-smooth regime;
  - 20.18× with quantum natural gradient on a 5-qubit Ising model.
- **Software in the paper:** Qiskit, PyTorch, Yao and PennyLane. One VQE run used the real IBM device ibmq_lima.
- **Reduced reproduction:** transverse-field Ising VQE (h = 0.5) at 4–8 qubits. Compare plain GD/Adam with GD/Adam + QuACK.
- **Statistical metric:** gradient (circuit) evaluations to reach a target energy. **Hardware metric:** wall-clock including the Koopman-fitting overhead.
- **Sample hypothesis:** "On 6-qubit Ising VQE, QuACK reaches the GD baseline's final energy (±10⁻³) with at least 3× fewer gradient evaluations, and its classical overhead is under 10% of wall-clock."

**V. QDEQ** (Schleich *et al.*, NeurIPS 2024).
- **Method:** a deep-equilibrium (implicit, fixed-point) quantum model trained with TorchQuantum, using a Broyden root solver capped at 10 steps.
- **Data:** MNIST-4 on 4 qubits; MNIST-10, Fashion-MNIST and CIFAR-10 on 10 qubits.
- **Reported test accuracy:**

  | Dataset | QDEQ (implicit) | Best direct (unrolled) |
  |---|---|---|
  | MNIST-10 | 73.68% | 72.78% |
  | Fashion-MNIST-10 | 72.11% | 71.83% |
  | MNIST-4, angle encoding | 86.7% | 85.3% |

- **Hardware numbers the paper reports** (runs on an RTX 2070):
  - MNIST-10 peak GPU memory: 273 MB vs 1,233 MB for the 10-layer direct model.
  - MNIST-4 runtime: 5,277 s vs 7,820 s.
- **Code:** `github.com/martaskrt/qdeq`
- **Why it's worth doing:** the paper reports single runs without error bars, so adding 3 seeds is a real contribution.

**W. QIREN** (Zhao *et al.*, ICML 2024).
- **Method:** hybrid layers, each a linear layer + BatchNorm + an 8-qubit data re-uploading circuit, used to fit signals.
- **Reported results** (MSE ×10⁻³, QIREN vs SIREN):

  | Signal | QIREN | SIREN |
  |---|---|---|
  | Astronaut image (32×32) | 4.0 | 9.0 |
  | Coffee image | 1.5 | 2.3 |
  | Bach cello sound (1,000 samples) | 5.5 | 5.5 (tie) |

- Uses about 650 parameters vs about 700 for SIREN, and reports ~35% memory savings.
- **Code:** `github.com/GGorMM1/QIREN`
- Skip the image-generation experiment (FFHQ/CelebA-HQ, 6,048 kimg); it's far too heavy.
- **Note:** the paper trains each model 5 times and reports the **best** run. Report mean ± std, which may shrink the gap.

**X. Parametrized Quantum Policies for RL** (Jerbi *et al.*, NeurIPS 2021).
- **Setup:** REINFORCE with PQC policies on CartPole-v1 (4 qubits), MountainCar (2) and Acrobot (6). SOFTMAX-PQC beats RAW-PQC and is comparable to DNN policies.
- The paper used Cirq with a Qulacs backend, and a TensorFlow Quantum tutorial reproduces it. As far as I know, TensorFlow Quantum doesn't support native Windows, so use WSL2 or re-implement in PennyLane.
- **Course link:** RL is on the syllabus.

**Y. Equivariant Quantum Graph Circuits** (Mernyei *et al.*, ICML 2022 spotlight).
- **Setup:** one qubit per node. Tasks are telling two triangles apart from a 6-cycle, and classifying graphs as one cycle vs two cycles (6–10 nodes, 8-cycles held out).
- Training: Adam, 100 epochs, 10 runs.
- **What you'd add:** the paper argues against message-passing GNNs only in theory, so adding a real GNN baseline would be your contribution.

**Z. Single-qubit native QNNs** (Yu *et al.*, NeurIPS 2022).
- **Claim:** a single-qubit data re-uploading circuit (the "YZY" structure) is a partial Fourier series. It can approximate any univariate function, but it has limits on multivariate ones.
- Experiments are small function-fitting tasks. Easy to run, but the hardware axis is weak.

**AA. eQMARL** (DeRieux & Saad, ICLR 2025). This is the closest match to quantum networks.
- **Setup:** 2 agents with 4 qubits each. Their quantum critics are coupled through entangled Bell-pair channels (Ψ⁺ chosen) on the CoinGame-2 environment.
- **Reported results:**
  - converges up to 17.8% faster than a centralized quantum baseline (POMDP);
  - converges 10.8% faster than a split classical baseline (MDP);
  - uses 25× fewer centralized critic parameters.
- **Compute:** 3,000 epochs, averaged over 10 runs. Each run took 3.5 h (MDP) or 8.5 h (POMDP) on a 128-core cluster node. I can't predict the laptop time from that.
- **Software:** TensorFlow Quantum + Cirq.
- **Reduced version:** MDP only, 3 seeds, fewer epochs, re-implemented in PennyLane. It's doable but the riskiest quantum option.

**AB. Exponential quantum communication advantage** (NeurIPS 2024).
- **Claim:** for certain compositional models, including some graph neural networks, distributed inference and training over a quantum network need exponentially less communication than classical versions. The abstract says the models "perform well on standard benchmarks," but I couldn't confirm experiment details or code.
- **Use it as:** background reading or motivation for a quantum-network angle, not as the reproduction target.

### My quantum recommendation
- **Lowest risk:** **I (Gaussian init)**.
- **Best optimization-flavoured alternative:** **U (QuACK)**. Its claim is a speedup in circuit evaluations, so both course axes come naturally. It also pairs with I: you could check whether Gaussian init and QuACK together cut iterations further.
- **For a quantum-network angle without eQMARL's risk:** reproduce I or U, and use AA/AB in the introduction to motivate why trainable, efficient QML matters for distributed quantum systems.

---

## 1d. Combined risk ranking (all four searches, 28 papers)

**Download** links: **DOI** is the paper's arXiv DOI, which is official and resolves to the arXiv page with the PDF. **Venue** links to the conference/journal version where I found it.

| Rank | # | Paper (venue) | C | S | R | F | **Total** | Download |
|---|---|---|---|---|---|---|---|---|
| 1 | H | DPM-Solver (NeurIPS 2022) | 1 | 1 | 2 | 1 | **5** | [DOI](https://doi.org/10.48550/arXiv.2206.00927) · [venue](https://neurips.cc/virtual/2022/poster/54266) |
| 2 | B | KAN: Kolmogorov–Arnold Networks (ICLR 2025) | 1 | 1 | 2 | 1 | **5** | [DOI](https://doi.org/10.48550/arXiv.2404.19756) |
| 3 | ⚛ **I** | Gaussian init for barren plateaus (NeurIPS 2022) | 1 | 2 | 1 | 1 | **5** | [DOI](https://doi.org/10.48550/arXiv.2203.09376) · [venue PDF](https://papers.nips.cc/paper_files/paper/2022/file/7611a3cb5d733e628081431445cb01fd-Paper-Conference.pdf) |
| 4 | D | Progress Measures for Grokking (ICLR 2023) | 1 | 1 | 1 | 2 | **5** | [DOI](https://doi.org/10.48550/arXiv.2301.05217) · [venue](https://iclr.cc/virtual/2023/poster/11385) |
| 5 | A | The Road Less Scheduled / Schedule-Free (NeurIPS 2024 oral) | 2 | 1 | 2 | 1 | **6** | [DOI](https://doi.org/10.48550/arXiv.2405.15682) |
| 6 | F | GaLore (ICML 2024 oral) | 1 | 2 | 2 | 1 | **6** | [DOI](https://doi.org/10.48550/arXiv.2403.03507) · [venue](https://icml.cc/virtual/2024/oral/35485) |
| 7 | ⚛ **U** | QuACK (NeurIPS 2023) | 1 | 2 | 2 | 1 | **6** | [DOI](https://doi.org/10.48550/arXiv.2211.01365) · [proceedings DOI](https://doi.org/10.52202/075280-1116) |
| 8 | ⚛ **W** | Quantum Implicit Neural Representations (ICML 2024) | 1 | 2 | 2 | 1 | **6** | [DOI](https://doi.org/10.48550/arXiv.2406.03873) · [venue](https://proceedings.mlr.press/v235/zhao24l.html) |
| 9 | N | GPTQ (ICLR 2023) | 1 | 2 | 1 | 2 | **6** | [DOI](https://doi.org/10.48550/arXiv.2210.17323) |
| 10 | J | Tent: test-time adaptation (ICLR 2021 spotlight) | 1 | 1 | 2 | 2 | **6** | [DOI](https://doi.org/10.48550/arXiv.2006.10726) · [venue](https://iclr.cc/virtual/2021/poster/2874) |
| 11 | ⚛ Z | Single-qubit native QNNs (NeurIPS 2022) | 1 | 2 | 1 | 2 | **6** | [DOI](https://doi.org/10.48550/arXiv.2205.07848) · [venue PDF](https://papers.neurips.cc/paper_files/paper/2022/file/b250de41980b58d34d6aadc3f4aedd4c-Paper-Conference.pdf) |
| 12 | ⚛ Y | Equivariant Quantum Graph Circuits (ICML 2022 spotlight) | 1 | 2 | 1 | 2 | **6** | [DOI](https://doi.org/10.48550/arXiv.2112.05261) · [venue](https://icml.cc/virtual/2022/poster/16081) |
| 13 | ⚛ Q | Exponentially Many Local Minima in QNNs (ICML 2021 spotlight) | 1 | 2 | 1 | 2 | **6** | [DOI](https://doi.org/10.48550/arXiv.2110.02479) · [venue](https://icml.cc/virtual/2021/spotlight/10074) |
| 14 | R | GPyTorch / BBMM (NeurIPS 2018 spotlight) | 1 | 1 | 1 | 3 | **6** | [DOI](https://doi.org/10.48550/arXiv.1809.11165) · [venue PDF](https://papers.nips.cc/paper/2018/file/27e8e17134dd7083b050476733207ea1-Paper.pdf) |
| 15 | E | FedAvg (AISTATS 2017) | 1 | 1 | 1 | 3 | **6** | [DOI](https://doi.org/10.48550/arXiv.1602.05629) · [venue](https://mlanthology.org/aistats/2017/mcmahan2017aistats-communication) |
| 16 | M | SGC: Simplifying GCNs (ICML 2019 oral) | 1 | 1 | 1 | 3 | **6** | [DOI](https://doi.org/10.48550/arXiv.1902.07153) · [venue](https://icml.cc/virtual/2019/poster/3683) |
| 17 | ⚛ **V** | Quantum Deep Equilibrium Models (NeurIPS 2024) | 2 | 2 | 2 | 1 | **7** | [DOI](https://doi.org/10.48550/arXiv.2410.23940) · [venue](https://proceedings.neurips.cc/paper_files/paper/2024/hash/386432c7534eec9a1cd7cbeea90d7e9f-Abstract.html) |
| 18 | G | LoRA+ (ICML 2024) | 2 | 1 | 2 | 2 | **7** | [DOI](https://doi.org/10.48550/arXiv.2402.12354) · [venue](https://proceedings.mlr.press/v235/hayou24a.html) |
| 19 | O | Task Arithmetic (ICLR 2023) + TIES-Merging (NeurIPS 2023) | 2 | 2 | 1 | 2 | **7** | [DOI](https://doi.org/10.48550/arXiv.2212.04089) · [TIES DOI](https://doi.org/10.48550/arXiv.2306.01708) |
| 20 | ⚛ X | Parametrized Quantum Policies for RL (NeurIPS 2021) | 2 | 2 | 1 | 2 | **7** | [DOI](https://doi.org/10.48550/arXiv.2103.05577) · [venue](https://neurips.cc/virtual/2021/poster/28290) |
| 21 | K | Classic GNNs are Strong Baselines (NeurIPS 2024 D&B) | 2 | 2 | 2 | 2 | **8** | [DOI](https://doi.org/10.48550/arXiv.2406.08993) · [venue](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b10ed15ff1aa864f1be3a75f1ffc021b-Abstract.html) |
| 22 | L | Deep Learning on a Data Diet (NeurIPS 2021) | 3 | 2 | 1 | 2 | **8** | [DOI](https://doi.org/10.48550/arXiv.2107.07075) |
| 23 | P | Sparse Autoencoders in LMs (ICLR 2024) | 1 | 2 | 3 | 2 | **8** | [DOI](https://doi.org/10.48550/arXiv.2309.08600) · [venue](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Abstract-Conference.html) |
| 24 | S | CleanRL (JMLR 2022) | 2 | 2 | 2 | 2 | **8** | [DOI](https://doi.org/10.48550/arXiv.2111.08819) · [venue](https://www.jmlr.org/beta/papers/v23/21-1342.html) |
| 25 | C | MeZO (NeurIPS 2023) | 3 | 3 | 2 | 1 | **9** | [DOI](https://doi.org/10.48550/arXiv.2305.17333) · [venue](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract.html) |
| 26 | ⚛ AA | eQMARL (ICLR 2025) | 3 | 3 | 2 | 1 | **9** | [DOI](https://doi.org/10.48550/arXiv.2405.17486) · [venue](https://iclr.cc/virtual/2025/poster/29050) |
| 27 | T | SAM (ICLR 2021 spotlight) | 3 | 2 | 2 | 2 | **9** | [DOI](https://doi.org/10.48550/arXiv.2010.01412) · [venue](https://iclr.cc/virtual/2021/spotlight/3497) |
| 28 | ⚛ AB | Exponential Quantum Communication Advantage (NeurIPS 2024) | 2 | 3 | 3 | 2 | **10** | [DOI](https://doi.org/10.48550/arXiv.2310.07136) · [venue](https://papers.neurips.cc/paper_files/paper/2024/hash/3639efedc51a522595372f76b91cbb25-Abstract-Conference.html) |

Related papers mentioned in the notes: D-Adaptation, ICML 2023 ([DOI](https://doi.org/10.48550/arXiv.2301.07733)); ZO-Bench, ICML 2024 ([DOI](https://doi.org/10.48550/arXiv.2402.11592)); Data Diet reproduction study, 2023 ([DOI](https://doi.org/10.48550/arXiv.2303.14753)).

### About the DOIs
NeurIPS, ICML, ICLR, AISTATS and JMLR mostly don't assign DOIs to their papers; QuACK's NeurIPS 2023 proceedings DOI is the one exception I found. Every arXiv paper has an official DOI of the form `10.48550/arXiv.<id>`, so those are the DOIs listed. The arXiv version is usually identical or very close to the conference version. When you cite in the report, cite the **venue** version (IEEE style) and add the arXiv DOI if you like. All of these are open access, so you don't need the MUN library proxy.

### Bottom line
- **Lowest risk overall (5):** H, B, **I ⚛**, D.
- **Lowest-risk quantum option:** **I**, tied for the lowest risk overall.
- **Next quantum options (6):** **U (QuACK)** and **W (QIREN)**. Both are recent, at accepted venues, and fit both course axes.
- **Avoid with the proposal due tonight:** C, AA, T, AB.

---

## 2. Search 1: original shortlist

| # | Paper | Venue | Topic | Fits on 8 GB? | Compute for a reduced reproduction (my estimate) | Risk |
|---|---|---|---|---|---|---|
| **A** | **The Road Less Scheduled** (Schedule-Free optimization) | NeurIPS 2024 (oral) | Optimization | ✅ | ~1–3 h per CIFAR-10 run; ~10–20 h total | Medium |
| **B** | **KAN: Kolmogorov–Arnold Networks** | ICLR 2025 | New architecture | ✅ (CPU is enough) | Minutes per run | Low |
| **C** | **MeZO: Fine-Tuning LMs with Just Forward Passes** | NeurIPS 2023 | Zeroth-order optimization | ✅ for RoBERTa-large | Paper scale is too slow; a reduced version takes ~1–3 h per run | Medium–High |
| D | **Progress Measures for Grokking via Mechanistic Interpretability** | ICLR 2023 | Interpretability / training dynamics | ✅ (tiny) | ~10–30 min per run | Low (weak hardware axis) |
| E | **Communication-Efficient Learning of Deep Networks from Decentralized Data** (FedAvg) | AISTATS 2017 | Federated learning (in syllabus) | ✅ (CPU is enough) | Minutes–1 h per run | Low (but not recent) |

The time figures are my rough estimates, scaled from model size. They are not measurements. Before committing, run each smoke test in §4 to get a real number.

(The "Risk" column above was my first rough call. The scored ranking in §1d supersedes it.)

---

## 3. Candidate details

### A. Schedule-Free optimization — Defazio *et al.*, NeurIPS 2024 (oral)
- **Claim to reproduce:** Schedule-Free SGD/AdamW does as well as or better than a tuned cosine learning-rate schedule, without needing to know the training length in advance. It needs no extra memory over SGD with momentum.
- **Code:** `github.com/facebookresearch/schedule_free` (`pip install schedulefree`)
- **Paper setup that fits your PC:** CIFAR-10, Wide ResNet 16-8, 300 epochs, on one V100 in the paper. There are also 12 convex logistic-regression problems from LIBSVM, which run on a CPU.
- **Reduced reproduction:**
  - **Baseline:** SGD with momentum and a cosine schedule.
  - **Method:** Schedule-Free SGD.
  - Optional third arm: D-Adaptation (ICML 2023 Outstanding Paper, same research group, code at `github.com/facebookresearch/dadaptation`).
  - Run 3 seeds each, at 100 epochs instead of 300.
  - If time allows, add the LIBSVM convex experiments, which are cheap.
- **Statistical metric:** test accuracy vs epoch; epochs to reach a target accuracy (e.g., 93%); final accuracy.
- **Hardware metric:** time per epoch, total wall-clock, peak GPU memory. The paper's claim of no extra memory is directly testable.
- **Watch out for:** BatchNorm models need their running statistics recomputed before each evaluation. The paper flags this. Warmup is still required, and the learning rate and weight decay need a new sweep.
- **Sample hypothesis:** "Schedule-Free SGD reaches the cosine baseline's final accuracy (±0.3%) on CIFAR-10 WRN-16-8 with no increase in peak memory and <5% extra time per epoch."

### B. KAN — Liu *et al.*, ICLR 2025
- **Claim to reproduce:** On function fitting, smaller KANs match or beat larger MLPs and follow a faster neural scaling law. Test RMSE falls as N^(−4) with cubic splines (k = 3).
- **Code:** `github.com/KindXiaoming/pykan` (`pip install pykan`)
- **Compute:** the authors say "all experiments reported in the paper are reproducible on CPUs usually within minutes, at most in a day."
- **Reduced reproduction:** the five toy functions from §3.1. Examples are J₀(20x), exp(sin(πx) + y²) and xy. Fit KANs at increasing grid sizes and MLPs at increasing widths, then plot test RMSE vs parameter count on log-log axes and fit the slope.
- **Statistical metric:** test RMSE vs number of parameters; the fitted scaling exponent; steps to reach a target RMSE.
- **Hardware metric:** wall-clock time per step and in total for KAN vs MLP at the same parameter count. The paper itself admits KANs are "slightly slower than MLPs to train in terms of wall time", so you can quantify that.
- **Sample hypothesis:** "KAN's empirical scaling exponent on the toy functions is close to 4, vs ≤ 2 for MLPs, but KAN wall-clock per step is at least 2× the MLP's at equal parameter count."
- **Note:** KAN has many published follow-ups and re-evaluations. That helps you find related work, but your contribution should be your own measurements, especially the hardware side.

### C. MeZO — Malladi *et al.*, NeurIPS 2023
- **Claim to reproduce:** a memory-efficient zeroth-order SGD method (an SPSA estimate from two forward passes, done in place) fine-tunes language models with inference-level memory. It gets close to full fine-tuning accuracy and uses up to 12× less memory.
- **Code:** `github.com/princeton-nlp/MeZO`. Its `medium_models/` folder covers the RoBERTa-large experiments.
- **What fits on your PC:** RoBERTa-large (350M parameters) with k = 16 examples per class on SST-2 / SST-5 / SNLI / TREC. MeZO itself needs only inference memory, about 1.5 GB of weights, so it fits easily.
  - Full fine-tuning with Adam needs roughly 5.7 GB for the fp32 weights, gradients and Adam states, plus activations. That likely **won't fit in your ~6.4 GB of free VRAM** without tricks such as a small batch, mixed precision or 8-bit Adam.
  - Your fallback baselines are FT with those tricks, or RoBERTa-base FT. Either way, the out-of-memory point is a hardware result you can report.
  - The repo's pinned library versions are old (see the Blackwell note in §1), so expect some setup work to get it running. The OPT-13B/30B/66B experiments are out of reach (the paper used 80 GB A100s).
- **Compute problem:** the paper runs MeZO for **100K steps** on RoBERTa-large, vs 1K steps for fine-tuning. By my rough estimate, that is roughly a day per run on a laptop GPU. A reduced version would use 1–2 tasks, fewer steps (e.g., 20K) and a smaller batch, and would report accuracy vs steps rather than only the final number.
- **Companion paper:** Zhang *et al.*, "Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark," ICML 2024 (`github.com/ZO-Bench/ZO-LLM`). It reports SST-2 accuracy with OPT-1.3B, e.g., 91.1% for FO-SGD vs 90.8% for ZO-SGD, using 20K ZO iterations vs 625 FO iterations. That's useful for reasoning about how many steps you need.
- **Statistical metric:** accuracy vs number of steps / forward passes; steps to reach a target accuracy.
- **Hardware metric:** peak GPU memory (the headline claim), time per step, total wall-clock.
- **Caveat from the authors' own project page:** MeZO isn't recommended for BERT-size models because they converge more slowly. That is worth testing, but it's also why the step count is so large.

### D. Grokking progress measures — Nanda *et al.*, ICLR 2023
- **Claim to reproduce:** a 1-layer transformer trained on modular addition (p = 113, 30% of pairs used for training) "groks" after ~10K epochs. It learns a Fourier-based algorithm with 5 key frequencies, and the "restricted" and "excluded" loss progress measures improve steadily before test accuracy jumps. Grokking doesn't happen without weight decay.
- **Setup:** d_model = 128, 4 heads, MLP width 512, full-batch AdamW, learning rate 1e-3, weight decay 1, 40K epochs. Tiny.
- **Code:** a Colab replication notebook, linked from `neelnanda.io/grokking-paper`.
- **Statistical metric:** epochs to grok vs weight decay and training fraction.
- **Hardware metric:** weak. Wall-clock is small and uninteresting. You'd have to add something such as GPU vs CPU throughput or a full-batch vs mini-batch cost comparison.
- **Best if:** you want a very safe project with lots of room for analysis, and you're comfortable with an interpretability topic.

### E. FedAvg — McMahan *et al.*, AISTATS 2017
- **Claim to reproduce:** compared with FedSGD, federated averaging cuts the number of **communication rounds needed to reach a target accuracy** by a large factor, on IID and on pathological non-IID splits of MNIST.
- **Why it fits:** the paper's own metric is *rounds to reach a target accuracy*, exactly the outline's statistical axis. Federated learning is also on the course syllabus.
- **Compute:** small CNNs on MNIST, simulated clients on one machine. Minutes to an hour.
- **Downside:** it's from 2017, while the guidance says "recent". You could ask the instructor, or reproduce SCAFFOLD (ICML 2020) on top of it as the "newer" method.

---

## 3b. More candidates (second search, Nov 3)

All of these are from venues the guidance accepts, and all fit in ~6.4 GB of VRAM or run on the CPU. Time figures are my rough estimates, not measurements.

| # | Paper | Venue | Syllabus topic | Compute (my estimate) | Statistical axis | Hardware axis | Main caveat |
|---|---|---|---|---|---|---|---|
| **F** | **GaLore**: memory-efficient training via gradient low-rank projection | ICML 2024 (oral) | Optimization / training large models | ~0.5–2 h per GLUE task (RoBERTa-base) | GLUE score: GaLore vs LoRA vs full FT | **Optimizer + total memory** (the headline claim) | Memory savings are smaller in absolute terms at RoBERTa-base scale |
| **G** | **LoRA+**: different learning rates for the LoRA A and B matrices | ICML 2024 | Fine-tuning (L12) | ~0.5–1 h per task (RoBERTa-base / GPT-2) | **Steps to reach accuracy** (claims up to ~2× faster fine-tuning, +1–2% accuracy) | Time per step, memory vs full FT | The method is small (a learning-rate ratio), so add analysis, e.g., a sweep of λ |
| **H** | **DPM-Solver**: fast ODE solver for diffusion sampling in ~10 steps | NeurIPS 2022 | Generative AI | Training-free; ~10–30 min per sampler setting (50K CIFAR-10 images) | **FID vs number of function evaluations**: claims 4.70 at 10 NFE, 2.87 at 20 NFE | **Sampling time per image**; claims 4–16× speedup over previous samplers | Absolute FID depends on the pretrained checkpoint |
| **I** | **Escaping the Barren Plateau via Gaussian Initializations in Deep VQCs** | NeurIPS 2022 | Optimization (quantum ML) | CPU; the paper says "each experiment finishes quickly on CPUs within a few hours" | **Gradient norm vs depth; iterations to reach ground energy**: Gaussian vs uniform vs zero init | Simulation wall-clock vs qubits and depth | No code link in the paper, so you implement it in PennyLane |
| **J** | **Tent**: fully test-time adaptation by entropy minimization | ICLR 2021 (spotlight) | Robustness / online learning | Minutes; you adapt a pretrained CIFAR-10 model on CIFAR-10-C | Error at severity 5: 40.8% (source) → 17.3% (BN adapt) → **14.3% (Tent)** | Extra cost of adaptation vs plain inference | 2021: borderline "recent" |
| **K** | **Classic GNNs are Strong Baselines** (tuned GCN/GAT/SAGE vs graph transformers) | NeurIPS 2024 (Datasets & Benchmarks track) | Graph learning | Minutes–1 h on the small/medium datasets; skip the huge ones | Accuracy: tuned classic GNNs match/beat GTs on 17 of 18 datasets | Training time and memory, GNN vs graph transformer | D&B track; GT baselines take more setup |
| **L** | **Deep Learning on a Data Diet** (EL2N/GraNd data pruning) | NeurIPS 2021 | Data efficiency | Heavy: scores averaged over 10 short runs plus several full ResNet-18 runs (~10–20 h reduced) | Accuracy vs fraction of data kept; claims you can **prune 50% of CIFAR-10** while slightly improving accuracy | Training time saved by pruning | A 2023 reproduction study already exists, so less is left to find |
| **M** | **Simplifying Graph Convolutional Networks (SGC)** | ICML 2019 (oral) | Graph learning | Minutes | Accuracy vs GCN on citation graphs | Claims **up to ~100× speedup over FastGCN** | 2019: not recent |

### Notes on the strongest new options

**H: DPM-Solver**: the cheapest clean fit to the course's two axes.
- **What you compare:** DPM-Solver vs DDIM (the baseline) at 10/20/50 NFE, using one pretrained CIFAR-10 diffusion model.
- **Statistical:** FID vs NFE. **Hardware:** seconds per 1K images.
- **Code:** official repo `github.com/LuChengTHU/dpm-solver`. The sampler also exists in Hugging Face `diffusers` (`DPMSolverMultistepScheduler`), with a pretrained CIFAR-10 DDPM checkpoint (`google/ddpm-cifar10-32`).
- **Caveat:** the paper's 4.70/2.87 FIDs come from its own score-model checkpoint. A different checkpoint gives different absolute FIDs, so reproduce the **trend** (DPM-Solver ≪ DDIM at low NFE) and say so in the report.
- **Sample hypothesis:** "At 10 NFE, DPM-Solver's FID is at least 2× lower than DDIM's on the same checkpoint, with equal time per step."

**I: Gaussian initialization for VQCs**: the best fit for your research, and it uses only your CPU.
- **Setup to reproduce:** Heisenberg-model ground energy, 15 qubits, 10 layers of RY-RX-CZ (300 parameters), Gaussian N(0, 1/160) vs uniform [0, 2π] vs zero init, GD and Adam (learning rate 0.01), 5 runs each. The second experiment is the LiH molecule at 10 qubits.
- **Theory to check numerically:** with Gaussian variance ∝ 1/L, the paper's Theorem 4.1 lower-bounds the expected squared gradient norm (local observables) by an inverse **polynomial** in depth L and qubit count N. By contrast, uniform init on deep enough circuits gives the usual barren plateau: gradient variance shrinks **exponentially** in N.
- **Implementation:** you write it in PennyLane yourself, which cleanly satisfies the outline's "implement a method" requirement. Avoid any overlap with your own published work.
- **Sample hypothesis:** "For N = 4…14 qubits at fixed depth, the gradient variance under uniform init falls roughly exponentially in N, while under Gaussian init it falls at most polynomially. On the 15-qubit Heisenberg task, Gaussian init reaches the ground energy within 1% in fewer iterations."

**F: GaLore** and **G: LoRA+** are both cheap at RoBERTa-base scale. They pair naturally if you want a "memory-efficient fine-tuning" project: GaLore is the memory story and LoRA+ the speed story.
- Code: `pip install galore-torch` (repo `github.com/jiaweizzhao/GaLore`); `github.com/nikhil-ghosh-berkeley/loraplus`.
- LoRA+ setup in the paper: 3 epochs for MNLI/QQP/QNLI and 10 for SST-2. The best ratio λ = η_B/η_A was about 2⁴ for RoBERTa.

---

## 3c. Third search (Nov 5)

All from accepted venues; all fit in ~6.4 GB of VRAM or run on the CPU. Time figures are my estimates.

| # | Paper | Venue | Syllabus topic | Compute (my estimate) | Statistical axis | Hardware axis | Main caveat |
|---|---|---|---|---|---|---|---|
| **N** | **GPTQ**: post-training quantization for GPT models | ICLR 2023 | Training/deploying large models | Minutes per OPT-125M–1.3B quantization | WikiText2/C4 perplexity, GPTQ vs round-to-nearest (RTN) at 4 and 3 bits | Model memory; quantization time | Repo tested on torch 1.10.1 / transformers 4.21.2; its 3-bit CUDA kernels are "optimized for OPT-175B on A100", so measure memory rather than speed |
| **O** | **Editing Models with Task Arithmetic** (+ optionally **TIES-Merging**, NeurIPS 2023) | ICLR 2023 | Fine-tuning / model editing | ~0.5 h per task to fine-tune CLIP ViT-B/32 if no checkpoints (2,000 iterations, batch 128 per the paper) | Merged-model accuracy on 8 tasks; negation for ViT-B/32 drops target accuracy 48.3 → 24.0 while the ImageNet control goes only 63.4 → 60.9 | Cost of merging vs multi-task training | 8 datasets to download (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN). The TIES repo covers T5/IA3, not the vision setup |
| **P** | **Sparse Autoencoders Find Highly Interpretable Features in LMs** | ICLR 2024 | Interpretability | One run took under 1 h on an A40 in the paper; a few hours on your GPU | Reconstruction loss and sparsity vs dictionary size R ∈ {0.5…8} on Pythia-70M | Training time per dictionary | Headline auto-interp score uses GPT-4 explanations + GPT-3.5 log-probs via the OpenAI API; without it you can only reproduce the reconstruction side |
| **Q** | **Exponentially Many Local Minima in Quantum Neural Networks** | ICML 2021 (spotlight) | Optimization (quantum ML) | Tiny; the paper ran on an i7-7700HQ laptop with 16 GB | Success rate of reaching the global minimum vs number of parameters (4, 8, 16), with Adam / RMSProp / L-BFGS | Weak; only simulation wall-clock | No code mentioned; simulated in PyTorch with an analytic loss |
| **R** | **GPyTorch**: Blackbox Matrix-Matrix GP inference with GPU acceleration | NeurIPS 2018 (spotlight) | Kernels | Minutes–1 h on UCI datasets | Test error / NLL, BBMM vs Cholesky | **Claims up to 20× faster exact GPs than Cholesky (~3K points); up to 15× (SKI) and 4× (SGPR) speedups at up to 500K points** | 2018. The hardware axis is excellent, but it isn't "recent" |
| **S** | **CleanRL**: single-file deep-RL implementations | JMLR 2022 | Reinforcement learning | Hours per environment × seeds (PPO on MuJoCo/classic control, mostly CPU) | Episodic return vs environment steps, vs the published benchmark curves | Steps per second | A software paper ("benchmarked to perform on par with reputable sources"), so you must define the target claim yourself |
| **T** | **Sharpness-Aware Minimization (SAM)** | ICLR 2021 (spotlight) | Optimization | WRN-28-10, 200 epochs, 5 seeds ≈ ~50 GPU-h; reduce to WRN-16-8 / fewer epochs | CIFAR-10 error with basic augmentation: **SAM 2.7% vs SGD 3.5%** (WRN-28-10, 200 epochs) | **Two backward passes per step (~2× cost)** | Paper used 8×V100. The official code is `google-research/sam`; small gains need several seeds |

### Notes on the strongest third-search options
- **R: GPyTorch** has the cleanest *hardware* claim of all 20 papers: a GPU speedup over Cholesky with matching accuracy. It drops in the ranking only because it's from 2018. If the instructor accepts it, it's very low risk.
- **N: GPTQ** is a good LLM-adjacent option. The core algorithm and perplexity evaluation are plain PyTorch, so you can port them to PyTorch ≥ 2.7 and skip the custom kernels. Report memory, not inference speed.
- **Q: QNN local minima** is a second quantum-ML option and could pair with **I**. Both are cheap, but Q's hardware axis is weaker.

---

## 4. How to decide today (proposal due tonight, ≈ 1.5 h total)

1. Install PyTorch with CUDA 12.8+ and run the one-line check in §1. (GPU confirmed: RTX 5070 Laptop, 8 GB.)
2. **Smoke test your top 2 picks** (~30 min each):
   - A: `pip install schedulefree`; train WRN-16-8 or ResNet-18 on CIFAR-10 for 2 epochs and time one epoch.
   - B: `pip install pykan`; run the pykan hello-world example, then fit one toy function at 2 grid sizes.
   - C: clone MeZO; run `medium_models` on SST-2 for 500 steps; record time per step and peak memory.
3. Multiply the time per epoch or step by the planned run count. If the total is more than ~25 GPU-hours before Dec 7, use the reduced scale.
4. The smoke-test numbers become the proposal's **"preliminary work"** section.

---

## 5. Proposal skeleton (reproduction version)

1. **Paper and claim:** which paper, and which 1–2 specific claims or figures you will reproduce.
2. **Why it matters:** 2–3 sentences.
3. **Plan:** setup, reduced scale vs the paper, baseline, seeds.
4. **Preliminary work:** smoke-test timing and memory, plus confirmation that the code runs.
5. **Experiment plan** (½ page): hypothesis · proxy metrics (statistical + hardware) · protocol · expected results.
6. **Contribution:** whether the claim holds at laptop scale, plus a hardware-cost analysis the paper didn't report.
7. **References:** IEEE style.

---

## Sources
- Schedule-Free: [arXiv 2405.15682](https://arxiv.org/abs/2405.15682) · [NeurIPS 2024 oral listing](https://deep-diver.github.io/neurips2024/oral-others/0xenkkenui)
- KAN: [arXiv 2404.19756](https://arxiv.org/abs/2404.19756) · [ICLR 2025 text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2025/kan_kolmogorovarnold_networks__856eb68a.txt)
- MeZO: [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract.html) · [arXiv 2305.17333](https://arxiv.org/html/2305.17333v3) · [project page](https://princeton-nlp.github.io/mezo)
- ZO-Bench: [arXiv 2402.11592](https://arxiv.org/html/2402.11592v3) · [ICML 2024 listing](https://mlanthology.org/icml/2024/zhang2024icml-revisiting) · [code](https://github.com/ZO-Bench/ZO-LLM)
- Grokking: [arXiv 2301.05217](https://arxiv.org/abs/2301.05217) · [ICLR 2023 page](https://iclr.cc/virtual/2023/poster/11385)
- D-Adaptation: [ICML 2023 page](https://icml.cc/virtual/2023/poster/25262) · [ICML 2023 awards](https://icml.cc/Conferences/2023/Awards)
- GaLore: [ICML 2024 oral](https://icml.cc/virtual/2024/oral/35485) · [arXiv 2403.03507](https://arxiv.org/html/2403.03507v2) · [listing](https://mlanthology.org/icml/2024/zhao2024icml-galore)
- LoRA+: [PMLR v235](https://proceedings.mlr.press/v235/hayou24a.html) · [arXiv 2402.12354](https://arxiv.org/pdf/2402.12354)
- DPM-Solver: [NeurIPS 2022](https://neurips.cc/virtual/2022/poster/54266) · [code](https://www.GitHub.com/LuChengTHU/dpm-solver)
- Gaussian init for VQCs: [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7611a3cb5d733e628081431445cb01fd-Abstract.html) · [paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/7611a3cb5d733e628081431445cb01fd-Paper-Conference.pdf)
- Tent: [ICLR 2021](https://iclr.cc/virtual/2021/poster/2874) · [paper text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2021/tent_fully_testtime_adaptation_by_entropy_minimization__7fa37d29.txt)
- Classic GNNs: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b10ed15ff1aa864f1be3a75f1ffc021b-Abstract.html) · code `github.com/LUOyk1999/tunedGNN`
- Data Diet: [NeurIPS 2021](https://neurips.cc/virtual/2021/poster/27634) · [arXiv 2107.07075](https://ar5iv.labs.arxiv.org/html/2107.07075) · [2023 reproduction study](https://arxiv.org/abs/2303.14753)
- SGC: [ICML 2019](https://icml.cc/virtual/2019/poster/3683) · [arXiv 1902.07153](https://arxiv.org/abs/1902.07153v2)
- GPTQ: [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) · [code](https://github.com/ist-daslab/gptq)
- Task Arithmetic: [ICLR 2023](https://www.iclr.cc/virtual/2023/poster/12254) · [paper text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2023/editing_models_with_task_arithmetic__3790a690.txt) · [code](https://github.com/mlfoundations/task_vectors)
- TIES-Merging: [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/69979) · [code](https://github.com/prateeky2806/ties-merging)
- Sparse Autoencoders: [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Abstract-Conference.html) · [arXiv 2309.08600](https://ar5iv.labs.arxiv.org/html/2309.08600)
- QNN local minima: [ICML 2021](https://icml.cc/virtual/2021/spotlight/10074) · [arXiv 2110.02479](https://arxiv.org/pdf/2110.02479)
- GPyTorch: [NeurIPS 2018](https://neurips.cc/virtual/2018/poster/11728) · [arXiv 1809.11165](https://ar5iv.labs.arxiv.org/html/1809.11165)
- CleanRL: [JMLR v23](https://www.jmlr.org/beta/papers/v23/21-1342.html) · [arXiv 2111.08819](https://arxiv.org/pdf/2111.08819)
- SAM: [ICLR 2021](https://iclr.cc/virtual/2021/spotlight/3497) · [arXiv 2010.01412](https://arxiv.org/pdf/2010.01412.pdf)
- FedAvg: [arXiv 1602.05629](https://arxiv.org/abs/1602.05629) · [AISTATS 2017 listing](https://mlanthology.org/aistats/2017/mcmahan2017aistats-communication)
