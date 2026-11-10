"""
Make the proposal figure from your own sweep results.
Run from the Project\\proposal folder:   python make_figures.py
Reads ..\\noise_depth_sweep_FakeManilaV2_5q_scale2.csv (paper angle convention) and writes
fig_noise_sweep.pdf (vector, for LaTeX) and .png.
"""
CSV_NAME = "noise_depth_sweep_FakeManilaV2_5q_scale2.csv"
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).resolve().parent
df = pd.read_csv(here.parent / CSV_NAME).sort_values("blocks")
b = df["blocks"]

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))

# (a) squared gradient norm at initialization
ax1.semilogy(b, df["gauss_off"], "o-", color="C0", label="Gaussian, noiseless")
ax1.semilogy(b, df["gauss_on"], "o--", color="C0", mfc="white", label="Gaussian, noisy sim.")
ax1.semilogy(b, df["unif_off"], "s-", color="C3", label="Uniform, noiseless")
ax1.semilogy(b, df["unif_on"], "s--", color="C3", mfc="white", label="Uniform, noisy sim.")
ax1.set_xlabel("Circuit depth (layers)")
ax1.set_ylabel("Squared gradient norm")
ax1.set_title("(a) Squared gradient norm at initialization")
ax1.set_ylim(0.8, 3000)  # headroom so the legend sits above the data
ax1.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
ax1.grid(alpha=0.3, which="both")

# (b) Gaussian / uniform ratio
ax2.plot(b, df["gauss_off"] / df["unif_off"], "o-", color="k", label="noiseless")
ax2.plot(b, df["gauss_on"] / df["unif_on"], "o--", color="k", mfc="white", label="noisy hardware sim.")
ax2.axhline(1, color="grey", lw=0.8)
ax2.set_ylim(0, 25)
for x, y in zip(b, df["gauss_on"] / df["unif_on"]):
    ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
ax2.set_xlabel("Circuit depth (layers)")
ax2.set_ylabel("Gaussian / uniform ratio")
ax2.set_title("(b) Gaussian-to-uniform ratio")
ax2.legend(frameon=False, fontsize=7)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(here / "fig_noise_sweep.pdf")
fig.savefig(here / "fig_noise_sweep.png", dpi=200)
print("Saved fig_noise_sweep.pdf and fig_noise_sweep.png")
