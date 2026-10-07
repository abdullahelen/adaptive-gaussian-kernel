"""
Comparison plots: Adaptive Gaussian (AG) vs. Gaussian (RBF) kernel.

Reference:
    Elen, A., Baş, S., & Közkurt, C. (2022). An Adaptive Gaussian Kernel for
    Support Vector Machine. Arabian Journal for Science and Engineering,
    47, 10579–10588. https://doi.org/10.1007/s13369-022-06654-3

Datasets (KEEL, in ../data): Dermatology, Vehicle, Haberman.
Protocol: stratified 10-fold cross-validation (as in Section 5 of the paper),
no hyper-parameter tuning for any kernel (C = 1).

Output (../figures):
    fig1_kernel_curves.png   AG (Eq. 20) vs. fixed-ρ Gaussian (Eq. 5) over
                             growing data ranges; reproduces Fig. 3 of the paper.
    fig2_feature_scales.png  Per-feature standard deviations of the raw data:
                             how unevenly scaled each dataset is.
    fig3_accuracy_raw.png    Accuracy on raw data: AG vs. RBF γ = 1 (paper's
                             setting, Section 5) vs. RBF gamma='scale'.
    fig4_raw_vs_std.png      AG's gain over the best RBF, raw vs. standardized.
    results.md               Numeric results table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ag_kernel import AGSVC, ag_gamma, ag_kernel, ag_parameters, load_keel

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

DATASETS = ["Dermatology", "Vehicle", "Haberman"]
C_AG, C_G05, C_G1, C_SCALE = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"


def load(name):
    return load_keel(DATA / f"{name.lower()}.dat")


# ---------------------------------------------------------------------------
# Figure 1: kernel curves (Fig. 3 of the paper)
# ---------------------------------------------------------------------------
def fig_kernel_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, a in zip(axes.ravel(), [1, 5, 10, 100]):
        x = np.linspace(-a, a, 401).reshape(-1, 1)           # 1-D data on [−a, a]
        delta, omega, eps_ = ag_parameters(x)                 # Eq. 16–18
        k_ag = ag_kernel(x, [[0.0]], delta, omega, eps_).ravel()   # Eq. 20, centre 0
        for rho, c in [(0.5, C_G05), (1.0, C_G1)]:
            ax.plot(x, np.exp(-x.ravel() ** 2 / (2 * rho**2)), color=c, lw=1.4,
                    label=f"Gaussian ρ={rho}")                # Eq. 5
        ax.plot(x, k_ag, color=C_AG, lw=2.4, label="AG (Eq. 20)")
        ax.set_title(f"Data range [−{a}, {a}]   (AG: ρ = {np.sqrt(delta / 2):.3g})")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower center", fontsize=9)
    fig.suptitle("AG adapts its width to the data; a fixed-ρ Gaussian does not\n"
                 "(after Elen et al., 2022, Fig. 3)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_kernel_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: feature scales of the raw data
# ---------------------------------------------------------------------------
def fig_feature_scales():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ratios = {}
    for i, name in enumerate(DATASETS):
        X, _, _ = load(name)
        s = X.std(axis=0)
        s = s[s > 0]
        ratios[name] = s.max() / s.min()
        ax.scatter(np.full(s.size, i) + np.random.default_rng(0).uniform(-0.12, 0.12, s.size),
                   s, s=18, alpha=0.7, color=C_AG)
        ax.text(i, s.max() * 2.2, f"max/min = {ratios[name]:,.0f}×", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    ax.set_ylabel("Feature standard deviation (log)")
    ax.set_title("Spread of raw feature scales", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_feature_scales.png")
    plt.close(fig)
    return ratios


# ---------------------------------------------------------------------------
# Accuracies (stratified 10-fold CV)
# ---------------------------------------------------------------------------
MODELS = {
    "AG": lambda: AGSVC(),
    "RBF γ=1": lambda: SVC(kernel="rbf", gamma=1.0),
    "RBF scale": lambda: SVC(kernel="rbf", gamma="scale"),
}


def evaluate():
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    res = {}
    for name in DATASETS:
        X, y, _ = load(name)
        for setting in ["raw", "standardized"]:
            for m, make in MODELS.items():
                est = make() if setting == "raw" else make_pipeline(StandardScaler(), make())
                res[(name, setting, m)] = cross_val_score(est, X, y, cv=cv).mean()
                print(f"{name:12s} {setting:13s} {m:10s} {res[(name, setting, m)]:.4f}")
    return res


def fig_accuracy_raw(res):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w, idx = 0.26, np.arange(len(DATASETS))
    for k, (m, c) in enumerate(zip(MODELS, [C_AG, C_G1, C_SCALE])):
        bars = ax.bar(idx + (k - 1) * w, [res[(d, "raw", m)] for d in DATASETS], w,
                      color=c, label=m)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=1)
    ax.set_xticks(idx, DATASETS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean accuracy (10-fold CV)")
    ax.set_title("Raw data: AG vs. Gaussian (RBF), no tuning", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_accuracy_raw.png")
    plt.close(fig)


def fig_raw_vs_std(res):
    gain = {s: [100 * (res[(d, s, "AG")] - max(res[(d, s, "RBF γ=1")], res[(d, s, "RBF scale")]))
                for d in DATASETS] for s in ["raw", "standardized"]}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w, idx = 0.36, np.arange(len(DATASETS))
    b1 = ax.bar(idx - w / 2, gain["raw"], w, color=C_AG, label="Raw data")
    b2 = ax.bar(idx + w / 2, gain["standardized"], w, color="#9ecae1", label="Standardized data")
    ax.bar_label(b1, fmt="%+.1f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%+.1f", fontsize=9, padding=2)
    ax.axhline(0, color="black", lw=0.8)
    allg = gain["raw"] + gain["standardized"]
    ax.set_ylim(min(allg + [0]) - 1.5, max(allg) + 1.5)
    ax.set_xticks(idx, DATASETS)
    ax.set_ylabel("AG − best RBF (accuracy points)")
    ax.set_title("AG's advantage is largest on raw, unevenly scaled data", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_raw_vs_std.png")
    plt.close(fig)
    return gain


def write_results(res, ratios, gain):
    L = ["| Dataset | Preprocessing | AG | RBF γ=1 | RBF `scale` | AG − best RBF |",
         "|---|---|---|---|---|---|"]
    for d in DATASETS:
        for i, s in enumerate(["raw", "standardized"]):
            r = [res[(d, s, m)] for m in MODELS]
            L.append(f"| {d} | {s} | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} | {gain[s][DATASETS.index(d)]:+.1f} |")
    L += ["", "| Dataset | n | Features | Classes | Feature-scale ratio (max/min std) | γ_AG (raw) |",
          "|---|---|---|---|---|---|"]
    for d in DATASETS:
        X, y, _ = load(d)
        L.append(f"| {d} | {len(y)} | {X.shape[1]} | {len(set(y))} | {ratios[d]:,.0f}× | {ag_gamma(X):.3g} |")
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    fig_kernel_curves()
    ratios = fig_feature_scales()
    res = evaluate()
    fig_accuracy_raw(res)
    gain = fig_raw_vs_std(res)
    write_results(res, ratios, gain)
    print(f"Figures and results.md written to {OUT}")
