"""
Adaptive Gaussian (AG) kernel for Support Vector Machines: Python implementation.

Reference:
    Elen, A., Baş, S., & Közkurt, C. (2022). An Adaptive Gaussian Kernel for
    Support Vector Machine. Arabian Journal for Science and Engineering,
    47, 10579–10588. https://doi.org/10.1007/s13369-022-06654-3

Equations from the paper (Section 4):
    Gaussian kernel  K(u, v)    = exp(−γ‖u−v‖²),  γ = 1/(2ρ²)          (Eq. 5, 15)
    AG kernel        K_AG(u, v) = exp(−(‖u−v‖² − δ + Ω) / δ)            (Eq. 16)
    Offset           Ω = |min(‖u−v‖² − δ)| if min < 0, else 0           (Eq. 17)
    Zero guard       denominator δ + ε                                  (Eq. 18)
    Shorthand        ξ = ‖u−v‖² − δ                                      (Eq. 19)
    Final form       K_AG(u, v) = exp(−(ξ + Ω) / (δ + ε))                (Eq. 20)
    δ(·): standard deviation function (Eq. 16)

Implementation note:
    δ and Ω are computed ONCE in fit(), over all pairs of the training set
    (diagonal included), following the paper's statement that the offset is
    "applied to the entire dataset". K(u, v) therefore depends only on u and v:
    the Gram matrix is symmetric and PSD, and training and prediction use the
    same function.

    Because ‖u−u‖² = 0 on the diagonal, min ξ = −δ and Ω = δ (Eq. 17), so
    Eq. 20 equals the Gaussian kernel (Eq. 15) with γ = 1/(δ+ε). AG's
    contribution is choosing γ automatically from the spread of the squared
    distances instead of by hand. Its bandwidth tracks the scale of the raw
    data, which is where AG shows its largest advantage.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

EPS = np.finfo(float).eps  # ε in Eq. 18 (same as MATLAB eps)


def ag_parameters(X_train):
    """Compute δ (Eq. 16), Ω (Eq. 17) and ε (Eq. 18) from the training data.

    δ = std(‖u−v‖²) over every element of the n×n squared-distance matrix
    (diagonal included, ddof=1). Computed from running sums without building
    the matrix: off-diagonal pairs count twice, the n zero diagonal entries once.

    Returns (delta, omega, epsilon).
    """
    X_train = np.asarray(X_train, dtype=float)
    n = X_train.shape[0]
    d2 = pdist(X_train, "sqeuclidean")              # ‖u−v‖² for i<j
    N = n * n
    s1 = 2.0 * d2.sum()
    s2 = 2.0 * np.square(d2).sum()
    mu = s1 / N
    delta = np.sqrt(max((s2 - N * mu**2) / (N - 1), 0.0))   # Eq. 16: δ(‖u−v‖²)

    xi_min = 0.0 - delta                            # Eq. 19: min ξ (min‖u−v‖² = 0)
    omega = abs(xi_min) if xi_min < 0 else 0.0      # Eq. 17: Ω
    return delta, omega, EPS


def ag_gamma(X_train):
    """γ that AG selects for the Gaussian kernel: γ = 1/(δ+ε) (Eq. 15, 20)."""
    delta, _, epsilon = ag_parameters(X_train)
    return 1.0 / (delta + epsilon)


def ag_kernel(U, V, delta, omega, epsilon=EPS):
    """AG Gram block, Eq. 20. U: m×p, V: n×p → m×n."""
    D2 = cdist(np.asarray(U, float), np.asarray(V, float), "sqeuclidean")  # ‖u−v‖²
    xi = D2 - delta                                     # Eq. 19
    return np.exp(-(xi + omega) / (delta + epsilon))    # Eq. 18 / Eq. 20


class AGSVC(ClassifierMixin, BaseEstimator):
    """SVM classifier with the AG kernel (scikit-learn compatible).

    δ and Ω (Eq. 16–17) are computed in fit() from the training data only, so
    with cross_val_score / GridSearchCV every fold is calibrated on its own
    training split and no test data leaks in.

    Parameters
    ----------
    C : float, SVM penalty parameter (Eq. 2). There is no γ to tune.
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.delta_, self.omega_, self.epsilon_ = ag_parameters(X)   # Eq. 16–18
        self.gamma_ = 1.0 / (self.delta_ + self.epsilon_)
        self.X_train_ = X
        K = ag_kernel(X, X, self.delta_, self.omega_, self.epsilon_)  # Eq. 20
        self.svc_ = SVC(C=self.C, kernel="precomputed").fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def _kernel_to_train(self, X):
        check_is_fitted(self, "svc_")
        return ag_kernel(X, self.X_train_, self.delta_, self.omega_, self.epsilon_)

    def predict(self, X):
        return self.svc_.predict(self._kernel_to_train(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._kernel_to_train(X))


def load_keel(path):
    """Load a KEEL .dat file. Numeric inputs are kept, nominal inputs are
    ordinal-encoded, rows with missing values ('?') are dropped.

    Returns (X, y, feature_names).
    """
    names, rows, in_data = [], [], False
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("@attribute"):
            names.append(line.split()[1])
        elif line.lower().startswith("@data"):
            in_data = True
        elif in_data and "?" not in line:
            rows.append([x.strip() for x in line.split(",")])
    A = np.array(rows, dtype=object)
    X = np.empty((len(A), A.shape[1] - 1))
    for j in range(A.shape[1] - 1):
        try:
            X[:, j] = A[:, j].astype(float)
        except ValueError:
            X[:, j] = np.unique(A[:, j], return_inverse=True)[1]
    return X, A[:, -1].astype(str), names[:-1]


if __name__ == "__main__":
    # Quick sanity check: symmetry, PSD, chunk invariance, equivalence to Eq. 15
    data = Path(__file__).resolve().parent.parent / "data" / "haberman.dat"
    X, _, _ = load_keel(data)
    delta, omega, eps_ = ag_parameters(X)
    G = ag_kernel(X, X, delta, omega, eps_)
    G_gauss = np.exp(-ag_gamma(X) * cdist(X, X, "sqeuclidean"))   # Eq. 15

    print(f"δ = {delta:.4g}, Ω = {omega:.4g}, γ_AG = {ag_gamma(X):.4g}")
    print(f"Asymmetry          : {np.abs(G - G.T).max():.2e}")
    print(f"Smallest eigenvalue: {np.linalg.eigvalsh(G).min():.2e}")
    print(f"Chunk difference   : {abs(ag_kernel(X[:1], X[1:2], delta, omega)[0, 0] - G[0, 1]):.2e}")
    print(f"Diff. to Eq. 15    : {np.abs(G - G_gauss).max():.2e}")
