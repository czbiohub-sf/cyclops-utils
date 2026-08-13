"""PCA fitting and variance-threshold utilities."""

import numpy as np


def fit_pca(X: np.ndarray, max_pcs: int = 500):
    """Fit PCA on a feature matrix.

    Returns (X_transformed, cumulative_variance, pca_model).

    Solver selection:
    - n_samples >> n_features (tall matrix, e.g. 5M cells × 384 features):
        uses 'covariance_eigh' — computes X.T @ X (n_features × n_features) then
        eigendecomposes. Avoids materializing the U matrix, cutting peak memory
        from ~150 GB to ~30 GB for 5M × 384 float64.
    - otherwise: sklearn default ('full' or 'randomized' via 'auto').
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_pcs = min(X.shape[0] - 1, X.shape[1] - 1, max_pcs)

    # For tall matrices use covariance_eigh to avoid O(n_samples²) SVD memory
    if X.shape[0] > 10 * X.shape[1]:
        svd_solver = "covariance_eigh"
    else:
        svd_solver = "auto"

    pca = PCA(n_components=n_pcs, svd_solver=svd_solver)
    X_pcs = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return X_pcs, cumvar, pca


def n_pcs_for_threshold(cumvar: np.ndarray, threshold: float) -> int:
    """Return the number of PCs needed to reach a cumulative variance threshold."""
    n = int(np.searchsorted(cumvar, threshold) + 1)
    return min(n, len(cumvar))
