"""
Z-score normalization and DataFrame↔AnnData conversion utilities.

Shared functions extracted from the CP Challenge stage for reuse across
OPS analysis pipelines (organelle attribution, CP challenge, etc.).

Functions
---------
zscore_normalize
    Z-score normalize feature columns in a DataFrame.
df_to_adata
    Convert a DataFrame to AnnData, dropping zero-variance features.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import List
import logging

logger = logging.getLogger(__name__)


def zscore_normalize(
    guide_df: pd.DataFrame,
    feature_cols: List[str],
    method: str = "global",
    perturbation_col: str = "perturbation",
    ntc_label: str = "NTC",
) -> pd.DataFrame:
    """
    Z-score normalize features.

    Parameters
    ----------
    guide_df : DataFrame
        DataFrame with feature columns and a perturbation column.
    feature_cols : list of str
        Names of numeric feature columns to normalize.
    method : str
        "global" (default) — use all-sample mean/std. Better for
        inter-perturbation comparisons (distinctiveness, consistency).
        "ntc" — use NTC-only mean/std. Centers relative to negative
        control baseline.
    perturbation_col : str
        Column name identifying perturbations. Default "perturbation".
    ntc_label : str
        Label for NTC guides. Default "NTC".

    Returns
    -------
    DataFrame
        Copy of input with feature columns z-score normalized.
    """
    if method == "ntc":
        ntc_mask = guide_df[perturbation_col] == ntc_label
        n_ref = ntc_mask.sum()
        if n_ref < 2:
            logger.warning(f"  Only {n_ref} NTC guides - falling back to global z-score")
            ref_mask = pd.Series(True, index=guide_df.index)
            n_ref = len(guide_df)
            label = "all samples (NTC fallback)"
        else:
            ref_mask = ntc_mask
            label = f"{n_ref} NTC guides"
    else:
        ref_mask = pd.Series(True, index=guide_df.index)
        n_ref = len(guide_df)
        label = f"all {n_ref} samples"

    ref_features = guide_df.loc[ref_mask, feature_cols].values.astype(np.float64)
    means = np.nanmean(ref_features, axis=0)
    stds = np.nanstd(ref_features, axis=0, ddof=1)
    stds[stds == 0] = 1.0  # avoid division by zero

    guide_df = guide_df.copy()
    guide_df[feature_cols] = (
        (guide_df[feature_cols].values.astype(np.float64) - means) / stds
    ).astype(np.float32)

    logger.info(
        f"  Z-score normalized using {label} "
        f"(mean range: {means.min():.2f} to {means.max():.2f})"
    )
    return guide_df


def df_to_adata(
    df: pd.DataFrame, feature_cols: List[str], obs_cols: List[str]
) -> ad.AnnData:
    """
    Convert a DataFrame to AnnData for copairs mAP functions.

    Filters out zero-variance features that add noise to cosine similarity.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with feature and metadata columns.
    feature_cols : list of str
        Names of feature columns for the X matrix.
    obs_cols : list of str
        Names of metadata columns for .obs.

    Returns
    -------
    AnnData
        AnnData with features in .X and metadata in .obs.
    """
    obs = df[[c for c in obs_cols if c in df.columns]].copy()
    obs.index = obs.index.astype(str)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    # Drop zero-variance features
    variances = np.var(X, axis=0)
    keep = variances > 0
    n_dropped = (~keep).sum()
    if n_dropped > 0:
        logger.info(f"  Dropped {n_dropped}/{len(feature_cols)} zero-variance features")
        X = X[:, keep]
        feature_cols = [f for f, k in zip(feature_cols, keep) if k]

    adata = ad.AnnData(
        X=X,
        obs=obs.reset_index(drop=True),
        var=pd.DataFrame(index=feature_cols),
    )
    return adata
