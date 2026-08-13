"""
Phenotypic scoring via copairs mAP (mean Average Precision).

Shared utility functions for computing mAP-based phenotypic activity,
distinctiveness, and consistency metrics across OPS pipelines.

Based on: https://github.com/cytomining/copairs/blob/v0.5.1/examples/phenotypic_activity.ipynb

Functions
---------
adata_to_copairs_df
    Convert AnnData → DataFrame for copairs.
phenotypic_activity_assesment
    Copairs mAP phenotypic activity (guide level).
phenotypic_distinctivness
    Copairs mAP phenotypic distinctiveness among active perturbations.
phenotypic_consistency_corum
    Copairs mAP consistency using CORUM protein complex annotations.
phenotypic_consistency_manual_annotation
    Copairs mAP consistency using CHAD manual gene cluster annotations.
phenotypic_consistency_ebi
    Copairs mAP consistency using EBI Complex Portal complex annotations.
phenotypic_consistency_ontology
    Copairs mAP consistency using ontology super-category groupings.
compute_auc_score
    Significance-weighted mean mAP (threshold-free).
compute_threshold_sweep_auc
    Threshold-sweep AUC integrating active_ratio × mean_mAP.
map_main
    Full phenotypic assessment pipeline (all 4 metrics).
"""

import ast
import yaml
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import logging
from cyclops_utils.paths import BASE_PATH

logger = logging.getLogger(__name__)


def _compute_single_complex_map(
    members,
    label,
    label_col,
    meta_base,
    feats_array,
    null_size=1_000_000,
    distance="cosine",
):
    """
    Compute copairs mAP for a single complex/cluster.

    Designed to be called in parallel via joblib. The feats_array is read-only
    shared memory; only the lightweight meta_c is copied per call.

    Returns (label, complex_map_df) or None on failure.
    """
    from copairs import map as copairs_map

    meta_c = meta_base.copy()
    meta_c["in_complex"] = meta_c["perturbation"].isin(members)
    try:
        results_complex = copairs_map.average_precision(
            meta_c,
            feats_array,
            pos_sameby=["in_complex"],
            pos_diffby=["perturbation"],
            neg_sameby=[],
            neg_diffby=["in_complex"],
            distance=distance,
        )
        map_result = copairs_map.mean_average_precision(
            results_complex,
            sameby=["in_complex"],
            null_size=null_size,
            threshold=0.05,
            seed=0,
        )
        complex_map = map_result[map_result["in_complex"] == True].copy()
        complex_map[label_col] = label
        complex_map.drop(
            columns=["in_complex", "indices"], inplace=True, errors="ignore"
        )
        return complex_map
    except Exception:
        return None


def _compute_single_complex_map_cached(
    members,
    label,
    label_col,
    meta_base,
    sim_matrix,
    perturbations,
    null_size=100_000,
    distance="cosine",
):
    """Compute mAP for a single complex using a precomputed similarity/distance matrix.

    Instead of calling copairs.average_precision (which recomputes pairwise
    distances from scratch for every complex), we rank neighbours from the
    cached matrix and compute AP directly.  This turns each complex evaluation
    from O(N^2 * D) to O(N_in * N * log N) — orders of magnitude faster when
    the matrix is shared across hundreds of complexes.

    For cosine similarity, higher values = more similar (sort descending).
    For euclidean distance, lower values = more similar (sort ascending).

    Returns a single-row DataFrame or None on failure.
    """
    higher_is_closer = distance == "cosine"
    try:
        in_complex = np.array([p in members for p in perturbations])
        n_in = in_complex.sum()
        if n_in < 2:
            return None

        n_total = len(perturbations)
        member_indices = np.where(in_complex)[0]
        pert_arr = np.array(perturbations)

        # --- Compute observed mean AP ---
        ap_scores = []
        for idx in member_indices:
            sims = sim_matrix[idx].copy()
            sims[idx] = -np.inf if higher_is_closer else np.inf  # exclude self

            same_pert = pert_arr == pert_arr[idx]
            pos_mask = in_complex & ~same_pert
            neg_mask = ~in_complex

            n_pos = pos_mask.sum()
            if n_pos == 0 or neg_mask.sum() == 0:
                continue

            relevant = pos_mask | neg_mask
            rel_sims = sims[relevant]
            rel_pos = pos_mask[relevant]

            order = np.argsort(-rel_sims) if higher_is_closer else np.argsort(rel_sims)
            rel_pos_sorted = rel_pos[order]

            cumsum = np.cumsum(rel_pos_sorted)
            precisions = cumsum / np.arange(1, len(cumsum) + 1, dtype=np.float64)
            ap = (precisions * rel_pos_sorted).sum() / n_pos
            ap_scores.append(ap)

        if not ap_scores:
            return None

        mean_ap = float(np.mean(ap_scores))

        # --- Null distribution via copairs mean_average_precision ---
        # Build a minimal copairs-compatible result to get null p-values.
        # This reuses copairs' efficient C-level null sampling without
        # recomputing the distance matrix.
        from copairs import map as copairs_map

        meta_c = meta_base.copy()
        meta_c["in_complex"] = in_complex
        # Construct the AP result DataFrame that mean_average_precision expects.
        # copairs expects: average_precision, normalized_average_precision,
        # n_pos_pairs, n_total_pairs columns.
        meta_c["average_precision"] = np.nan
        meta_c["normalized_average_precision"] = np.nan
        meta_c["n_pos_pairs"] = 0
        meta_c["n_total_pairs"] = 0
        for i, idx in enumerate(member_indices):
            if i < len(ap_scores):
                ap_val = ap_scores[i]
                same_pert = pert_arr == pert_arr[idx]
                n_pos = int((in_complex & ~same_pert).sum())
                n_neg = int((~in_complex).sum())
                meta_c.loc[idx, "average_precision"] = ap_val
                meta_c.loc[idx, "n_pos_pairs"] = n_pos
                meta_c.loc[idx, "n_total_pairs"] = n_pos + n_neg
                # Normalized AP: (AP - expected) / (1 - expected), where
                # expected = n_pos / (n_pos + n_neg)
                expected = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
                if expected < 1.0:
                    meta_c.loc[idx, "normalized_average_precision"] = (
                        ap_val - expected
                    ) / (1.0 - expected)
                else:
                    meta_c.loc[idx, "normalized_average_precision"] = 0.0

        map_result = copairs_map.mean_average_precision(
            meta_c,
            sameby=["in_complex"],
            null_size=null_size,
            threshold=0.05,
            seed=0,
        )
        complex_map = map_result[map_result["in_complex"] == True].copy()
        complex_map[label_col] = label
        complex_map.drop(
            columns=["in_complex", "indices"], inplace=True, errors="ignore"
        )
        return complex_map
    except Exception:
        return None


def adata_to_copairs_df(adata: ad.AnnData) -> pd.DataFrame:
    """
    Convert AnnData to DataFrame format expected by copairs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with features in .X and metadata in .obs

    Returns
    -------
    df : DataFrame
        DataFrame with metadata and feature columns
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    features_df = pd.DataFrame(
        X.astype(np.float32), index=adata.obs_names, columns=adata.var_names
    )

    obs = adata.obs.reset_index(drop=True)
    feats = features_df.reset_index(drop=True)
    return obs, feats


def compute_auc_score(map_df: pd.DataFrame, p_floor: float = 1e-6) -> float:
    """
    Compute significance-weighted mean mAP (AUC score).

    Each gene's contribution = mAP_i * w_i, where w_i is its -log10(p-value)
    normalized to [0, 1]. The overall score is the mean across all genes,
    giving a continuous, threshold-free measure of phenotypic signal strength.

    Parameters
    ----------
    map_df : DataFrame
        Copairs mAP output with 'mean_average_precision' and 'corrected_p_value'.
    p_floor : float
        Minimum p-value clamp to avoid infinite -log10. Default 1e-6.

    Returns
    -------
    float
        Weighted mAP score, approximately bounded [0, 1].
    """
    if len(map_df) < 3:
        logger.warning(f"  AUC score: too few entities ({len(map_df)}), returning NaN")
        return float("nan")

    mAP = map_df["mean_average_precision"].values.astype(np.float64)
    p = np.clip(map_df["corrected_p_value"].values.astype(np.float64), p_floor, 1.0)
    neg_log_p = -np.log10(p)
    w = neg_log_p / -np.log10(p_floor)  # normalize to [0, 1]
    return float(np.nanmean(mAP * w))


def compute_threshold_sweep_auc(
    map_df: pd.DataFrame, n_thresholds: int = 100, p_floor: float = 1e-6
) -> float:
    """
    Compute threshold-sweep AUC: integrate active_ratio * mean_mAP over
    a range of p-value thresholds.

    Parameters
    ----------
    map_df : DataFrame
        Copairs mAP output with 'mean_average_precision' and 'corrected_p_value'.
    n_thresholds : int
        Number of log-spaced thresholds to sweep (1.0 down to p_floor).
    p_floor : float
        Minimum threshold in the sweep. Default 1e-6.

    Returns
    -------
    float
        Integrated AUC score.
    """
    if len(map_df) < 3:
        logger.warning(f"  Sweep AUC: too few entities ({len(map_df)}), returning NaN")
        return float("nan")

    mAP = map_df["mean_average_precision"].values.astype(np.float64)
    p = map_df["corrected_p_value"].values.astype(np.float64)
    n = len(map_df)

    # Log-spaced thresholds from 1.0 down to p_floor
    thresholds = np.logspace(0, np.log10(p_floor), n_thresholds)

    x_vals = -np.log10(thresholds)
    y_vals = np.empty(n_thresholds)

    for i, t in enumerate(thresholds):
        mask = p < t
        n_active = mask.sum()
        if n_active == 0:
            y_vals[i] = 0.0
        else:
            y_vals[i] = (n_active / n) * np.nanmean(mAP[mask])

    # Normalize x to [0, 1] for comparable AUC across experiments
    x_norm = x_vals / x_vals.max()
    return float(np.trapz(y_vals, x_norm))


def phenotypic_activity_assesment(
    adata: ad.AnnData,
    plot_results: bool = True,
    null_size: int = 1_000_000,
    distance: str = "cosine",
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic activity via copairs mAP.

    Positive pairs: same perturbation, different sgRNA.
    Negative pairs: different NTC status.

    Parameters
    ----------
    adata : AnnData
        Guide-level AnnData with 'perturbation', 'sgRNA', 'n_cells' in .obs.
    plot_results : bool
        Whether to display a scatter plot.
    null_size : int
        Number of null samples for p-value estimation (default 1M).
        Use smaller values (e.g. 10_000) for faster approximate ranking.

    Returns
    -------
    activity_map : DataFrame
        Per-perturbation mAP results with 'below_corrected_p' column.
    active_ratio : float
        Fraction of perturbations that are phenotypically active.
    """
    from copairs import map as copairs_map

    obs, feats = adata_to_copairs_df(adata)
    obs["is_NTC"] = obs["perturbation"].apply(lambda x: x == "NTC")
    meta_cols = ["sgRNA", "n_cells", "perturbation", "is_NTC"]
    meta = obs[meta_cols]

    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["is_NTC"],
        distance=distance,
    )
    activity_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"], null_size=null_size, threshold=0.05, seed=0
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    active_ratio = activity_map.below_corrected_p.mean()

    if plot_results:
        plt.scatter(
            data=activity_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic activity assessment")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically active = {100 * active_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return activity_map, active_ratio


def phenotypic_distinctivness(
    adata: ad.AnnData,
    activity_map: Optional[pd.DataFrame] = None,
    plot_results: bool = True,
    null_size: int = 1_000_000,
    active_only: bool = False,
    distance: str = "cosine",
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic distinctiveness among active perturbations.

    Parameters
    ----------
    adata : AnnData
        Guide-level AnnData.
    activity_map : DataFrame, optional
        Output from phenotypic_activity_assesment. Only consulted when
        ``active_only=True`` to filter the distinctiveness test to active
        perturbations. May be None when ``active_only=False``.
    plot_results : bool
        Whether to display a scatter plot.
    null_size : int
        Number of null samples for p-value estimation (default 1M).

    Returns
    -------
    distinctiveness_map : DataFrame
    distinctive_ratio : float
    """
    from copairs import map as copairs_map

    if active_only:
        if activity_map is None:
            raise ValueError(
                "active_only=True requires an activity_map; pass the output of "
                "phenotypic_activity_assesment or set active_only=False."
            )
        active_perturbations = activity_map[activity_map["below_corrected_p"] == True][
            "perturbation"
        ].tolist()
        logger.info(f"Number of active perturbations: {len(active_perturbations)}")
        adata_filtered = adata[
            adata.obs["perturbation"].isin(active_perturbations)
        ].copy()
        logger.info(
            f"Filtered to {adata_filtered.n_obs} observations from {len(active_perturbations)} active perturbations"
        )
    else:
        logger.info(
            "Computing distinctiveness over all perturbations (active_only=False)"
        )
        adata_filtered = adata

    obs, feats = adata_to_copairs_df(adata_filtered)
    meta_cols = ["sgRNA", "n_cells", "perturbation"]
    meta_cols = [col for col in meta_cols if col in obs.columns]
    meta = obs[meta_cols]

    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["perturbation"],
        distance=distance,
    )

    distinctiveness_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"], null_size=null_size, threshold=0.05, seed=0
    )
    distinctiveness_map["-log10(p-value)"] = -distinctiveness_map[
        "corrected_p_value"
    ].apply(np.log10)
    distinctive_ratio = distinctiveness_map.below_corrected_p.mean()
    logger.info(
        f"Proportion of phenotypically distinctive perturbations: {100 * distinctive_ratio:.2f}%"
    )

    if plot_results:
        plt.scatter(
            data=distinctiveness_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic distinctiveness")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * distinctive_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return distinctiveness_map, distinctive_ratio


def phenotypic_consistency_corum(
    adata: ad.AnnData,
    activity_map: Optional[pd.DataFrame] = None,
    plot_results: bool = True,
    null_size: int = 1_000_000,
    cache_similarity: bool = False,
    distance: str = "cosine",
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic consistency using CORUM protein complex annotations.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData.
    activity_map : DataFrame, optional
        Output from phenotypic_activity_assesment. When provided, consistency is
        scored only over perturbations where ``below_corrected_p`` is True. When
        None (default), every perturbation in ``adata`` is used.
    plot_results : bool
        Whether to display a scatter plot.
    null_size : int
        Number of null samples for p-value estimation (default 1M).
    cache_similarity : bool
        When True, precompute the full NxN similarity/distance matrix once and
        reuse it across all complexes.  Much faster (~10-50x) for large gene
        panels but uses more memory (N^2 floats).  Default False for backward
        compatibility.
    distance : str
        Distance metric for copairs: "cosine" or "euclidean". Default "cosine".

    Returns
    -------
    all_complex_results_df : DataFrame
    consistency_corum_ratio : float
    """
    from copairs import map as copairs_map

    path = f"{BASE_PATH}/configs/annotated_gene_panel_July2025.csv"
    gene_panel = pd.read_csv(path)

    if activity_map is not None:
        active_genes = activity_map[activity_map["below_corrected_p"]][
            "perturbation"
        ].tolist()
    else:
        active_genes = adata.obs["perturbation"].unique().tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    obs, feats = adata_to_copairs_df(adata)
    mask = obs["perturbation"].isin(active_genes)
    obs = obs[mask].reset_index(drop=True)
    feats = feats[mask].reset_index(drop=True)

    # Build unique complexes: deduplicate so each complex is computed once,
    # not once per member gene. Key = frozenset of active members.
    active_genes_set = set(active_genes)
    seen_complexes: dict = {}  # frozenset(members) -> representative gene label
    for p in active_genes:
        complex_col = gene_panel.loc[
            gene_panel["Gene.name"] == p, "In_same_complex_with"
        ]
        if complex_col.empty:
            continue
        raw = complex_col.iloc[0]
        raw_list = ast.literal_eval(raw) if isinstance(raw, str) and raw.strip() else []
        members = frozenset(g for g in (raw_list + [p]) if g in active_genes_set)
        if len(members) > 1 and members not in seen_complexes:
            seen_complexes[members] = p  # first gene encountered is the label

    n_complexes = len(seen_complexes)
    logger.info(
        f"  {len(active_genes)} active genes -> {n_complexes} unique CORUM complexes"
    )

    # Precompute base meta + features once
    static_meta_cols = [
        "perturbation",
        "n_cells",
        "guides",
        "reporter",
        "experiment",
        "is_NTC",
        "n_experiments",
    ]
    static_meta_cols = [c for c in static_meta_cols if c in obs.columns]
    meta_base = obs[static_meta_cols]
    feats_array = np.asarray(feats, dtype=np.float32)

    try:
        from cyclops_utils.hpc.resource_manager import get_optimal_workers

        n_workers = get_optimal_workers(
            use_gpu=False, model_ram_gb=4.0, data_ram_gb=8.0, verbose=False
        )
    except Exception:
        n_workers = 4
    n_workers = min(n_workers, n_complexes, 8)

    if cache_similarity:
        perturbations = meta_base["perturbation"].tolist()
        import time as _time

        if distance == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity

            logger.info(
                f"  Precomputing {len(perturbations)}x{len(perturbations)} cosine similarity matrix..."
            )
            _t_sim = _time.time()
            sim_matrix = cosine_similarity(feats_array)
        else:
            from sklearn.metrics.pairwise import euclidean_distances

            logger.info(
                f"  Precomputing {len(perturbations)}x{len(perturbations)} euclidean distance matrix..."
            )
            _t_sim = _time.time()
            sim_matrix = euclidean_distances(feats_array)
        logger.info(f"  Matrix computed in {_time.time() - _t_sim:.1f}s")

        logger.info(
            f"  Using {n_workers} workers for CORUM complex computation (cached {distance})"
        )
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_compute_single_complex_map_cached)(
                members,
                label,
                "complex_id",
                meta_base,
                sim_matrix,
                perturbations,
                null_size=null_size,
                distance=distance,
            )
            for members, label in tqdm(
                seen_complexes.items(), total=n_complexes, desc="CORUM complexes"
            )
        )
    else:
        logger.info(f"  Using {n_workers} workers for CORUM complex computation")
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_compute_single_complex_map)(
                members,
                label,
                "complex_id",
                meta_base,
                feats_array,
                null_size=null_size,
                distance=distance,
            )
            for members, label in tqdm(
                seen_complexes.items(), total=n_complexes, desc="CORUM complexes"
            )
        )
    complex_map_list = [r for r in results if r is not None]

    all_complex_results_df = pd.concat(complex_map_list, ignore_index=True)
    all_complex_results_df["-log10(p-value)"] = -all_complex_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    consistency_corum_ratio = all_complex_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_complex_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (CORUM)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * consistency_corum_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_complex_results_df, consistency_corum_ratio


def phenotypic_consistency_manual_annotation(
    adata: ad.AnnData,
    activity_map: Optional[pd.DataFrame] = None,
    plot_results: bool = True,
    null_size: int = 1_000_000,
    cache_similarity: bool = False,
    distance: str = "cosine",
    annotation_path: str = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic consistency using manual gene cluster annotations.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData.
    activity_map : DataFrame, optional
        Output from phenotypic_activity_assesment. When provided, consistency is
        scored only over perturbations where ``below_corrected_p`` is True. When
        None (default), every perturbation in ``adata`` is used.
    plot_results : bool
        Whether to display a scatter plot.
    null_size : int
        Number of null samples for p-value estimation (default 1M).
    cache_similarity : bool
        When True, precompute the full NxN similarity/distance matrix once and
        reuse it across all clusters.  Default False for backward compatibility.
    distance : str
        Distance metric for copairs: "cosine" or "euclidean". Default "cosine".
    annotation_path : str, optional
        Path to CHAD annotation YAML. Defaults to chad_positive_controls_v4.yml.

    Returns
    -------
    all_complex_results_df : DataFrame
    phenotypic_consistency_ratio : float
    """
    from copairs import map as copairs_map

    path = (
        annotation_path
        or f"{BASE_PATH}/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
    )
    with open(path, "r") as f:
        gene_clusters = yaml.safe_load(f)

    if activity_map is not None:
        active_genes = activity_map[activity_map["below_corrected_p"]][
            "perturbation"
        ].tolist()
    else:
        active_genes = adata.obs["perturbation"].unique().tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    obs, feats = adata_to_copairs_df(adata)
    mask = obs["perturbation"].isin(active_genes)
    obs = obs[mask].reset_index(drop=True)
    feats = feats[mask].reset_index(drop=True)

    # Collect clusters that have >=2 active members
    active_genes_set = set(active_genes)
    # v5 is hierarchical: leaf clusters have `genes`, parent nodes have
    # `components` (child-cluster names). The flat consistency metric uses leaf
    # clusters, so skip entries without `genes` (no-op for the flat v4 file).
    valid_clusters = {
        k: [g for g in v["genes"] if g in active_genes_set]
        for k, v in gene_clusters.items()
        if "genes" in v and len([g for g in v["genes"] if g in active_genes_set]) > 1
    }
    n_clusters = len(valid_clusters)
    logger.info(
        f"  {len(active_genes)} active genes -> {n_clusters} manual clusters to test"
    )

    # Precompute base meta + features once (shared read-only across threads)
    static_meta_cols = [
        "perturbation",
        "n_cells",
        "guides",
        "reporter",
        "experiment",
        "is_NTC",
        "n_experiments",
    ]
    static_meta_cols = [c for c in static_meta_cols if c in obs.columns]
    meta_base = obs[static_meta_cols]
    feats_array = np.asarray(feats, dtype=np.float32)

    try:
        from cyclops_utils.hpc.resource_manager import get_optimal_workers

        n_workers = get_optimal_workers(
            use_gpu=False, model_ram_gb=4.0, data_ram_gb=8.0, verbose=False
        )
    except Exception:
        n_workers = 4
    n_workers = min(n_workers, n_clusters, 8)

    if cache_similarity:
        perturbations = meta_base["perturbation"].tolist()
        import time as _time

        if distance == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity

            logger.info(
                f"  Precomputing {len(perturbations)}x{len(perturbations)} cosine similarity matrix..."
            )
            _t_sim = _time.time()
            sim_matrix = cosine_similarity(feats_array)
        else:
            from sklearn.metrics.pairwise import euclidean_distances

            logger.info(
                f"  Precomputing {len(perturbations)}x{len(perturbations)} euclidean distance matrix..."
            )
            _t_sim = _time.time()
            sim_matrix = euclidean_distances(feats_array)
        logger.info(f"  Matrix computed in {_time.time() - _t_sim:.1f}s")

        logger.info(
            f"  Using {n_workers} workers for manual cluster computation (cached {distance})"
        )
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_compute_single_complex_map_cached)(
                members,
                k,
                "complex_num",
                meta_base,
                sim_matrix,
                perturbations,
                null_size=null_size,
                distance=distance,
            )
            for k, members in tqdm(
                valid_clusters.items(), total=n_clusters, desc="Manual clusters"
            )
        )
    else:
        logger.info(f"  Using {n_workers} workers for manual cluster computation")
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_compute_single_complex_map)(
                members,
                k,
                "complex_num",
                meta_base,
                feats_array,
                null_size=null_size,
                distance=distance,
            )
            for k, members in tqdm(
                valid_clusters.items(), total=n_clusters, desc="Manual clusters"
            )
        )
    complex_map_list = [r for r in results if r is not None]

    all_complex_results_df = pd.concat(complex_map_list, ignore_index=True)
    all_complex_results_df["-log10(p-value)"] = -all_complex_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    phenotypic_consistency_ratio = all_complex_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_complex_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (Manual)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * phenotypic_consistency_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_complex_results_df, phenotypic_consistency_ratio


# Default EBI Complex Portal annotation YAML — mirrors the CHAD default that
# `phenotypic_consistency_manual_annotation` falls back to.
EBI_DEFAULT_ANNOTATION_PATH = (
    f"{BASE_PATH}/configs/gene_clusters/"
    "EBI_complexes_v1_old_gene_names.yaml"
)


def phenotypic_consistency_ebi(
    adata: ad.AnnData,
    activity_map: Optional[pd.DataFrame] = None,
    plot_results: bool = True,
    null_size: int = 1_000_000,
    cache_similarity: bool = False,
    distance: str = "cosine",
    annotation_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic consistency using EBI Complex Portal complex annotations.

    Thin wrapper around :func:`phenotypic_consistency_manual_annotation` that
    defaults ``annotation_path`` to the EBI Complex Portal YAML and rebrands
    log lines / plot title to "EBI". The underlying compute path is identical
    — both reduce to per-complex copairs mAP — so this exists mainly to give
    EBI a first-class API alongside :func:`phenotypic_consistency_corum`.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData.
    activity_map : DataFrame, optional
        Output from phenotypic_activity_assesment. When provided, consistency
        is scored only over perturbations where ``below_corrected_p`` is True.
        When None (default), every perturbation in ``adata`` is used.
    plot_results : bool
        Whether to display an EBI-labeled mAP vs -log10(p) scatter.
    null_size : int
        Number of null samples for p-value estimation (default 1M).
    cache_similarity : bool
        When True, precompute the full NxN similarity matrix once and reuse
        across complexes.
    distance : str
        Distance metric for copairs: "cosine" or "euclidean".
    annotation_path : str, optional
        Path to EBI complex YAML. Defaults to
        :data:`EBI_DEFAULT_ANNOTATION_PATH`.

    Returns
    -------
    ebi_results_df : DataFrame
        Per-complex copairs mAP results.
    phenotypic_consistency_ratio : float
        Fraction of complexes with ``below_corrected_p`` True.
    """
    path = annotation_path or EBI_DEFAULT_ANNOTATION_PATH
    logger.info(f"Running EBI Complex Portal consistency from {path}")

    ebi_results_df, ratio = phenotypic_consistency_manual_annotation(
        adata,
        activity_map=activity_map,
        plot_results=False,
        null_size=null_size,
        cache_similarity=cache_similarity,
        distance=distance,
        annotation_path=path,
    )

    if plot_results:
        plt.scatter(
            data=ebi_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (EBI Complex Portal)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically consistent = {100 * ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return ebi_results_df, ratio


def phenotypic_ebi_plus(
    adata: ad.AnnData,
    plot_results: bool = True,
    null_size: int = 1_000_000,
    distance: str = "cosine",
    annotation_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    EBI+ : guide-level distinctiveness with EBI-complex-expanded groups.

    A single copairs mAP pass on the guide-level AnnData where each guide's
    positive *group* is its gene's EBI Complex Portal complex when the gene
    belongs to one, and the gene itself otherwise. Concretely:

    * pos_sameby = ["ebi_group"], pos_diffby = ["sgRNA"] — a guide's positives
      are all *other* guides sharing its group (its own gene's other guides
      plus, for complex members, every guide of every complex-mate gene).
    * neg_diffby = ["ebi_group"] — negatives are guides of any other group.
    * mAP is aggregated ``sameby=["ebi_group"]`` → one score per group.

    Genes not in any EBI complex reduce *exactly* to
    :func:`phenotypic_distinctivness` (group == gene). Genes sharing a complex
    collapse into one group, so the result has ~(#singleton genes + #complexes)
    rows — every geneKO is covered, but complex members are merged.

    Parameters
    ----------
    adata : AnnData
        Guide-level AnnData with 'perturbation', 'sgRNA', 'n_cells' in .obs.
    plot_results : bool
        Whether to display a mAP vs -log10(p) scatter.
    null_size : int
        Number of null samples for p-value estimation (default 1M).
    distance : str
        Distance metric for copairs: "cosine" or "euclidean".
    annotation_path : str, optional
        Path to EBI complex YAML. Defaults to :data:`EBI_DEFAULT_ANNOTATION_PATH`.

    Returns
    -------
    ebi_plus_map : DataFrame
        Per-group copairs mAP results (one row per ebi_group), with an
        ``ebi_group`` column and ``below_corrected_p``.
    ebi_plus_ratio : float
        Fraction of groups with ``below_corrected_p`` True.
    """
    from copairs import map as copairs_map

    path = annotation_path or EBI_DEFAULT_ANNOTATION_PATH
    logger.info(f"Running EBI+ (guide-level, complex-grouped) from {path}")
    with open(path, "r") as f:
        ebi_complexes = yaml.safe_load(f)

    # gene -> group id. EBI complexes must be DISJOINT: a gene in two complexes
    # is ambiguous (which group owns its guides?), so fail loud rather than
    # silently pick one. Genes with no complex fall back to themselves below.
    gene_to_complexes: Dict[str, List[str]] = {}
    for key, entry in ebi_complexes.items():
        grp = f"EBI_complex_{key}"
        genes = entry.get("genes", []) if isinstance(entry, dict) else []
        for g in genes:
            lst = gene_to_complexes.setdefault(g, [])
            if grp not in lst:
                lst.append(grp)
    multi = {g: cs for g, cs in gene_to_complexes.items() if len(cs) > 1}
    if multi:
        detail = "; ".join(f"{g} in {cs}" for g, cs in sorted(multi.items()))
        raise ValueError(
            f"EBI+ requires disjoint EBI complexes, but {len(multi)} gene(s) "
            f"appear in multiple complexes: {detail}"
        )
    gene_to_group: Dict[str, str] = {g: cs[0] for g, cs in gene_to_complexes.items()}

    obs, feats = adata_to_copairs_df(adata)
    obs["ebi_group"] = obs["perturbation"].map(lambda p: gene_to_group.get(p, p))
    n_groups = obs["ebi_group"].nunique()
    n_complex_groups = obs.loc[
        obs["ebi_group"].str.startswith("EBI_complex_"), "ebi_group"
    ].nunique()
    logger.info(
        f"  {obs['perturbation'].nunique()} perturbations -> {n_groups} EBI+ groups "
        f"({n_complex_groups} complexes + {n_groups - n_complex_groups} singletons)"
    )

    meta_cols = ["sgRNA", "n_cells", "perturbation", "ebi_group"]
    meta_cols = [c for c in meta_cols if c in obs.columns]
    meta = obs[meta_cols]

    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["ebi_group"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["ebi_group"],
        distance=distance,
    )
    ebi_plus_map = copairs_map.mean_average_precision(
        results, sameby=["ebi_group"], null_size=null_size, threshold=0.05, seed=0
    )
    ebi_plus_map["-log10(p-value)"] = -ebi_plus_map["corrected_p_value"].apply(np.log10)
    ebi_plus_ratio = ebi_plus_map.below_corrected_p.mean()
    logger.info(
        f"  Proportion of significant EBI+ groups: {100 * ebi_plus_ratio:.2f}%"
    )

    if plot_results:
        plt.scatter(
            data=ebi_plus_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("EBI+ (guide-level, complex-grouped)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"EBI+ significant = {100 * ebi_plus_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return ebi_plus_map, ebi_plus_ratio


def phenotypic_consistency_ontology(
    adata: ad.AnnData,
    activity_map: Optional[pd.DataFrame] = None,
    gene_to_categories: Optional[Dict[str, List[str]]] = None,
    source_label: str = "ontology",
    plot_results: bool = True,
    null_size: int = 1_000_000,
    min_genes_per_category: int = 3,
    distance: str = "cosine",
) -> Tuple[pd.DataFrame, float]:
    """Compute phenotypic consistency using ontology super-category groupings.

    Like CORUM/CHAD consistency but uses high-level pathway categories
    (CHAD, CHAD-boosted, or Reactome top-level) as the grouping.  Genes
    assigned to the same ontology category should cluster together if the
    feature space captures that biology.

    Always uses a precomputed similarity/distance matrix for speed.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData.
    activity_map : DataFrame, optional
        Output from phenotypic_activity_assesment. When provided, only
        perturbations where ``below_corrected_p`` is True are scored. When
        None (default), every perturbation in ``adata`` is used.
    gene_to_categories : dict
        Mapping of gene name → list of category strings.  Genes may belong
        to multiple categories (multi-mapping, e.g. Reactome). Required.
    source_label : str
        Label for logging/plot titles (e.g. "chad_boosted", "reactome_toplevel").
    plot_results : bool
        Whether to display a scatter plot.
    null_size : int
        Number of null samples for p-value estimation.
    min_genes_per_category : int
        Minimum active genes in a category to include it (default 3).

    Returns
    -------
    all_category_results_df : DataFrame
        Per-category mAP results (one row per category).
    consistency_ratio : float
        Fraction of categories with corrected p < 0.05.
    """
    import time as _time

    if gene_to_categories is None:
        raise ValueError("gene_to_categories is required")
    if activity_map is not None:
        active_genes = activity_map[activity_map["below_corrected_p"]][
            "perturbation"
        ].tolist()
    else:
        active_genes = adata.obs["perturbation"].unique().tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    obs, feats = adata_to_copairs_df(adata)
    mask = obs["perturbation"].isin(active_genes)
    obs = obs[mask].reset_index(drop=True)
    feats = feats[mask].reset_index(drop=True)

    # Build category → active members
    cat_to_genes: Dict[str, List[str]] = {}
    for gene in active_genes:
        cats = gene_to_categories.get(gene, [])
        for cat in cats:
            if cat == "Other":
                continue
            cat_to_genes.setdefault(cat, []).append(gene)

    # Keep only categories with enough members
    valid_categories = {
        cat: sorted(set(members))
        for cat, members in cat_to_genes.items()
        if len(set(members)) >= min_genes_per_category
    }
    n_categories = len(valid_categories)
    logger.info(
        f"  {len(active_genes)} active genes -> {n_categories} ontology categories "
        f"(source={source_label}, min_genes={min_genes_per_category})"
    )

    if n_categories == 0:
        logger.warning("  No valid ontology categories — returning empty results")
        empty = pd.DataFrame(
            columns=[
                "mean_average_precision",
                "corrected_p_value",
                "below_corrected_p",
                "category",
                "-log10(p-value)",
            ]
        )
        return empty, 0.0

    # Precompute base meta + features
    static_meta_cols = [
        "perturbation",
        "n_cells",
        "guides",
        "reporter",
        "experiment",
        "is_NTC",
        "n_experiments",
    ]
    static_meta_cols = [c for c in static_meta_cols if c in obs.columns]
    meta_base = obs[static_meta_cols]
    feats_array = np.asarray(feats, dtype=np.float32)

    # Always use cached similarity/distance matrix for speed
    perturbations = meta_base["perturbation"].tolist()
    if distance == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity

        logger.info(
            f"  Precomputing {len(perturbations)}x{len(perturbations)} cosine similarity matrix..."
        )
        _t_sim = _time.time()
        sim_matrix = cosine_similarity(feats_array)
    else:
        from sklearn.metrics.pairwise import euclidean_distances

        logger.info(
            f"  Precomputing {len(perturbations)}x{len(perturbations)} euclidean distance matrix..."
        )
        _t_sim = _time.time()
        sim_matrix = euclidean_distances(feats_array)
    logger.info(f"  Matrix computed in {_time.time() - _t_sim:.1f}s")

    try:
        from cyclops_utils.hpc.resource_manager import get_optimal_workers

        n_workers = get_optimal_workers(
            use_gpu=False, model_ram_gb=4.0, data_ram_gb=8.0, verbose=False
        )
    except Exception:
        n_workers = 4
    n_workers = min(n_workers, n_categories, 8)

    logger.info(
        f"  Using {n_workers} workers for ontology category computation (cached {distance})"
    )
    results = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(_compute_single_complex_map_cached)(
            members,
            cat_name,
            "category",
            meta_base,
            sim_matrix,
            perturbations,
            null_size=null_size,
            distance=distance,
        )
        for cat_name, members in tqdm(
            valid_categories.items(),
            total=n_categories,
            desc=f"Ontology categories ({source_label})",
        )
    )
    category_map_list = [r for r in results if r is not None]

    if not category_map_list:
        logger.warning("  All ontology categories failed — returning empty results")
        empty = pd.DataFrame(
            columns=[
                "mean_average_precision",
                "corrected_p_value",
                "below_corrected_p",
                "category",
                "-log10(p-value)",
            ]
        )
        return empty, 0.0

    all_category_results_df = pd.concat(category_map_list, ignore_index=True)
    all_category_results_df["-log10(p-value)"] = -all_category_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    consistency_ratio = all_category_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_category_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title(f"Phenotypic consistency ({source_label})")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically consistent = {100 * consistency_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_category_results_df, consistency_ratio


def map_main(
    adata_guide_path: Optional[str],
    adata_guide: Optional[ad.AnnData],
    adata_gene_path: Optional[str],
    adata_gene: Optional[ad.AnnData],
    save_dir: str,
    distance: str = "cosine",
) -> Dict:
    """
    Run full phenotypic assessment pipeline.

    1. Phenotypic activity assessment (guide level)
    2. Phenotypic distinctiveness (guide level)
    3. Phenotypic consistency - CORUM (gene level)
    4. Phenotypic consistency - manual annotation (gene level)

    Parameters
    ----------
    adata_guide_path : str or None
        Path to guide-level h5ad.
    adata_guide : AnnData or None
        Pre-loaded guide-level AnnData.
    adata_gene_path : str or None
        Path to gene-level h5ad.
    adata_gene : AnnData or None
        Pre-loaded gene-level AnnData.
    save_dir : str
        Directory to save results.

    Returns
    -------
    dict with all results and ratios.
    """
    if adata_guide is None and adata_guide_path is None:
        raise ValueError("Either adata_guide or adata_guide_path must be provided")
    if adata_gene is None and adata_gene_path is None:
        raise ValueError("Either adata_gene or adata_gene_path must be provided")

    if adata_guide is None:
        logger.info(f"Loading guide-level AnnData from {adata_guide_path}")
        adata_guide = ad.read_h5ad(adata_guide_path)

    if adata_gene is None:
        logger.info(f"Loading gene-level AnnData from {adata_gene_path}")
        adata_gene = ad.read_h5ad(adata_gene_path)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running phenotypic activity assessment...")
    activity_map, active_ratio = phenotypic_activity_assesment(
        adata_guide,
        plot_results=False,
        distance=distance,
    )
    activity_csv_path = save_path / "phenotypic_activity.csv"
    activity_map.to_csv(activity_csv_path, index=False)
    logger.info(f"Saved activity results to {activity_csv_path}")
    logger.info(f"Active ratio: {100 * active_ratio:.2f}%")

    logger.info("Running phenotypic distinctiveness...")
    distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
        adata_guide,
        activity_map,
        plot_results=False,
        distance=distance,
    )
    distinctiveness_csv_path = save_path / "phenotypic_distinctiveness.csv"
    distinctiveness_map.to_csv(distinctiveness_csv_path, index=False)
    logger.info(f"Saved distinctiveness results to {distinctiveness_csv_path}")
    logger.info(f"Distinctive ratio: {100 * distinctive_ratio:.2f}%")

    logger.info("Running phenotypic consistency (CORUM)...")
    consistency_corum_map, consistency_corum_ratio = phenotypic_consistency_corum(
        adata_gene,
        activity_map,
        plot_results=False,
        distance=distance,
    )
    consistency_corum_csv_path = save_path / "phenotypic_consistency_corum.csv"
    consistency_corum_map.to_csv(consistency_corum_csv_path, index=False)
    logger.info(f"Saved CORUM consistency results to {consistency_corum_csv_path}")
    logger.info(f"CORUM consistency ratio: {100 * consistency_corum_ratio:.2f}%")

    logger.info("Running phenotypic consistency (manual annotation)...")
    consistency_manual_map, consistency_manual_ratio = (
        phenotypic_consistency_manual_annotation(
            adata_gene,
            activity_map,
            plot_results=False,
            distance=distance,
        )
    )
    consistency_manual_csv_path = save_path / "phenotypic_consistency_manual.csv"
    consistency_manual_map.to_csv(consistency_manual_csv_path, index=False)
    logger.info(f"Saved manual consistency results to {consistency_manual_csv_path}")
    logger.info(f"Manual consistency ratio: {100 * consistency_manual_ratio:.2f}%")

    logger.info(f"All results saved to {save_dir}")

    return {
        "activity_map": activity_map,
        "active_ratio": active_ratio,
        "distinctiveness_map": distinctiveness_map,
        "distinctive_ratio": distinctive_ratio,
        "consistency_corum_map": consistency_corum_map,
        "consistency_corum_ratio": consistency_corum_ratio,
        "consistency_manual_map": consistency_manual_map,
        "consistency_manual_ratio": consistency_manual_ratio,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_map_scatter(ax, metric_df, metric_label, ratio, show_ntc=True):
    """Plot p-value vs mAP scatter with density coloring on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    metric_df : pd.DataFrame
        DataFrame with columns ``mean_average_precision``,
        ``corrected_p_value`` (or ``-log10(p-value)``),
        ``below_corrected_p``, and ``perturbation``.
    metric_label : str
        Label shown in the title (e.g. ``"Activity"``).
    ratio : float
        Fraction of significant perturbations (used in title).
    """
    from scipy.stats import gaussian_kde

    if metric_df is None or len(metric_df) == 0:
        ax.text(
            0.5,
            0.5,
            f"{metric_label}\nNo data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return

    if (
        "-log10(p-value)" not in metric_df.columns
        and "corrected_p_value" in metric_df.columns
    ):
        metric_df = metric_df.copy()
        metric_df["-log10(p-value)"] = -np.log10(
            metric_df["corrected_p_value"].clip(lower=1e-300)
        )

    mAP_vals = metric_df["mean_average_precision"].values
    log10p_vals = metric_df["-log10(p-value)"].values
    is_sig = (
        metric_df["below_corrected_p"].values
        if "below_corrected_p" in metric_df.columns
        else np.zeros(len(metric_df), dtype=bool)
    )
    is_ntc = (
        metric_df["perturbation"]
        .str.contains("NTC|non-targeting", case=False, na=False)
        .values
        if "perturbation" in metric_df.columns
        else np.zeros(len(metric_df), dtype=bool)
    )

    nonsig = ~is_sig & ~is_ntc
    sig = is_sig & ~is_ntc

    # Density coloring for significant points
    if sig.sum() > 5:
        xy_sig = np.vstack([mAP_vals[sig], log10p_vals[sig]])
        try:
            kde = gaussian_kde(xy_sig)
            density_sig = kde(xy_sig)
            order = density_sig.argsort()
            sc = ax.scatter(
                mAP_vals[sig][order],
                log10p_vals[sig][order],
                c=density_sig[order],
                s=40,
                alpha=0.8,
                cmap="viridis",
                edgecolors="black",
                linewidths=0.2,
                label=f"Significant ({sig.sum()})",
                zorder=3,
            )
            plt.colorbar(sc, ax=ax, label="Density", shrink=0.7, pad=0.02)
        except Exception:
            ax.scatter(
                mAP_vals[sig],
                log10p_vals[sig],
                c="steelblue",
                s=40,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                label=f"Significant ({sig.sum()})",
            )
    elif sig.any():
        ax.scatter(
            mAP_vals[sig],
            log10p_vals[sig],
            c="steelblue",
            s=40,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
            label=f"Significant ({sig.sum()})",
        )

    if nonsig.any():
        ax.scatter(
            mAP_vals[nonsig],
            log10p_vals[nonsig],
            c="lightgrey",
            s=30,
            alpha=0.4,
            label="Not significant",
            zorder=1,
        )

    if show_ntc and is_ntc.any():
        ax.scatter(
            mAP_vals[is_ntc],
            log10p_vals[is_ntc],
            c="#E03030",
            s=80,
            alpha=0.9,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
            label=f"NTC ({is_ntc.sum()})",
            zorder=5,
        )

    ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(
        ax.get_xlim()[0] + 0.01,
        -np.log10(0.05) + 0.1,
        "p=0.05 (Bonferroni)",
        fontsize=7,
        alpha=0.7,
    )

    ax.set_xlabel("Mean Average Precision (mAP)")
    ax.set_ylabel("-log10(corrected p-value)")
    ax.set_title(f"{metric_label} — {ratio:.1%} significant")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
