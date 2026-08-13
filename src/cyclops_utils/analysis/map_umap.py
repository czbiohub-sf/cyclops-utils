"""
UMAP visualization of copairs mAP metric scores.

Shared utility for plotting phenotypic activity/distinctiveness/consistency
metrics on UMAP embeddings. Used by both cyclops_model (interactive notebooks)
and organelle_profiler (batch pipeline with file saving).

Functions
---------
metric_umap
    Visualize mAP metrics on a pre-computed UMAP (interactive, plt.show).
plot_metric_umap
    Compute UMAP from features and visualize mAP metrics (batch, saves to file).
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def metric_umap(
    adata: ad.AnnData,
    metric_map: pd.DataFrame,
    metric_name: str = "activity",
    umap_key: str = "X_umap_cuml",
) -> None:
    """
    Visualize phenotypic metrics on a pre-computed UMAP.

    Creates a 2-panel figure (mAP + -log10(p-value)) with significant points
    colored and non-significant/NTC shown in grey.

    Parameters
    ----------
    adata : AnnData
        AnnData object at perturbation level with UMAP coordinates.
    metric_map : DataFrame
        DataFrame with columns: 'perturbation', 'mean_average_precision',
        '-log10(p-value)', 'below_corrected_p'.
    metric_name : str
        Name of the metric ('activity' or 'distinctiveness') for labeling.
    umap_key : str
        Key in adata.obsm containing UMAP coordinates (default: 'X_umap_cuml').
    """
    if umap_key not in adata.obsm.keys():
        raise ValueError(
            f"UMAP coordinates not found in adata.obsm['{umap_key}']. "
            f"Please compute UMAP first."
        )

    # Map metric values to adata.obs
    metric_dict = metric_map.set_index("perturbation")[
        ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
    ].to_dict("index")

    map_col = f"{metric_name}_mAP"
    log10p_col = f"{metric_name}_log10p"
    significance_col = f"is_{metric_name}"

    adata.obs[map_col] = adata.obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("mean_average_precision", np.nan)
    )
    adata.obs[log10p_col] = adata.obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("-log10(p-value)", np.nan)
    )
    adata.obs[significance_col] = adata.obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("below_corrected_p", False)
    )
    adata.obs["is_NTC"] = adata.obs["perturbation"] == "NTC"

    umap_coords = adata.obsm[umap_key]

    # Masks
    significant_mask = (adata.obs[significance_col] == True) & (~adata.obs["is_NTC"])
    nonsignificant_mask = (adata.obs[significance_col] == False) & (~adata.obs["is_NTC"])
    ntc_mask = adata.obs["is_NTC"]
    grey_mask = nonsignificant_mask | ntc_mask

    # Labels
    if metric_name == "activity":
        title_prefix = "Phenotypic Activity"
        legend_label = "Phenotypically active"
        grey_label = "Not active / NTC"
        summary_label = "Active"
    elif metric_name == "distinctiveness":
        title_prefix = "Phenotypic Distinctiveness"
        legend_label = "Phenotypically distinctive"
        grey_label = "Not distinctive / NTC"
        summary_label = "Distinctive"
    else:
        title_prefix = f"Phenotypic {metric_name.capitalize()}"
        legend_label = f"Phenotypically {metric_name}"
        grey_label = f"Not {metric_name} / NTC"
        summary_label = metric_name.capitalize()

    fig, axes = plt.subplots(2, 1, figsize=(10, 16))

    # Plot 1: Colored by mAP
    ax = axes[0]
    if grey_mask.any():
        ax.scatter(
            umap_coords[grey_mask, 0], umap_coords[grey_mask, 1],
            c="lightgrey", s=20, alpha=0.5, label=grey_label,
        )
    if significant_mask.any():
        scatter1 = ax.scatter(
            umap_coords[significant_mask, 0], umap_coords[significant_mask, 1],
            c=adata.obs.loc[significant_mask, map_col],
            s=30, alpha=0.8, cmap="viridis", label=legend_label,
        )
        plt.colorbar(scatter1, ax=ax, label="Mean Average Precision")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"{title_prefix}: Colored by mAP", fontsize=14, fontweight="bold")
    ax.legend(markerscale=1.5, loc="best")

    # Plot 2: Colored by -log10(p-value)
    ax = axes[1]
    if grey_mask.any():
        ax.scatter(
            umap_coords[grey_mask, 0], umap_coords[grey_mask, 1],
            c="lightgrey", s=20, alpha=0.5, label=grey_label,
        )
    if significant_mask.any():
        scatter2 = ax.scatter(
            umap_coords[significant_mask, 0], umap_coords[significant_mask, 1],
            c=adata.obs.loc[significant_mask, log10p_col],
            s=30, alpha=0.8, cmap="plasma", label=legend_label,
        )
        plt.colorbar(scatter2, ax=ax, label="-log10(p-value)")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"{title_prefix}: Colored by -log10(p-value)", fontsize=14, fontweight="bold"
    )
    ax.legend(markerscale=1.5, loc="best")

    plt.tight_layout()
    plt.show()

    # Print summary
    n_significant = significant_mask.sum()
    n_nonsignificant = nonsignificant_mask.sum()
    n_ntc = ntc_mask.sum()
    print(f"\nUMAP Summary ({metric_name}):")
    print(
        f"  {summary_label} perturbations: {n_significant} "
        f"({100 * n_significant / len(adata):.1f}%)"
    )
    print(
        f"  Non-{metric_name} perturbations: {n_nonsignificant} "
        f"({100 * n_nonsignificant / len(adata):.1f}%)"
    )
    print(f"  NTC: {n_ntc} ({100 * n_ntc / len(adata):.1f}%)")


def plot_metric_umap(
    adata: ad.AnnData,
    metric_map: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    filename: str,
    title: str = "",
    subtitle: str = "",
    save_fn=None,
) -> Optional[Path]:
    """
    Compute UMAP from features and visualize mAP metric scores.

    Batch-oriented version that computes UMAP internally and saves the figure
    to a file. Used by pipeline stages.

    Parameters
    ----------
    adata : AnnData
        Guide- or gene-level AnnData (features in .X, 'perturbation' in .obs).
    metric_map : DataFrame
        Output from copairs mAP functions with columns:
        'perturbation', 'mean_average_precision', 'corrected_p_value',
        'below_corrected_p'.
    metric_name : str
        Short name like "activity", "distinctiveness", "corum", "chad".
    output_dir : Path
        Directory to save figure.
    filename : str
        Output filename (e.g. "cp_ops0094_activity_umap.png").
    title : str
        Main title for the figure.
    subtitle : str
        Subtitle with additional context.
    save_fn : callable, optional
        Custom save function (fig, path) -> Path. If None, uses fig.savefig.

    Returns
    -------
    Path or None
        Path to saved figure, or None if UMAP computation failed.
    """
    import scanpy as sc

    if adata.n_obs < 10:
        logger.warning(f"  Too few observations ({adata.n_obs}) for UMAP, skipping")
        return None

    # Compute UMAP (copy to avoid modifying the original)
    adata_umap = adata.copy()
    try:
        sc.pp.neighbors(adata_umap, n_neighbors=min(15, adata_umap.n_obs - 1), use_rep="X")
        sc.tl.umap(adata_umap)
    except Exception as e:
        logger.warning(f"  UMAP computation failed: {e}")
        return None

    umap_coords = adata_umap.obsm["X_umap"]

    # Add -log10(p-value) to metric_map if missing
    if "-log10(p-value)" not in metric_map.columns:
        metric_map = metric_map.copy()
        metric_map["-log10(p-value)"] = -metric_map["corrected_p_value"].apply(np.log10)

    # Map metric values to adata.obs
    metric_dict = metric_map.set_index("perturbation")[
        ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
    ].to_dict("index")

    obs = adata_umap.obs
    obs["mAP"] = obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("mean_average_precision", np.nan)
    )
    obs["log10p"] = obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("-log10(p-value)", np.nan)
    )
    obs["significant"] = obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("below_corrected_p", False)
    )
    obs["is_NTC"] = obs["perturbation"] == "NTC"

    ntc_mask = obs["is_NTC"].values
    significant_mask = (obs["significant"] == True).values & ~ntc_mask
    nonsig_mask = ~significant_mask & ~ntc_mask

    metric_title = metric_name.replace("_", " ").title()
    n_ntc = ntc_mask.sum()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, color_col, cmap_name, cbar_label, panel_title_suffix in [
        (axes[0], "mAP", "viridis", "Mean Average Precision", "mAP"),
        (axes[1], "log10p", "plasma", "-log10(p-value)", "-log10(p)"),
    ]:
        # Layer 1: non-significant (grey)
        if nonsig_mask.any():
            ax.scatter(
                umap_coords[nonsig_mask, 0], umap_coords[nonsig_mask, 1],
                c="lightgrey", s=20, alpha=0.5, label="Not significant",
            )
        # Layer 2: significant (colored)
        if significant_mask.any():
            sc_plot = ax.scatter(
                umap_coords[significant_mask, 0], umap_coords[significant_mask, 1],
                c=obs.loc[significant_mask, color_col], s=30, alpha=0.8,
                cmap=cmap_name, edgecolors="black", linewidths=0.3,
                label="Significant",
            )
            plt.colorbar(sc_plot, ax=ax, label=cbar_label, shrink=0.8)
        # Layer 3: NTC (red diamonds, on top)
        if ntc_mask.any():
            ax.scatter(
                umap_coords[ntc_mask, 0], umap_coords[ntc_mask, 1],
                c="#E03030", s=50, alpha=0.9, marker="D",
                edgecolors="black", linewidths=0.5,
                label=f"NTC ({n_ntc})", zorder=5,
            )
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(f"{title} — {metric_title}: {panel_title_suffix}",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")

    n_sig = significant_mask.sum()
    n_total = len(obs) - n_ntc
    fig.suptitle(
        f"{metric_title} UMAP — {subtitle}\n"
        f"{n_sig}/{n_total} significant perturbations ({100 * n_sig / max(n_total, 1):.1f}%)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if save_fn is not None:
        path = save_fn(fig, path)
    else:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return path
