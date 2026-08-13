"""Shared embedding visualization helpers (UMAP / PHATE overlays)."""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_X_for_embedding(adata_level) -> np.ndarray:
    """Clean X matrix for UMAP/PHATE (float32, fill NaN/inf)."""
    X = np.asarray(adata_level.X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def get_perts_col(adata_level) -> np.ndarray:
    """Get perturbation labels from adata obs."""
    obs = adata_level.obs
    pert_col = "perturbation" if "perturbation" in obs.columns else "label_str"
    return obs[pert_col].values


def build_metric_lookup(activity_map) -> Dict:
    """Build perturbation -> metric dict from activity_map for embedding coloring."""
    if activity_map is None:
        return {}
    act_df = activity_map
    if "-log10(p-value)" not in act_df.columns and "corrected_p_value" in act_df.columns:
        act_df = act_df.copy()
        act_df["-log10(p-value)"] = -np.log10(act_df["corrected_p_value"].clip(lower=1e-300))
    return act_df.set_index("perturbation")[
        ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
    ].to_dict("index")


def plot_embedding_overlay(
    coords, perts, metric_lookup, level_name, embed_name,
    plots_dir, n_obs, n_vars, plt,
):
    """Plot 2-panel embedding (mAP viridis + p-value plasma) with NTC red diamonds.

    Returns the filename of the saved plot.
    """
    mAP_vals = np.array([metric_lookup.get(p, {}).get("mean_average_precision", np.nan) for p in perts])
    log10p_vals = np.array([metric_lookup.get(p, {}).get("-log10(p-value)", np.nan) for p in perts])
    is_sig = np.array([metric_lookup.get(p, {}).get("below_corrected_p", False) for p in perts])
    is_ntc = np.array([str(p).upper().startswith("NTC") or "non-targeting" in str(p).lower() for p in perts])

    ntc_mask = is_ntc
    sig_mask = is_sig & ~is_ntc
    nonsig_mask = ~is_sig & ~is_ntc
    n_ntc = ntc_mask.sum()
    n_sig = sig_mask.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, color_vals, cmap_name, cbar_label, panel_title in [
        (ax1, mAP_vals, "viridis", "Mean Average Precision", "mAP"),
        (ax2, log10p_vals, "plasma", "-log10(p-value)", "-log10(p)"),
    ]:
        if nonsig_mask.any():
            ax.scatter(coords[nonsig_mask, 0], coords[nonsig_mask, 1],
                       c="lightgrey", s=40, alpha=0.5, label="Not significant")
        if sig_mask.any():
            sc = ax.scatter(coords[sig_mask, 0], coords[sig_mask, 1],
                            c=color_vals[sig_mask], s=50, alpha=0.8,
                            cmap=cmap_name, edgecolors="black", linewidths=0.3,
                            label=f"Significant ({n_sig})")
            plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
        if ntc_mask.any():
            ax.scatter(coords[ntc_mask, 0], coords[ntc_mask, 1],
                       c="#E03030", s=80, alpha=0.9, marker="D",
                       edgecolors="black", linewidths=0.5,
                       label=f"NTC ({n_ntc})", zorder=5)

        ax.set_xlabel(f"{embed_name} 1", fontsize=11)
        ax.set_ylabel(f"{embed_name} 2", fontsize=11)
        ax.set_title(f"{level_name.title()} — Activity: {panel_title}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{level_name.title()}-Level {embed_name} — {n_obs} obs, {n_vars} features\n"
        f"{n_sig}/{n_obs - n_ntc} significant ({100 * n_sig / max(n_obs - n_ntc, 1):.1f}%)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{embed_name.lower()}_{level_name}.png"
    fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname
