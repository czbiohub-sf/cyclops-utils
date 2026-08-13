"""CHAD positive control cluster loading and embedding overlay plots."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from cyclops_utils.paths import BASE_PATH

logger = logging.getLogger(__name__)

CHAD_V4_PATH = Path(f"{BASE_PATH}/configs/gene_clusters/chad_positive_controls_v4.yml")
SKIP_CLUSTERS = {"NTCs"}
MIN_GENES_PER_CLUSTER = 2


def load_positive_controls(path: Path = CHAD_V4_PATH) -> Dict[str, List[str]]:
    """Load CHAD positive control clusters from YAML. Returns {name: [genes]}."""
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f)
    clusters = {}
    for _id, data in raw.items():
        name = data.get("name", f"cluster_{_id}")
        genes = data.get("genes", [])
        if name in SKIP_CLUSTERS or len(genes) < MIN_GENES_PER_CLUSTER:
            continue
        clusters[name] = genes
    return clusters


def plot_positive_controls_grid(
    embeddings: Dict[str, np.ndarray],
    perts: np.ndarray,
    level_name: str,
    plots_dir: Path,
    plt_mod,
    random_seed: int = 42,
):
    """Generate UMAP/PHATE canvases with CHAD positive control groups highlighted.

    Each canvas is a multi-column grid:
      - First panel: NTC groups highlighted (null baseline)
      - Remaining panels: one per CHAD cluster

    Parameters
    ----------
    embeddings : dict
        {"UMAP": coords, "PHATE": coords} -- 2D arrays of shape (n_obs, 2)
    perts : np.ndarray
        Perturbation labels for each observation
    level_name : str
        "gene" or "guide"
    plots_dir : Path
        Where to save the figures
    plt_mod : module
        matplotlib.pyplot
    random_seed : int
        Seed for NTC subsampling
    """
    import seaborn as sns

    clusters = load_positive_controls()
    if not clusters:
        logger.warning("  Positive controls grid skipped: could not load CHAD v4")
        return

    embed_names = [k for k in ["UMAP", "PHATE"] if k in embeddings]
    if not embed_names:
        return

    # Filter clusters to genes present in data
    all_perts = set(perts)
    filtered = {}
    for name, genes in clusters.items():
        present = [g for g in genes if g in all_perts]
        if len(present) >= MIN_GENES_PER_CLUSTER:
            filtered[name] = present
    if not filtered:
        logger.warning("  Positive controls grid: no clusters found in data")
        return

    # Find NTC indices for the first panel
    is_ntc = np.array([
        str(p).upper().startswith("NTC") or "non-targeting" in str(p).lower()
        for p in perts
    ])
    ntc_indices = np.where(is_ntc)[0]

    n_clusters = len(filtered)
    cluster_colors = sns.color_palette("husl", n_clusters)

    for embed_name in embed_names:
        coords = embeddings[embed_name]
        n_panels = 1 + n_clusters
        n_cols = min(6, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig_width = 4.5 * n_cols
        fig_height = 4 * n_rows
        fig, axes = plt_mod.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        flat_axes = axes.flatten()

        # Panel 0: NTC groups highlighted
        ax0 = flat_axes[0]
        ax0.scatter(
            coords[:, 0], coords[:, 1],
            c="lightgray", s=12, alpha=0.4, rasterized=True,
        )
        if len(ntc_indices) > 0:
            ax0.scatter(
                coords[ntc_indices, 0], coords[ntc_indices, 1],
                c="#E03030", s=60, alpha=0.9, marker="D",
                edgecolors="black", linewidths=0.6, zorder=4,
                label=f"NTC ({len(ntc_indices)})",
            )
            ax0.legend(fontsize=7, loc="best")
        ax0.set_title(f"NTCs ({len(ntc_indices)} groups of 4)", fontsize=9, fontweight="bold")
        ax0.set_xlabel(f"{embed_name} 1", fontsize=8)
        ax0.set_ylabel(f"{embed_name} 2", fontsize=8)
        ax0.tick_params(labelsize=6)

        # Panels 1..N: one per CHAD cluster
        for panel_idx, (cluster_name, genes) in enumerate(filtered.items(), start=1):
            if panel_idx >= len(flat_axes):
                break
            ax = flat_axes[panel_idx]
            gene_mask = np.isin(perts, genes)

            ax.scatter(
                coords[:, 0], coords[:, 1],
                c="lightgray", s=12, alpha=0.4, rasterized=True,
            )

            if gene_mask.any():
                color_idx = panel_idx - 1
                ax.scatter(
                    coords[gene_mask, 0], coords[gene_mask, 1],
                    c=[cluster_colors[color_idx % len(cluster_colors)]],
                    s=70, alpha=0.95,
                    edgecolors="black", linewidths=0.7, rasterized=True, zorder=4,
                )

                if level_name == "gene" and len(genes) <= 15:
                    for gene in genes:
                        g_mask = perts == gene
                        if g_mask.any():
                            idx = np.where(g_mask)[0][0]
                            ax.annotate(
                                gene, coords[idx],
                                fontsize=6, alpha=0.9,
                                xytext=(3, 3), textcoords="offset points",
                            )

            ax.set_title(f"{cluster_name} ({len(genes)}g)", fontsize=8, fontweight="bold")
            ax.set_xlabel(f"{embed_name} 1", fontsize=8)
            ax.set_ylabel(f"{embed_name} 2", fontsize=8)
            ax.tick_params(labelsize=6)

        # Hide empty panels
        for idx in range(n_panels, len(flat_axes)):
            flat_axes[idx].axis("off")

        fig.suptitle(
            f"CHAD Positive Controls — {level_name.title()} {embed_name}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        fname = f"positive_controls_{embed_name.lower()}_{level_name}.png"
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt_mod.close(fig)
        logger.info(f"  Saved plots/{fname}")
