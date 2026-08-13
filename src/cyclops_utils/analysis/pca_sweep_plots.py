"""PCA sweep visualization: per-channel curves, summary overlays, peak bar charts."""

import logging
from pathlib import Path
from typing import Dict, List, Optional  # noqa: F401 (Optional used in plot_pca_sweep signature)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_pca_sweep(
    sweep_df, signal, peak_t, peak_n, suptitle,
    plots_dir, file_prefix, min_pcs: int = 10,
    fixed_threshold: Optional[float] = None,
    sweep_peak_t: Optional[float] = None,
    metric_peaks: Optional[Dict] = None,
    best_act_t=None,  # unused, kept for back-compat
    sweep_metric: str = "ratio",
):
    """4-panel sweep plot: Activity, Distinctiveness (all geneKOs), EBI (all geneKOs), #PCs.

    Each metric panel shows its own per-metric peak (colored dashed line).
    If fixed_threshold: orange = fixed choice, red dotted = consensus sweep peak.
    If no fixed_threshold: consensus peak shown in purple on all panels.

    ``sweep_metric``: ``"ratio"`` (default) means the y-axis values are
    fraction-significant percentages; ``"mean_map"`` means they're per-item
    mean mAP (also rendered as a percentage). Y-axis labels switch accordingly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mp = metric_peaks or {}
    peak_act_t = mp.get("peak_act_t")
    peak_dist_t = mp.get("peak_dist_t")
    peak_ebi_t = mp.get("peak_ebi_t")

    sweep_metric = (sweep_metric or "ratio").lower()
    if sweep_metric == "mean_map":
        act_y = "Mean mAP (activity)"
        dist_y = "Mean mAP (distinctiveness)"
        ebi_y = "Mean mAP (EBI consistency)"
        # Mean mAP is already in [0, 1] — don't rescale to percent
        y_scale = 1.0
    else:
        act_y = "% Active Perturbations"
        dist_y = "% Distinctive (all geneKOs)"
        ebi_y = "% EBI consistent (all geneKOs)"
        # Ratios in [0, 1] → display as percent
        y_scale = 100.0

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    ax1, ax2, ax3, ax4 = axes
    ts = sweep_df["threshold"].values
    acts = sweep_df["activity"].values * y_scale
    dist_all = sweep_df["distinctiveness_all"].values * y_scale if "distinctiveness_all" in sweep_df.columns else None
    ebi_all = sweep_df["ebi_all"].values * y_scale if "ebi_all" in sweep_df.columns else None
    pcs = sweep_df["n_pcs"].values

    def _add_selection_markers(ax):
        """Add fixed (orange) or consensus (purple) + consensus reference."""
        if fixed_threshold is not None:
            ax.axvline(fixed_threshold, color="orange", linestyle="--", linewidth=2,
                       alpha=0.9, label=f"Fixed={fixed_threshold:.0%}")
            if sweep_peak_t is not None and sweep_peak_t != fixed_threshold:
                ax.axvline(sweep_peak_t, color="purple", linestyle=":", linewidth=1.5,
                           alpha=0.6, label=f"Consensus={sweep_peak_t:.0%}")
        else:
            if peak_t is not None:
                ax.axvline(peak_t, color="purple", linestyle="--", linewidth=1.5,
                           alpha=0.7, label=f"Consensus={peak_t:.0%}")

    ax1.plot(ts, acts, "o-", color="steelblue", linewidth=2, markersize=6)
    if peak_act_t is not None:
        ax1.axvline(peak_act_t, color="steelblue", linestyle="--", alpha=0.7,
                    label=f"Act peak={peak_act_t:.0%}")
    _add_selection_markers(ax1)
    ax1.set_xlabel("Explained Variance Threshold")
    ax1.set_ylabel(act_y)
    ax1.set_title(f"{signal}: Activity")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if dist_all is not None:
        ax2.plot(ts, dist_all, "o-", color="mediumseagreen", linewidth=2, markersize=6)
        if peak_dist_t is not None:
            ax2.axvline(peak_dist_t, color="mediumseagreen", linestyle="--", alpha=0.7,
                        label=f"Dist peak={peak_dist_t:.0%}")
        _add_selection_markers(ax2)
        ax2.set_ylabel(dist_y)
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Explained Variance Threshold")
    ax2.set_title(f"{signal}: Distinctiveness (all geneKOs)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    if ebi_all is not None:
        ax3.plot(ts, ebi_all, "o-", color="darkorange", linewidth=2, markersize=6)
        if peak_ebi_t is not None:
            ax3.axvline(peak_ebi_t, color="darkorange", linestyle="--", alpha=0.7,
                        label=f"EBI peak={peak_ebi_t:.0%}")
        _add_selection_markers(ax3)
        ax3.set_ylabel(ebi_y)
    else:
        ax3.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Explained Variance Threshold")
    ax3.set_title(f"{signal}: EBI Consistency (all geneKOs)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4.plot(ts, pcs, "o-", color="slategray", linewidth=2, markersize=6)
    ax4.axhline(min_pcs, color="gray", linestyle="--", alpha=0.5, label=f"MIN_PCS={min_pcs}")
    _add_selection_markers(ax4)
    ax4.set_xlabel("Explained Variance Threshold")
    ax4.set_ylabel("Number of PCs")
    ax4.set_title(f"{signal}: PCs vs Threshold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{file_prefix}_sweep.png", dpi=150)
    plt.close(fig)


def plot_sweep_curves_summary(
    per_unit_dir, output_dir, plots_dir, r, a, plt, _logger,
    min_pcs: int = 10,
    fixed_threshold: float | None = None,
):
    """Collect sweep CSVs and generate per-channel sweep, n_pcs, and summary plots.

    ``fixed_threshold`` places the "current operating point" dashed marker. If
    None, we peek at ``output_dir/pca_report.csv`` (peak_threshold column) and
    fall back to 0.80 if that's not readable.
    """
    per_unit_dir = Path(per_unit_dir)
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    sweep_csvs = sorted(per_unit_dir.glob("*_sweep.csv"))
    if not sweep_csvs:
        return

    if fixed_threshold is None:
        report_csv = output_dir / "pca_report.csv"
        if report_csv.exists():
            try:
                _rep = pd.read_csv(report_csv, usecols=["peak_threshold"])
                fixed_threshold = float(_rep["peak_threshold"].median())
            except Exception:
                fixed_threshold = 0.80
        else:
            fixed_threshold = 0.80
    _thr_label = f"{fixed_threshold:.0%} threshold"

    sweep_df = pd.concat([pd.read_csv(f) for f in sweep_csvs], ignore_index=True)
    sweep_df.to_csv(output_dir / "pca_sweep_all_channels.csv", index=False)
    _logger.info(f"  Saved pca_sweep_all_channels.csv ({len(sweep_df)} rows)")

    signals = sorted(sweep_df["signal"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(signals)))
    legend_cols = max(1, len(signals) // 20)

    has_dist = "distinctiveness_all" in sweep_df.columns
    has_ebi = "ebi_all" in sweep_df.columns

    # Plot 1: Per-channel sweep curves (activity + distinctiveness_all + ebi_all)
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    ax1, ax2, ax3 = axes
    for i, sig in enumerate(signals):
        sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
        ax1.plot(sub["threshold"], sub["activity"] * 100, "o-", color=colors[i],
                 linewidth=1.5, markersize=4, label=sig, alpha=0.8)
        if has_dist:
            ax2.plot(sub["threshold"], sub["distinctiveness_all"] * 100, "o-", color=colors[i],
                     linewidth=1.5, markersize=4, label=sig, alpha=0.8)
        if has_ebi:
            ax3.plot(sub["threshold"], sub["ebi_all"] * 100, "o-", color=colors[i],
                     linewidth=1.5, markersize=4, label=sig, alpha=0.8)
    ax1.axhline(r * 100, color="black", linestyle=":", alpha=0.5, label=f"Pooled baseline ({r:.1%})")
    for ax in [ax1, ax2, ax3]:
        ax.axvline(fixed_threshold, color="gray", linestyle="--", alpha=0.3, label=_thr_label)
        ax.set_xlabel("Explained Variance Threshold")
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("% Active Perturbations")
    ax1.set_title("Per-Channel PCA Sweep: Activity")
    ax2.set_ylabel("% Distinctive (all geneKOs)")
    ax2.set_title("Per-Channel PCA Sweep: Distinctiveness (all geneKOs)")
    ax3.set_ylabel("% EBI consistent (all geneKOs)")
    ax3.set_title("Per-Channel PCA Sweep: EBI Consistency (all geneKOs)")
    ax3.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5),
               ncol=legend_cols, borderaxespad=0, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(plots_dir / "per_channel_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/per_channel_sweep.png")

    # Plot 2: N PCs vs threshold
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, sig in enumerate(signals):
        sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
        ax.plot(sub["threshold"], sub["n_pcs"], "o-", color=colors[i],
                linewidth=1.5, markersize=4, label=sig, alpha=0.8)
    ax.axhline(min_pcs, color="red", linestyle="--", alpha=0.5, label=f"MIN_PCS={min_pcs}")
    ax.set_xlabel("Explained Variance Threshold")
    ax.set_ylabel("Number of PCs")
    ax.set_title("PCs Selected vs Variance Threshold (per channel)")
    ax.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=legend_cols, borderaxespad=0, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(plots_dir / "n_pcs_vs_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/n_pcs_vs_threshold.png")

    # Plot 3: Summary sweep with mean curve (activity + distinctiveness_all + ebi_all + n_pcs)
    fig, axes = plt.subplots(1, 4, figsize=(34, 8))
    ax_act, ax_dist, ax_ebi, ax_pcs = axes
    all_thresholds = sorted(sweep_df["threshold"].unique())
    act_matrix, dist_matrix, ebi_matrix, pcs_matrix = [], [], [], []
    for sig in signals:
        sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
        act_matrix.append(np.interp(all_thresholds, sub["threshold"], sub["activity"]))
        pcs_matrix.append(np.interp(all_thresholds, sub["threshold"], sub["n_pcs"]))
        if has_dist:
            dist_matrix.append(np.interp(all_thresholds, sub["threshold"], sub["distinctiveness_all"]))
        if has_ebi:
            ebi_matrix.append(np.interp(all_thresholds, sub["threshold"], sub["ebi_all"]))
    mean_act = np.array(act_matrix).mean(axis=0)
    mean_dist = np.array(dist_matrix).mean(axis=0) if dist_matrix else None
    mean_ebi = np.array(ebi_matrix).mean(axis=0) if ebi_matrix else None
    mean_pcs = np.array(pcs_matrix).mean(axis=0)

    for i, sig in enumerate(signals):
        sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
        ax_act.plot(sub["threshold"], sub["activity"] * 100, "-", color=colors[i],
                    linewidth=0.8, alpha=0.35, label=sig)
        if has_dist:
            ax_dist.plot(sub["threshold"], sub["distinctiveness_all"] * 100, "-", color=colors[i],
                         linewidth=0.8, alpha=0.35, label=sig)
        if has_ebi:
            ax_ebi.plot(sub["threshold"], sub["ebi_all"] * 100, "-", color=colors[i],
                        linewidth=0.8, alpha=0.35, label=sig)
        ax_pcs.plot(sub["threshold"], sub["n_pcs"], "-", color=colors[i],
                    linewidth=0.8, alpha=0.35, label=sig)

    ax_act.plot(all_thresholds, mean_act * 100, "o-", color="black",
                linewidth=2.5, markersize=5, label="Mean", zorder=10)
    if mean_dist is not None:
        ax_dist.plot(all_thresholds, mean_dist * 100, "o-", color="black",
                     linewidth=2.5, markersize=5, label="Mean", zorder=10)
    if mean_ebi is not None:
        ax_ebi.plot(all_thresholds, mean_ebi * 100, "o-", color="black",
                    linewidth=2.5, markersize=5, label="Mean", zorder=10)
    ax_pcs.plot(all_thresholds, mean_pcs, "o-", color="black",
                linewidth=2.5, markersize=5, label="Mean", zorder=10)

    ax_act.axhline(r * 100, color="red", linestyle=":", alpha=0.6, label=f"Pooled baseline ({r:.1%})")
    ax_pcs.axhline(min_pcs, color="red", linestyle="--", alpha=0.5, label=f"MIN_PCS={min_pcs}")
    for ax in axes:
        ax.axvline(fixed_threshold, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Explained Variance Threshold")
        ax.grid(True, alpha=0.3)
    ax_pcs.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  ncol=legend_cols, borderaxespad=0, frameon=True)
    ax_act.set_ylabel("% Active Perturbations")
    ax_act.set_title("Activity vs Threshold")
    ax_dist.set_ylabel("% Distinctive (all geneKOs)")
    ax_dist.set_title("Distinctiveness (all geneKOs) vs Threshold")
    ax_ebi.set_ylabel("% EBI consistent (all geneKOs)")
    ax_ebi.set_title("EBI Consistency (all geneKOs) vs Threshold")
    ax_pcs.set_ylabel("Number of PCs")
    ax_pcs.set_title("PCs vs Threshold")
    fig.suptitle(f"Summary Sweep: {len(signals)} channels, mean shown in black", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(plots_dir / "summary_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/summary_sweep.png")


def plot_metric_map_bar(map_df, metric_name, entity_col, ratio, plots_dir, plt, _logger):
    """Bar chart of per-entity mAP scores for a phenotypic metric (activity, distinctiveness, consistency).

    Args:
        map_df: DataFrame with entity_col, mean_average_precision, and optionally below_corrected_p.
        metric_name: Display name for the metric (e.g. "Activity", "Distinctiveness").
        entity_col: Column name for entity labels (e.g. "perturbation", "complex_id").
        ratio: Fraction significant (used as baseline line label).
        plots_dir: Directory to save the plot.
        plt: matplotlib.pyplot module.
        _logger: Logger instance.
    """
    if map_df is None or len(map_df) == 0:
        return

    plots_dir = Path(plots_dir)
    df = map_df.copy()

    # Exclude NTC from bar chart
    if entity_col in df.columns:
        df = df[~df[entity_col].astype(str).str.contains("NTC|non-targeting", case=False, na=False)]

    if len(df) == 0:
        return

    df = df.sort_values("mean_average_precision", ascending=False).reset_index(drop=True)
    labels = df[entity_col].astype(str)
    values = df["mean_average_precision"].values
    is_sig = df["below_corrected_p"].astype(bool).values if "below_corrected_p" in df.columns else np.ones(len(df), dtype=bool)

    colors = ["steelblue" if s else "#BBBBBB" for s in is_sig]
    bar_width = max(14, len(df) * 0.45)
    fig, ax = plt.subplots(figsize=(bar_width, 6))
    x = np.arange(len(df))
    ax.bar(x, values, color=colors, alpha=0.85)

    if ratio is not None:
        ax.axhline(ratio, color="red", linestyle="--", alpha=0.6, label=f"Ratio significant: {ratio:.1%}")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=max(4, min(8, 200 // max(len(df), 1))))
    ax.set_ylabel(f"{metric_name} mAP")
    ax.set_title(f"{metric_name} — per entity mAP (sorted, blue=significant)")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue", label="Significant"),
        Patch(facecolor="#BBBBBB", label="Not significant"),
    ] + ([plt.Line2D([0], [0], color="red", ls="--", lw=1.5, label=f"Ratio: {ratio:.1%}")] if ratio is not None else []),
        fontsize=8, loc="upper right")

    fig.subplots_adjust(bottom=0.30)
    safe_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"map_{safe_name}_bar.png"
    fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/{fname}")


def plot_metric_violins(
    metric_maps,
    plots_dir,
    plt,
    _logger,
    filename: str = "violin_metric_mAPs.png",
    suptitle: Optional[str] = None,
):
    """Single violin figure showing per-item mAP distributions for each metric.

    Parameters
    ----------
    metric_maps : dict[str, pd.DataFrame]
        ``{display_name: map_df}`` where ``map_df`` has a
        ``mean_average_precision`` column (one row per gene or per complex,
        depending on the metric). Pass any subset of the 5 OPS metrics:
        Activity, Distinctiveness, EBI, CHAD, CORUM. ``None`` values are
        skipped so callers can pass-everything and the plot adapts.
    plots_dir : Path
    suptitle : Optional[str]
        Figure title. Defaults to ``"Per-item mAP distributions"``.

    Each violin's mean mAP is annotated above it; the count of items
    contributing (number of gene / complex rows) is annotated below.
    """
    import numpy as np

    valid = [(name, df) for name, df in metric_maps.items()
             if df is not None and "mean_average_precision" in df.columns and len(df) > 0]
    if not valid:
        _logger.info(f"  Skipping {filename}: no metric maps provided")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(valid) + 2), 6))
    positions = np.arange(1, len(valid) + 1)
    data = [df["mean_average_precision"].dropna().values for _, df in valid]
    means = [float(np.mean(arr)) if len(arr) else float("nan") for arr in data]
    counts = [len(arr) for arr in data]
    names = [name for name, _ in valid]

    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False, widths=0.8)
    # Color per metric (cycle through a balanced palette)
    palette = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(palette[i % len(palette)])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # Box overlay (median line, IQR box, whiskers)
    bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                    boxprops=dict(facecolor="white", edgecolor="black"),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4),
                    showfliers=False)

    # Mean marker (red diamond) ON the violin — annotation goes below x-axis
    for pos, m in zip(positions, means):
        ax.scatter([pos], [m], marker="D", color="red", s=40,
                   zorder=5, edgecolors="black", linewidths=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Per-item mAP", fontsize=12)
    ax.set_title(suptitle or "Per-item mAP distributions", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Mean + n annotations BELOW the x-axis tick labels. Uses an x-in-data /
    # y-in-axes blended transform so positions stay locked to each violin
    # while the y offset is independent of data range.
    trans = ax.get_xaxis_transform()  # x: data, y: axes fraction
    for pos, m, c in zip(positions, means, counts):
        ax.text(pos, -0.11, f"mean={m:.3f}", transform=trans,
                ha="center", va="top",
                fontsize=11, fontweight="bold", color="red")
        ax.text(pos, -0.18, f"n={c}", transform=trans,
                ha="center", va="top",
                fontsize=9, color="#555555")
    # Reserve room below the axes for the two annotation rows.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    out = plots_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/{filename}")
    return out


def plot_channel_peaks_bar(report_rows, r, plots_dir, plt, _logger,
                           dist_ratio=None, corum_ratio=None, chad_ratio=None,
                           filename="per_channel_peaks.png"):
    """Bar chart of per-channel peak metrics (activity, distinctiveness, CORUM, CHAD).

    Aggregate baselines (red dashed lines) are drawn for each metric when provided.
    """
    report_df = pd.DataFrame(report_rows)
    if len(report_df) == 0:
        return
    def _exp_prefix(exp_str):
        if pd.isna(exp_str) or str(exp_str).strip() == "":
            return ""
        exps = [e.split("_")[0] for e in str(exp_str).split(",") if e.strip()]
        if len(exps) > 2:
            return f"{len(exps)}ops_exps"
        return "_".join(exps)

    report_df["bar_label"] = (
        report_df["experiment"].apply(_exp_prefix) + "_" + report_df["signal"].astype(str)
    )

    # Determine which ratio metrics are available (distinctiveness/corum/chad added later)
    ratio_metrics = [
        ("activity",        "% Active",           "steelblue",      r,           True),
        ("distinctiveness", "% Distinctive",       "mediumseagreen", dist_ratio,  True),
        ("corum",           "% CORUM consistent",  "mediumpurple",   corum_ratio, True),
        ("chad",            "% CHAD consistent",   "darkorange",     chad_ratio,  True),
    ]
    ratio_panels = [(col, lbl, col_, base, pct)
                    for col, lbl, col_, base, pct in ratio_metrics
                    if col in report_df.columns and report_df[col].notna().any()]

    n_panels = len(ratio_panels)
    if n_panels == 0:
        return
    fig_h = 6 * n_panels
    fig_w = max(16, len(report_df) * 1.0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    for ax, (col, ylabel, color, baseline, _as_pct) in zip(axes, ratio_panels):
        df_sorted = report_df.sort_values(col, ascending=False).reset_index(drop=True)
        x = np.arange(len(df_sorted))
        vals = df_sorted[col].fillna(0).values * 100
        ax.bar(x, vals, color=color, alpha=0.8)
        if baseline is not None:
            ax.axhline(baseline * 100, color="red", linestyle="--", alpha=0.6,
                       label=f"Pooled baseline ({baseline:.1%})")
            ax.legend(fontsize=8)
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            v = (row[col] if pd.notna(row[col]) else 0) * 100
            ax.text(x[i], v + 0.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(x[i], v + 0.5 + 5.5,
                    f"thr={row['peak_threshold']:.0%}\n{row['n_pcs']}PCs",
                    ha="center", va="bottom", fontsize=6, color="dimgrey")
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted["bar_label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-Reporter {ylabel} — sorted")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(vals.max(), baseline * 100 if baseline else 0) * 1.35)
        ax.margins(x=0.02)

    fig.subplots_adjust(bottom=0.20, hspace=1.1)
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/{filename}")
