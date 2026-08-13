"""Discover embedding processing status across experiments.

Scans the experiment base directory for all opsXXXX_YYYYMMDD experiments,
filters out bad/excluded experiments, and classifies each as:

  - Fully processed   : CSV(s) + complete anndata_objects/ present
  - Embeddings only   : CSV(s) present but anndata incomplete or missing
  - No embeddings     : Embedding directory or CSV(s) absent
  - No config         : No embedding config found for this experiment

Usage
-----
    ops-embedding-status --model cell_dino
    ops-embedding-status --model dinov3
    ops-embedding-status -m subcell
    ops-embedding-status -m cell_profiler
    ops-embedding-status --reporter MAP1LC3B
    ops-embedding-status -r SQSTM1
    ops-embedding-status --list-reporters
"""

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

from cyclops_utils.data.bad_experiments import is_excluded
from cyclops_utils.paths import BASE_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENT_BASE = Path(f"{BASE_PATH}")

# Monorepo root: cyclops-monorepo/cyclops_utils/src/cyclops_utils/analysis/embedding_discovery.py
#   parents[0] = analysis/
#   parents[1] = cyclops_utils/  (package)
#   parents[2] = src/
#   parents[3] = cyclops_utils/  (project dir)
#   parents[4] = cyclops-monorepo/
MONOREPO_ROOT = Path(__file__).parents[4]
CONFIGS_BASE = MONOREPO_ROOT / "experiments" / "embedding" / "configs"

EXP_PATTERN = re.compile(r"^ops\d{4}_\d{8}$")

ALL_EXCLUDE_CATEGORIES = (
    "bad",
    "iss_only",
    "do_not_run",
    "non_standard",
    "positive_control",
    "need_rescue",
)

# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    config_dir: str  # subdir under CONFIGS_BASE
    embed_subdir: str  # subdir under {exp}/3-assembly/
    csv_stem: str  # CSV filename prefix: f"{csv_stem}_{channel}.csv"
    single_csv: bool  # True → check for "{csv_stem}.csv" (no channel suffix)
    check_per_channel_anndata: bool  # False → just check anndata_objects/ non-empty


MODEL_SPECS: dict[str, ModelSpec] = {
    "cell_profiler": ModelSpec(
        config_dir="cell-profiler",
        embed_subdir="cell-profiler",
        csv_stem="cp_features",
        single_csv=True,
        check_per_channel_anndata=False,
    ),
    "dinov3": ModelSpec(
        config_dir="dinov3",
        embed_subdir="dino_features",
        csv_stem="dinov3_features",
        single_csv=False,
        check_per_channel_anndata=True,
    ),
    "cell_dino": ModelSpec(
        config_dir="cell_dino",
        embed_subdir="cell_dino_features",
        csv_stem="cell_dino_features",
        single_csv=False,
        check_per_channel_anndata=True,
    ),
    "subcell": ModelSpec(
        config_dir="subcell",
        embed_subdir="subcell_features",
        csv_stem="subcell_features",
        single_csv=False,
        check_per_channel_anndata=True,
    ),
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_model_configs(model: str) -> dict[str, list[str]]:
    """Return {experiment_name: [channel, ...]} from all configs for a model.

    If multiple configs exist for the same experiment (e.g. re-runs with
    different dates), the last one alphabetically wins.
    """
    spec = MODEL_SPECS[model]
    config_dir = CONFIGS_BASE / spec.config_dir
    result: dict[str, list[str]] = {}

    for cfg_path in sorted(config_dir.glob("*.yml")):
        try:
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            dm = data.get("data_manager", {}) or {}
            experiments = dm.get("experiments", {}) or {}
            channels = dm.get("out_channels", []) or []
            for exp_name in experiments:
                result[exp_name] = [str(ch) for ch in channels]
        except Exception:
            continue

    return result


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------


def _discover_experiments() -> list[str]:
    """Return sorted list of opsXXXX_YYYYMMDD directory names in EXPERIMENT_BASE."""
    return sorted(
        d.name
        for d in EXPERIMENT_BASE.iterdir()
        if d.is_dir() and EXP_PATTERN.match(d.name)
    )


# ---------------------------------------------------------------------------
# Status checking
# ---------------------------------------------------------------------------


def _h5ad_exists(adata_dir: Path, stem: str, channel: str) -> bool:
    """Check for an anndata file, accepting both current and legacy naming.

    Standard:  {stem}_{channel}.h5ad            (e.g. features_processed_GFP.h5ad)
    Legacy:    {stem}_features_{channel}.h5ad   (e.g. features_processed_features_GFP.h5ad)
    """
    return (adata_dir / f"{stem}_{channel}.h5ad").exists() or (
        adata_dir / f"{stem}_features_{channel}.h5ad"
    ).exists()


def _check_status(exp: str, spec: ModelSpec, channels: list[str]) -> tuple[bool, bool]:
    """Return (has_embeddings, has_anndata) for an experiment.

    has_embeddings : all expected CSV files are present
    has_anndata    : anndata_objects/ exists and is complete
    """
    embed_dir = EXPERIMENT_BASE / exp / "3-assembly" / spec.embed_subdir

    if not embed_dir.exists():
        return False, False

    # --- CSV check ---
    if spec.single_csv:
        csv_ok = (embed_dir / f"{spec.csv_stem}.csv").exists()
    else:
        csv_ok = all(
            (embed_dir / f"{spec.csv_stem}_{ch}.csv").exists() for ch in channels
        )

    if not csv_ok:
        return False, False

    # --- Anndata check ---
    adata_dir = embed_dir / "anndata_objects"
    if not adata_dir.exists():
        return True, False

    if not spec.check_per_channel_anndata:
        anndata_ok = any(adata_dir.glob("*.h5ad"))
    else:
        from cyclops_utils.data.feature_metadata import FeatureMetadata

        meta = FeatureMetadata()
        reporters = [meta.get_biological_signal(exp, ch) for ch in channels]
        anndata_ok = all(
            _h5ad_exists(adata_dir, "features_processed", reporter)
            and _h5ad_exists(adata_dir, "gene_bulked", reporter)
            and _h5ad_exists(adata_dir, "guide_bulked", reporter)
            for reporter in reporters
        )

    return True, anndata_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EXP_NAME_WIDTH = 15  # len("opsXXXX_YYYYMMDD")
_COL_PADDING = 2


def _print_grid(experiments: list[str]) -> None:
    """Print experiments in a column-major grid sized to the terminal width."""
    term_width = shutil.get_terminal_size().columns
    col_width = _EXP_NAME_WIDTH + _COL_PADDING
    n_cols = max(1, (term_width - _COL_PADDING) // col_width)
    n_rows = math.ceil(len(experiments) / n_cols)
    for row in range(n_rows):
        items = [
            experiments[col * n_rows + row]
            for col in range(n_cols)
            if col * n_rows + row < len(experiments)
        ]
        click.echo("  " + "  ".join(item.ljust(_EXP_NAME_WIDTH) for item in items))


def _print_group(label: str, experiments: list[str]) -> None:
    click.echo(f"{label} ({len(experiments)}):")
    if experiments:
        _print_grid(experiments)
    else:
        click.echo("  (none)")
    click.echo()


def _run_reporter_mode(reporter: str, experiments: list[str]) -> None:
    """Print embedding status for all models, filtered to experiments with REPORTER."""
    from cyclops_utils.data.feature_metadata import FeatureMetadata

    meta = FeatureMetadata()

    click.echo(f"\nReporter: {reporter}\n")

    for model, spec in MODEL_SPECS.items():
        config_channels = _load_model_configs(model)

        fully_processed: list[str] = []
        embeddings_only: list[str] = []
        no_embeddings: list[str] = []
        no_config: list[str] = []

        for exp in experiments:
            channels = config_channels.get(exp)
            if channels is None:
                no_config.append(exp)
                continue

            reporters = [meta.get_biological_signal(exp, ch) for ch in channels]
            if reporter not in reporters:
                continue

            has_emb, has_adata = _check_status(exp, spec, channels)
            if has_emb and has_adata:
                fully_processed.append(exp)
            elif has_emb:
                embeddings_only.append(exp)
            else:
                no_embeddings.append(exp)

        click.echo(f"Model: {model}")
        if not any([fully_processed, embeddings_only, no_embeddings, no_config]):
            click.echo(f"  (no {model} not yet run for {reporter})\n")
            continue

        _print_group("  Fully processed", fully_processed)
        _print_group("  Embeddings but no anndata", embeddings_only)
        _print_group("  No embeddings", no_embeddings)
        _print_group("  No config", no_config)


@click.command()
@click.option(
    "--model",
    "-m",
    default=None,
    type=click.Choice(list(MODEL_SPECS.keys())),
    help="Embedding model to check (mutually exclusive with --reporter).",
)
@click.option(
    "--reporter",
    "-r",
    default=None,
    help="Biological signal reporter to check across all models (mutually exclusive with --model).",
)
@click.option(
    "--list-reporters",
    is_flag=True,
    default=False,
    help="List all known biological signal reporters across all models and experiments.",
)
def main(model: str | None, reporter: str | None, list_reporters: bool) -> None:
    """Show embedding processing status across all experiments.

    Specify either --model to see status for one model across all experiments,
    or --reporter to see status for one biological signal reporter across all models.

    Scans $OPS_BASE_PATH for opsXXXX_YYYYMMDD directories,
    filters bad/excluded experiments, then classifies each by embedding status.
    """
    if list_reporters:
        from cyclops_utils.data.feature_metadata import FeatureMetadata

        meta = FeatureMetadata()
        seen: set[str] = set()
        for model_name in MODEL_SPECS:
            for exp, channels in _load_model_configs(model_name).items():
                for ch in channels:
                    try:
                        seen.add(meta.get_biological_signal(exp, ch))
                    except Exception:
                        pass
        click.echo("\nKnown reporters:\n")
        click.echo("  " + "  ".join(sorted(seen)))
        click.echo()
        return

    if model and reporter:
        raise click.UsageError("--model and --reporter are mutually exclusive.")
    if not model and not reporter:
        raise click.UsageError("One of --model or --reporter is required.")

    experiments = [
        exp
        for exp in _discover_experiments()
        if not is_excluded(exp, categories=ALL_EXCLUDE_CATEGORIES, date_cutoff=True)
    ]

    if reporter:
        _run_reporter_mode(reporter, experiments)
        return

    spec = MODEL_SPECS[model]
    config_channels = _load_model_configs(model)

    fully_processed: list[str] = []
    embeddings_only: list[str] = []
    no_embeddings: list[str] = []
    no_config: list[str] = []

    for exp in experiments:
        channels = config_channels.get(exp)
        if channels is None:
            no_config.append(exp)
            continue

        has_emb, has_adata = _check_status(exp, spec, channels)

        if has_emb and has_adata:
            fully_processed.append(exp)
        elif has_emb:
            embeddings_only.append(exp)
        else:
            no_embeddings.append(exp)

    click.echo(f"\nModel: {model}\n")
    _print_group("Fully processed", fully_processed)
    _print_group("Embeddings but no anndata", embeddings_only)
    _print_group("No embeddings", no_embeddings)
    _print_group("No config", no_config)
    click.echo(f"Total (non-excluded): {len(experiments)}")


if __name__ == "__main__":
    main()
