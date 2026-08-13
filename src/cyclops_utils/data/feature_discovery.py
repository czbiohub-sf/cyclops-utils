"""Shared experiment discovery and channel resolution utilities.

Provides functions for:
- Resolving h5ad channel names to biological signal labels via FeatureMetadata
- Discovering experiments with DINO or CellProfiler feature files
- Grouping (experiment, channel) pairs by biological signal
- Loading attribution config and resolving storage roots

These are shared between cyclops_model (pca_optimization) and organelle_profiler
(organelle_attribution_stage) to avoid code duplication.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from cyclops_utils.paths import BASE_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_MAPS_PATH = Path(f"{BASE_PATH}/configs/ops_channel_maps.yaml")
CHANNEL_MAPS_PATH_FALLBACK = Path(
    f"{BASE_PATH}/configs/ops_channel_maps.yaml"
)

DEFAULT_STORAGE_ROOTS = [
    Path(f"{BASE_PATH}"),
]

# Sibling feature dirs live alongside the standard feature dir, e.g.
#   3-assembly/dino_features/        ← standard
#   3-assembly/dino_features_4i/     ← 4i (iterative-IF) sibling
#   3-assembly/dino_features_cp/     ← Cell Painting sibling (new layout)
# Channels discovered from a sibling dir are tagged with the sibling's tag so
# downstream loaders know which dir to read from. The tag is stripped before
# any filename construction or channel-map lookup.
#
# Each entry: tag → (feature_dir suffix, cell-profiler dir name, label suffix).
# label_suffix is appended to the resolved biological signal so sibling
# variants of the same marker form their own signal groups.
SIBLING_TAGS: Dict[str, Dict[str, str]] = {
    "4i:": {"feature_suffix": "_4i", "cp_dir": "cell-profiler-4i", "label_suffix": " (4i)"},
    "cp:": {"feature_suffix": "_cp", "cp_dir": "cell-profiler-cp", "label_suffix": " (cp)"},
}

# Back-compat aliases (4i is the original sibling).
FOUR_I_TAG = "4i:"


def _sibling_tag_for(channel: str) -> Optional[str]:
    for tag in SIBLING_TAGS:
        if channel.startswith(tag):
            return tag
    return None


def _strip_sibling_tag(channel: str) -> str:
    tag = _sibling_tag_for(channel)
    return channel[len(tag):] if tag else channel


def _is_4i_channel(channel: str) -> bool:
    return channel.startswith(FOUR_I_TAG)


def _strip_4i(channel: str) -> str:
    return _strip_sibling_tag(channel)


def _resolve_feature_dir(channel: str, feature_dir: str) -> Tuple[str, str]:
    """Return (effective_feature_dir, untagged_channel) given a possibly-tagged channel.

    Handles both DINO-style (e.g. ``dino_features`` + ``_4i`` → ``dino_features_4i``)
    and CellProfiler-style (``cell-profiler`` → ``cell-profiler-4i``) feature dirs.
    """
    tag = _sibling_tag_for(channel)
    if tag is None:
        return feature_dir, channel
    cfg = SIBLING_TAGS[tag]
    raw_channel = channel[len(tag):]
    if feature_dir == "cell-profiler":
        return cfg["cp_dir"], raw_channel
    return f"{feature_dir}{cfg['feature_suffix']}", raw_channel

# Default config path — resolve relative to the monorepo root if available,
# otherwise fall back to the organelle_profiler package location.
def _find_default_config() -> Path:
    """Search common locations for organelle_attribution_config.yaml."""
    candidates = [
        Path(__file__).parents[5] / "organelle_profiler" / "configs" / "organelle_attribution_config.yaml",
        Path(f"{BASE_PATH}/configs/organelle_attribution_config.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # return first candidate even if missing


DEFAULT_CONFIG_PATH = _find_default_config()


def get_channel_maps_path() -> str:
    """Return the best available channel maps YAML path as a string."""
    if CHANNEL_MAPS_PATH.exists():
        return str(CHANNEL_MAPS_PATH)
    return str(CHANNEL_MAPS_PATH_FALLBACK)


def load_attribution_config(config_path: Optional[Path] = None) -> dict:
    """Load the attribution YAML config, returning an empty dict on missing file."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config not found: {config_path}, using defaults")
    return {}


def get_storage_roots(config: Optional[dict] = None) -> List[Path]:
    """Resolve storage roots from config dict or use defaults."""
    config = config or {}
    return [
        Path(p)
        for p in config.get("storage_roots", [str(p) for p in DEFAULT_STORAGE_ROOTS])
    ]


# ---------------------------------------------------------------------------
# Experiment directory & cell data loading
# ---------------------------------------------------------------------------

def find_experiment_dir(
    experiment: str,
    storage_roots: List[Path],
) -> Optional[Path]:
    """Locate an experiment directory across storage roots.

    Tries an exact match first, then falls back to a glob on the short
    experiment key (e.g. ``ops0046``).

    Parameters
    ----------
    experiment : str
        Full experiment name (e.g. ``ops0046_20250501``).
    storage_roots : list of Path
        Root directories to search.

    Returns
    -------
    Path or None
        The first matching experiment directory, or *None* if not found.
    """
    from cyclops_utils.data.filesystem import extract_ops_key

    exp_short = extract_ops_key(experiment) or experiment.split("_")[0]

    for root in storage_roots:
        candidate = root / experiment
        if candidate.exists():
            return candidate
        matches = sorted(root.glob(f"{exp_short}*"))
        if matches:
            return matches[0]
    return None


def load_cell_h5ad(
    experiment: str,
    channel: str,
    storage_roots: List[Path],
    feature_dir: str,
    metadata_path: str,
):
    """Load a cell-level h5ad for a single experiment/channel pair.

    Resolves the biological signal via :class:`FeatureMetadata` and searches
    storage roots for the corresponding ``features_processed_*.h5ad`` file.

    Parameters
    ----------
    experiment : str
        Full experiment name.
    channel : str
        Channel identifier (microscope channel or reporter name).
    storage_roots : list of Path
        Directories to search for the experiment.
    feature_dir : str
        Subdirectory under ``3-assembly/`` (e.g. ``"dino_features"``).
    metadata_path : str
        Path to the channel-maps YAML for :class:`FeatureMetadata`.

    Returns
    -------
    anndata.AnnData or None
    """
    path = find_cell_h5ad_path(experiment, channel, storage_roots, feature_dir, metadata_path)
    if path is None:
        return None

    import anndata as ad

    return ad.read_h5ad(path)


def find_cell_h5ad_path(
    experiment: str,
    channel: str,
    storage_roots: List[Path],
    feature_dir: str,
    metadata_path: str,
) -> Optional[Path]:
    """Locate a cell-level h5ad file without loading it.

    Useful for lightweight pre-scanning (e.g. reading shape via h5py).

    Returns
    -------
    Path or None
    """
    from cyclops_utils.data.feature_metadata import FeatureMetadata

    effective_feature_dir, raw_channel = _resolve_feature_dir(channel, feature_dir)

    exp_short = experiment.split("_")[0]
    fm = FeatureMetadata(metadata_path=metadata_path)
    reporter = fm.get_biological_signal(exp_short, raw_channel)

    exp_dir = find_experiment_dir(experiment, storage_roots)
    if exp_dir is None:
        return None

    anndata_dir = exp_dir / "3-assembly" / effective_feature_dir / "anndata_objects"
    cell_file = anndata_dir / f"features_processed_{reporter}.h5ad"
    if not cell_file.exists():
        cell_file = anndata_dir / f"features_processed_{raw_channel}.h5ad"
    if not cell_file.exists():
        return None

    return cell_file


def count_cells_per_signal_group(
    signal_groups: Dict[str, List[Tuple[str, str]]],
    storage_roots: List[Path],
    feature_dir: str,
    metadata_path: str,
    max_workers: int = 16,
) -> Dict[str, int]:
    """Pre-scan cell counts per signal group via concurrent ``h5py.File`` reads.

    Each task opens one h5ad to read ``X.shape[0]`` only — never falls back to
    a full ``ad.read_h5ad`` (that pulled multi-GB X matrices into memory under
    the old fallback, masking I/O issues as multi-minute stalls). Files that
    fail h5py open are reported and counted as 0.

    Progress is logged every 25 files plus a final summary. Concurrent
    workers parallelize NFS metadata I/O — the bottleneck on shared
    filesystems with hundreds of large h5ads (e.g. CellProfiler at
    paper-v1 scale).
    """
    import h5py
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_tasks = []
    for signal, pairs in signal_groups.items():
        for exp, ch in pairs:
            all_tasks.append((signal, exp, ch))

    n_tasks = len(all_tasks)
    counts: Dict[str, int] = {signal: 0 for signal in signal_groups}
    errors: List[Tuple[str, str, str]] = []  # (exp, ch, err)

    def _one(task):
        signal, exp, ch = task
        cell_file = find_cell_h5ad_path(exp, ch, storage_roots, feature_dir, metadata_path)
        if cell_file is None:
            return signal, 0, None
        try:
            with h5py.File(cell_file, "r") as f:
                return signal, int(f["X"].shape[0]), None
        except Exception as exc:
            return signal, 0, f"{exp}/{ch}: {exc}"

    logger.info(
        f"Pre-scanning {n_tasks} h5ad files across {len(signal_groups)} signal "
        f"groups ({max_workers} parallel workers)..."
    )
    t_start = _time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, t): t for t in all_tasks}
        for fut in as_completed(futures):
            signal, n, err = fut.result()
            counts[signal] += n
            if err:
                errors.append((signal, err.split(":")[0], err))
            done += 1
            if done % 25 == 0 or done == n_tasks:
                logger.info(
                    f"  Scanned {done}/{n_tasks} files in {_time.time()-t_start:.0f}s"
                )

    if errors:
        logger.warning(f"  {len(errors)} files failed h5py.File read (counted as 0):")
        for sig, exp_ch, err in errors[:5]:
            logger.warning(f"    {err}")
        if len(errors) > 5:
            logger.warning(f"    ... +{len(errors) - 5} more")

    logger.info(f"Cell counts per signal group:")
    for signal, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {signal:<45} {n:>12,}")
    logger.info(f"  {'TOTAL':<45} {sum(counts.values()):>12,}")

    return counts


# ---------------------------------------------------------------------------
# Channel label resolution
# ---------------------------------------------------------------------------

def resolve_channel_label(fm, experiment: str, file_channel: str) -> Dict[str, str]:
    """Resolve an h5ad channel name to its biological label.

    h5ad files are sometimes named by microscope channel (GFP, mCherry, Phase)
    and sometimes by reporter protein (EEA1, TOMM70A, LAMP1), and sometimes
    by live-cell dye name (LysoTracker_live-cell_dye).  This tries:

    1. Direct lookup: fm.get_channel_info(exp, file_channel)
    2. Reverse lookup (exact): scan YAML channels, match marker == file_channel
    3. Reverse lookup (fuzzy): normalize underscores -> spaces and compare
    4. Cell Painting channels (CP1_organelle_marker)
    5. Hardcoded fixes for known naming mismatches

    Parameters
    ----------
    fm : FeatureMetadata
        Loaded FeatureMetadata instance.
    experiment : str
        Full experiment name (e.g. "ops0046_...").
    file_channel : str
        Channel name from the h5ad filename.

    Returns
    -------
    dict
        Keys: label, short, yaml_channel, method
    """
    exp_short = experiment.split("_")[0]

    # Sibling-tagged channels (4i, cp): strip the tag for lookup; the resolved
    # label is suffixed with the sibling label so sibling variants of the same
    # marker form their own signal groups. ``_tag_result_sibling`` is applied
    # to whichever dict we return.
    sibling_tag = _sibling_tag_for(file_channel)
    if sibling_tag is not None:
        file_channel = file_channel[len(sibling_tag):]

    def _tag_result_sibling(result: Dict[str, str]) -> Dict[str, str]:
        if sibling_tag is None:
            return result
        cfg = SIBLING_TAGS[sibling_tag]
        suffix = cfg["label_suffix"]
        method_tag = sibling_tag.rstrip(":")
        out = dict(result)
        if not out.get("label", "").endswith(suffix):
            out["label"] = f"{out['label']}{suffix}"
        out["method"] = f"{out.get('method', '')}+{method_tag}"
        out["yaml_channel"] = f"{sibling_tag}{out.get('yaml_channel', file_channel)}"
        return out

    # Back-compat alias used in this function before the cp sibling existed.
    _tag_result_4i = _tag_result_sibling

    # Phase2D -> treat as Phase (same label-free brightfield)
    _PHASE_ALIASES = {"Phase2D", "Phase3D"}
    if file_channel in _PHASE_ALIASES:
        file_channel = "Phase"

    # --- Attempt 1: direct lookup ---
    info = fm.get_channel_info(experiment, file_channel)
    if info.get("label") != "unknown":
        label = info["label"]
        if label == "no label":
            yaml_ch = info["channel_name"]
            label = f"autofluorescence, {yaml_ch.lower()}"
            return _tag_result_4i({
                "label": label,
                "short": f"autofluorescence_{yaml_ch.lower()}",
                "yaml_channel": yaml_ch,
                "method": "direct_autofluorescence",
            })
        return _tag_result_4i({
            "label": label,
            "short": fm.get_short_label(experiment, file_channel),
            "yaml_channel": info["channel_name"],
            "method": "direct",
        })

    # --- Attempt 2: reverse lookup ---
    def _norm(s: str) -> str:
        return s.replace("_", " ").replace("-", " ").lower()

    def _collapse(s: str) -> str:
        return s.replace("_", "").replace("-", "").replace(" ", "").lower()

    file_norm = _norm(file_channel)
    file_collapsed = _collapse(file_channel)

    if exp_short in fm.metadata:
        for ch_entry in fm.metadata[exp_short]:
            if not isinstance(ch_entry, dict) or "label" not in ch_entry:
                continue
            label = ch_entry["label"]

            # 2a. Exact marker match
            if "," in label:
                marker = label.split(",", 1)[1].strip()
                if marker == file_channel:
                    return _tag_result_4i({
                        "label": label,
                        "short": marker,
                        "yaml_channel": ch_entry.get("channel_name", file_channel),
                        "method": "reverse_marker",
                    })
            # 2b. Exact whole-label match
            if label == file_channel:
                return _tag_result_4i({
                    "label": label,
                    "short": file_channel,
                    "yaml_channel": ch_entry.get("channel_name", file_channel),
                    "method": "reverse_label",
                })
            # 2c. Fuzzy marker match
            if "," in label:
                marker = label.split(",", 1)[1].strip()
                if _norm(marker) == file_norm:
                    return _tag_result_4i({
                        "label": label,
                        "short": marker,
                        "yaml_channel": ch_entry.get("channel_name", file_channel),
                        "method": "reverse_marker_fuzzy",
                    })
            # 2d. Fuzzy whole-label match
            if _norm(label) == file_norm:
                return _tag_result_4i({
                    "label": label,
                    "short": label,
                    "yaml_channel": ch_entry.get("channel_name", file_channel),
                    "method": "reverse_label_fuzzy",
                })
            # 2e. Collapsed marker match
            if "," in label:
                marker = label.split(",", 1)[1].strip()
                if _collapse(marker) == file_collapsed:
                    return _tag_result_4i({
                        "label": label,
                        "short": marker,
                        "yaml_channel": ch_entry.get("channel_name", file_channel),
                        "method": "reverse_marker_collapsed",
                    })
            # 2f. Collapsed whole-label match
            if _collapse(label) == file_collapsed:
                return _tag_result_4i({
                    "label": label,
                    "short": label,
                    "yaml_channel": ch_entry.get("channel_name", file_channel),
                    "method": "reverse_label_collapsed",
                })

    # --- Attempt 3: Cell Painting channels ---
    if file_channel.startswith(("CP1_", "CP2_")):
        parts = file_channel.split("_", 2)
        if len(parts) == 3:
            return _tag_result_4i({
                "label": f"{parts[1]}, {parts[2]}",
                "short": parts[2],
                "yaml_channel": file_channel,
                "method": "cellpainting",
            })

    # --- Attempt 4: hardcoded fixes ---
    _HARDCODED = {
        "ChromaLive_488_emission": "ChromaLIVE 488 excitation",
        "ChromaLive488emission": "ChromaLIVE 488 excitation",
        "FastAct": "actin filament, FastAct_SPY555 Live Cell Dye",
        "unlabeled": "no label",
    }
    if file_channel in _HARDCODED:
        hardcoded_label = _HARDCODED[file_channel]
        return _tag_result_4i({
            "label": hardcoded_label,
            "short": hardcoded_label,
            "yaml_channel": file_channel,
            "method": "hardcoded",
        })

    # --- Unresolved ---
    return _tag_result_4i({
        "label": f"(unmapped: {file_channel})",
        "short": file_channel,
        "yaml_channel": file_channel,
        "method": "unresolved",
    })


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def _get_non_default_library_experiments() -> set:
    """Return experiment short names that use a non-default gene_index library."""
    from cyclops_utils.data.bad_experiments import load_library_map

    lib_map = load_library_map()
    default_gene_index = lib_map.get("default", {}).get("gene_index")
    excluded = set()
    for exp_short, override in lib_map.get("overrides", {}).items():
        if override.get("gene_index") and override["gene_index"] != default_gene_index:
            excluded.add(exp_short)
    return excluded


def discover_dino_experiments(
    storage_roots: List[Path],
    feature_dir: str = "dino_features",
    include_cellpainting: bool = False,
    include_4i: bool = False,
    include_cp: bool = False,
    include_standard: bool = True,
    force_include: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """Discover experiments with DINO guide_bulked_*.h5ad files.

    Filters out bad experiments via ``cyclops_utils.data.bad_experiments.is_excluded``
    and experiments with non-default gene libraries.

    Parameters
    ----------
    storage_roots : list of Path
        Root directories to scan.
    feature_dir : str
        Subdirectory under ``3-assembly/`` containing DINO features.
    include_cellpainting : bool
        Include legacy ``CP1_*``/``CP2_*`` channels stored inside the standard
        ``feature_dir`` (older layout). Default: False.
    include_4i : bool
        Also scan ``<feature_dir>_4i/`` sibling and tag those channels ``4i:``.
    include_cp : bool
        Also scan ``<feature_dir>_cp/`` sibling and tag those channels ``cp:``
        (new Cell Painting layout, e.g. ops0094 ConA/Hoechst/...).

    Returns
    -------
    list of (experiment_name, channel) tuples
    """
    from cyclops_utils.data.bad_experiments import is_excluded
    non_default_lib = _get_non_default_library_experiments()
    force_set = set(force_include) if force_include else set()

    pairs: List[Tuple[str, str]] = []
    seen: set = set()

    for root in storage_roots:
        if not root.exists():
            continue
        try:
            exp_dirs = sorted(root.iterdir())
        except PermissionError:
            logger.warning(f"  Permission denied: {root}")
            continue

        for exp_dir in exp_dirs:
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue

            exp_name = exp_dir.name
            exp_short = exp_name.split("_")[0]

            if exp_short in seen:
                continue
            # ``force_include`` bypasses both the bad-experiment list and the
            # non-default-library filter — used when the caller explicitly
            # asked for these experiments (e.g. via --experiments).
            if exp_short not in force_set and (
                is_excluded(exp_short) or exp_short in non_default_lib
            ):
                continue

            # (subdir, sibling_tag_or_None)
            scan_dirs: List[Tuple[str, Optional[str]]] = []
            if include_standard:
                scan_dirs.append((feature_dir, None))
            if include_4i:
                scan_dirs.append((f"{feature_dir}{SIBLING_TAGS['4i:']['feature_suffix']}", "4i:"))
            if include_cp:
                scan_dirs.append((f"{feature_dir}{SIBLING_TAGS['cp:']['feature_suffix']}", "cp:"))

            found_for_exp = False
            for fdir, sibling_tag in scan_dirs:
                anndata_dir = exp_dir / "3-assembly" / fdir / "anndata_objects"
                try:
                    if not anndata_dir.exists():
                        continue
                except PermissionError:
                    continue

                for h5ad in sorted(anndata_dir.glob("guide_bulked_*.h5ad")):
                    channel = h5ad.stem.replace("guide_bulked_", "")
                    if channel.startswith("umap_"):
                        continue
                    if not include_cellpainting and channel.startswith(("CP1_", "CP2_")):
                        continue
                    if sibling_tag:
                        channel = f"{sibling_tag}{channel}"
                    pairs.append((exp_name, channel))
                    found_for_exp = True

            if found_for_exp:
                seen.add(exp_short)

    extras = []
    if not include_standard:
        extras.append("-standard")
    if include_4i:
        extras.append("+4i")
    if include_cp:
        extras.append("+cp")
    extras_str = f" [{' '.join(extras)}]" if extras else ""
    logger.info(
        f"  Discovered {len(pairs)} DINO (experiment, channel) pairs "
        f"across {len(seen)} experiments{extras_str}"
    )
    return pairs


def discover_cellprofiler_experiments(
    storage_roots: List[Path],
    include_cellpainting: bool = False,
    include_4i: bool = False,
    include_cp: bool = False,
    include_standard: bool = True,
    force_include: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """Discover experiments with CellProfiler features_processed_*.h5ad files.

    When ``include_4i`` / ``include_cp`` is True, also scans ``cell-profiler-4i/``
    or ``cell-profiler-cp/`` siblings and tags channels with ``4i:`` / ``cp:``.
    When ``include_standard`` is False, skips the canonical ``cell-profiler/`` dir
    (use with ``include_4i`` / ``include_cp`` to run on a single sibling).

    ``force_include``: set of experiment short names (e.g. ``{"ops0146"}``)
    that should bypass both the bad-experiment list and the non-default-library
    filter — used when the caller explicitly asked for these experiments.
    """
    from cyclops_utils.data.bad_experiments import is_excluded
    non_default_lib = _get_non_default_library_experiments()
    force_set = set(force_include) if force_include else set()

    pairs: List[Tuple[str, str]] = []
    seen: set = set()

    for root in storage_roots:
        if not root.exists():
            continue
        try:
            exp_dirs = sorted(root.iterdir())
        except PermissionError:
            continue

        for exp_dir in exp_dirs:
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue

            exp_name = exp_dir.name
            exp_short = exp_name.split("_")[0]

            if exp_short in seen:
                continue
            if exp_short not in force_set and (
                is_excluded(exp_short) or exp_short in non_default_lib
            ):
                continue

            scan_dirs: List[Tuple[str, Optional[str]]] = []
            if include_standard:
                scan_dirs.append(("cell-profiler", None))
            if include_4i:
                scan_dirs.append((SIBLING_TAGS["4i:"]["cp_dir"], "4i:"))
            if include_cp:
                scan_dirs.append((SIBLING_TAGS["cp:"]["cp_dir"], "cp:"))

            found_for_exp = False
            for fdir, sibling_tag in scan_dirs:
                anndata_dir = exp_dir / "3-assembly" / fdir / "anndata_objects"
                try:
                    if not anndata_dir.exists():
                        continue
                except PermissionError:
                    continue

                # Discover via guide_bulked_*.h5ad (the per-channel aggregates),
                # matching DINO discovery. This avoids picking up orphan
                # features_processed_*_nofilters.h5ad files that have no
                # corresponding guide_bulked_* aggregate.
                for h5ad in sorted(anndata_dir.glob("guide_bulked_*.h5ad")):
                    channel = h5ad.stem.replace("guide_bulked_", "")
                    if not include_cellpainting and channel.startswith(("CP1_", "CP2_")):
                        continue
                    if sibling_tag:
                        channel = f"{sibling_tag}{channel}"
                    pairs.append((exp_name, channel))
                    found_for_exp = True

            if found_for_exp:
                seen.add(exp_short)

    extras = []
    if not include_standard:
        extras.append("-standard")
    if include_4i:
        extras.append("+4i")
    if include_cp:
        extras.append("+cp")
    extras_str = f" [{' '.join(extras)}]" if extras else ""
    logger.info(
        f"  Discovered {len(pairs)} CellProfiler (experiment, channel) pairs "
        f"across {len(seen)} experiments{extras_str}"
    )
    return pairs


# ---------------------------------------------------------------------------
# Signal grouping
# ---------------------------------------------------------------------------

def build_signal_groups(
    all_pairs: List[Tuple[str, str]],
    fm,
) -> Dict[str, List[Tuple[str, str]]]:
    """Group (exp, channel) pairs by resolved biological signal label.

    Returns dict: signal_label -> [(exp, channel), ...].
    Skips unknown/unmapped channels.
    """
    import contextlib
    import io

    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    skipped = []

    seen: set = set()
    for exp, ch in all_pairs:
        with contextlib.redirect_stdout(io.StringIO()):
            resolved = resolve_channel_label(fm, exp, ch)
        sig = resolved["label"]
        if sig == "unknown" or sig.startswith("(unmapped:"):
            skipped.append((exp, ch, sig))
            continue
        # Deduplicate: one entry per (experiment_short, signal)
        exp_short = exp.split("_")[0]
        key = (exp_short, sig)
        if key in seen:
            raise ValueError(
                f"DUPLICATE: {exp_short} has multiple channels mapping to signal '{sig}' "
                f"(channel '{ch}'). Delete stale h5ads from anndata_objects/."
            )
        seen.add(key)
        groups[sig].append((exp, ch))

    if skipped:
        logger.warning(
            f"{len(skipped)} experiment-channel pairs could not be mapped and will be skipped"
        )
        for exp, ch, sig in skipped:
            logger.warning(f"  {exp} / {ch} -> {sig!r}")

    total_mapped = sum(len(v) for v in groups.values())
    logger.info(
        f"Signal grouping: {total_mapped}/{len(all_pairs)} channels -> "
        f"{len(groups)} signal groups"
    )
    for sig in sorted(groups.keys()):
        pairs = groups[sig]
        exps = [e.split("_")[0] for e, _ in pairs]
        logger.info(f"  {sig:<45} {len(pairs)} exp(s): {', '.join(exps)}")

    return dict(groups)


def sanitize_signal_filename(signal: str) -> str:
    """Sanitize a biological signal label for use as a filename."""
    return (
        signal.replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )
