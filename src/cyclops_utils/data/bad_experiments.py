"""Shared experiment exclusion lists loaded from bad_experiment.yaml.

Provides a single source of truth for experiment categories (bad, iss_only,
do_not_run, non_standard, positive_control, need_rescue) and a date cutoff.

The YAML lives under ``OPS_CONFIGS_DIR`` alongside ops_library_map.yaml and
ops_channel_maps.yaml, not in the package. When it is absent (e.g. outside the
Biohub filesystem) every category reads empty and nothing is excluded.

YAML entries can be plain values or dicts with ``name``/``reason`` keys::

    bad:
      - name: ops0033_20250429
        reason: known bad quality
      - ops0039_20250508          # plain string (no reason)

Usage::

    from cyclops_utils.data.bad_experiments import (
        is_excluded, get_category, get_date_cutoff, get_reason,
    )

    # Check if an experiment should be skipped (default categories)
    if is_excluded("ops0033_20250429"):
        print("skip")

    # Get a specific category list (names/numbers only, reasons stripped)
    iss_only_nums = get_category("iss_only")         # [11, 28, 29, ...]
    bad_names = get_category("bad")                   # ["ops0033_20250429", ...]

    # Look up why an experiment is excluded
    reason = get_reason("ops0033_20250429")           # "known bad quality"

    # Get all excluded experiment numbers (union of categories)
    excluded_nums = get_excluded_experiment_numbers()  # {1, 2, 3, 5, 8, ...}
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from cyclops_utils.data.filesystem import extract_ops_key
from cyclops_utils.paths import BASE_PATH


_CONFIGS_DIR = Path(os.environ.get("OPS_CONFIGS_DIR", f"{BASE_PATH}/configs"))
_YAML_PATH = _CONFIGS_DIR / "bad_experiment.yaml"
_LIBRARY_MAP_PATH = _CONFIGS_DIR / "library" / "ops_library_map.yaml"
_CHANNEL_MAPS_PATH = _CONFIGS_DIR / "ops_channel_maps.yaml"
_PAPER_V1_LIST_PATH = _CONFIGS_DIR / "good_experiment_list_v1.yml"
_PAPER_V2_LIST_PATH = _CONFIGS_DIR / "good_experiment_list_v2.yml"

# Recognized project tags. Projects are config-driven: the per-experiment
# ``project`` field in ops_library_map.yaml (see :func:`derive_project`). This
# baseline set is used for ``--project`` filter normalization; ``paper_v1`` /
# ``paper_v2`` are curated overlays (good_experiment_list_*.yml) that cross-cut
# the per-experiment project rather than replacing it.
KNOWN_PROJECTS = ("Validation", "40_marker", "paper_v1", "paper_v2")

# Recognized experimental_design (imaging modality) tags. Each experiment
# carries a *list* of designs: ``livecell_OPS`` is always present (it's the
# baseline for every experiment), and add-on modalities like ``Cell_Painting``,
# ``4i``, or ``MERFISH`` stack on top when applicable.
KNOWN_EXPERIMENTAL_DESIGNS = ("livecell_OPS", "Cell_Painting", "4i", "MERFISH", "CROPseq")

# Default categories used by is_excluded() — matches what report_pipeline_status skips
DEFAULT_EXCLUDE_CATEGORIES = ("bad", "iss_only", "do_not_run", "non_standard", "positive_control", "need_rescue")


def _ops_key_to_number(ops_key: str | None) -> int | None:
    """Convert 'ops0033' → 33. Returns None if ops_key is None."""
    if ops_key is None:
        return None
    m = re.search(r"ops0*(\d+)", ops_key)
    return int(m.group(1)) if m else None


def _extract_date(name: str) -> str | None:
    """Extract date from 'ops0033_20250429' → '20250429'."""
    m = re.match(r"^ops\d{4}_(\d{8})$", name)
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def load_bad_experiments() -> dict:
    """Load the raw YAML dict from bad_experiment.yaml (empty if missing).

    Resolved under ``OPS_CONFIGS_DIR`` (see :data:`_CONFIGS_DIR`) alongside the
    library and channel maps. When the file is absent nothing is excluded.
    """
    if not _YAML_PATH.exists():
        return {}
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f) or {}


def _normalize_entry(entry) -> tuple:
    """Normalize a YAML entry to (value, reason).

    Handles both plain values (str/int) and dicts with name/number + reason.

    Returns:
        (value, reason) where value is str or int, reason is str or None.
    """
    if isinstance(entry, dict):
        value = entry.get("name") or entry.get("number")
        reason = entry.get("reason")
        return value, reason
    return entry, None


def get_date_cutoff() -> str:
    """Return the date cutoff string (e.g. '20250424')."""
    return load_bad_experiments().get("date_cutoff", "")


@lru_cache(maxsize=1)
def _get_tag_map() -> dict[str, str]:
    """Map ops prefix → per-experiment display tag, from the ``tag`` field in
    ops_library_map.yaml (e.g. 'ops0148' → 'POOL_0003'). Only experiments that
    declare a ``tag`` are included.
    """
    overrides = load_library_map().get("overrides", {})
    if not isinstance(overrides, dict):
        return {}
    return {
        key: cfg["tag"]
        for key, cfg in overrides.items()
        if isinstance(cfg, dict) and cfg.get("tag")
    }


@lru_cache(maxsize=1)
def _load_channel_maps() -> dict:
    """Load ops_channel_maps.yaml as a dict (empty if missing)."""
    if not _CHANNEL_MAPS_PATH.exists():
        return {}
    with open(_CHANNEL_MAPS_PATH) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_library_map() -> dict:
    """Load ops_library_map.yaml as a dict (empty if missing).

    Resolved under ``OPS_CONFIGS_DIR`` (see :data:`_CONFIGS_DIR`), so every
    caller sees the same canonical map rather than a checkout-local copy.
    """
    if not _LIBRARY_MAP_PATH.exists():
        return {}
    with open(_LIBRARY_MAP_PATH) as f:
        return yaml.safe_load(f) or {}


def derive_project(experiment_name: str) -> str:
    """Classify an experiment into a project tag.

    Reads the per-experiment ``project`` field from ops_library_map.yaml.
    Experiments without an explicit project fall back to ``40_marker`` (the
    default OPS project; the imaging modality is captured separately by
    :func:`derive_experimental_design`).

    Args:
        experiment_name: Full experiment name or ops prefix.
    """
    ops_key = extract_ops_key(experiment_name) or experiment_name
    cfg = load_library_map().get("overrides", {}).get(ops_key)
    if isinstance(cfg, dict) and cfg.get("project"):
        return cfg["project"]
    return "40_marker"


def derive_experimental_design(experiment_name: str) -> list[str]:
    """Return the imaging modalities used in an experiment as a list.

    ``livecell_OPS`` is always included (every experiment in this dataset has
    live-cell OPS imaging). Add-on modalities stack on top when their flags
    are set:
        - ``cell_painting: enabled: true`` (channel map)     → ``Cell_Painting``
        - ``four_i: enabled: true`` (channel map)            → ``4i``
        - codebook filename contains 'merfish' (library map) → ``MERFISH``

    ``CROPseq`` is not derivable from configs alone; it is added by
    ``generate_config_files.py`` (hardcoded experiment-number list) and
    persisted into the per-experiment config's ``experimental_design`` field.

    Args:
        experiment_name: Full experiment name or ops prefix.
    """
    ops_key = extract_ops_key(experiment_name) or experiment_name
    designs: list[str] = ["livecell_OPS"]

    channel_entry = _load_channel_maps().get(ops_key)
    if isinstance(channel_entry, list):
        for item in channel_entry:
            if not isinstance(item, dict):
                continue
            cp = item.get("cell_painting")
            if isinstance(cp, dict) and cp.get("enabled") and "Cell_Painting" not in designs:
                designs.append("Cell_Painting")
            four_i = item.get("four_i")
            if isinstance(four_i, dict) and four_i.get("enabled") and "4i" not in designs:
                designs.append("4i")

    overrides = load_library_map().get("overrides", {})
    cfg = overrides.get(ops_key)
    if isinstance(cfg, dict):
        codebook = (cfg.get("codebook") or "").lower()
        if "merfish" in codebook and "MERFISH" not in designs:
            designs.append("MERFISH")

    return designs


def _load_paper_list(path: Path) -> set:
    """Return the set of experiment names from a good_experiment_list_*.yml file."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    channels = data.get("experiments_channels", {})
    if not isinstance(channels, dict):
        return set()
    return set(channels.keys())


@lru_cache(maxsize=1)
def get_paper_v1_experiments() -> set:
    """Return the set of experiment names curated in good_experiment_list_v1.yml.

    Used by the ``paper_v1`` project tag, which is an overlay/subset rather
    than a mutually exclusive project classification.
    """
    return _load_paper_list(_PAPER_V1_LIST_PATH)


def is_in_paper_v1(experiment_name: str) -> bool:
    """Check whether an experiment is in the curated paper_v1 list."""
    return experiment_name in get_paper_v1_experiments()


@lru_cache(maxsize=1)
def get_paper_v2_experiments() -> set:
    """Return the set of experiment names curated in good_experiment_list_v2.yml.

    The ``paper_v2`` overlay extends ``paper_v1`` to the current curated
    experiment set (see good_experiment_list_v2.yml).
    """
    return _load_paper_list(_PAPER_V2_LIST_PATH)


def is_in_paper_v2(experiment_name: str) -> bool:
    """Check whether an experiment is in the curated paper_v2 list."""
    return experiment_name in get_paper_v2_experiments()


@lru_cache(maxsize=64)
def count_codebook_perturbations(codebook_filename: str) -> int:
    """Return the number of perturbations (rows) in a codebook CSV.

    Resolves the file under ``_CONFIGS_DIR / "library"`` and subtracts the
    header row. Returns 0 when the file is missing or unreadable.
    """
    if not codebook_filename:
        return 0
    path = _CONFIGS_DIR / "library" / codebook_filename
    if not path.exists():
        return 0
    try:
        import csv as _csv
        with open(path) as f:
            count = sum(1 for _ in _csv.reader(f))
        return max(count - 1, 0)
    except Exception:
        return 0


def derive_library(experiment_name: str) -> str:
    """Return a short, human-readable library name for an experiment.

    Prefers the explicit ``library`` field in ops_library_map.yaml; otherwise
    falls back to the codebook filename stem (no ``.csv``), reading the
    experiment's override then the default codebook.
    """
    lib_map = load_library_map()
    overrides = lib_map.get("overrides", {})
    defaults = lib_map.get("default", {})

    ops_key = extract_ops_key(experiment_name) or experiment_name
    cfg = overrides.get(ops_key, {}) if isinstance(overrides, dict) else {}
    if isinstance(cfg, dict) and cfg.get("library"):
        return cfg["library"]

    codebook = (cfg.get("codebook") if isinstance(cfg, dict) else None) or defaults.get("codebook") or ""
    stem = codebook.rsplit("/", 1)[-1]
    if stem.lower().endswith(".csv"):
        stem = stem[:-4]
    return stem or "unknown"


def get_experiment_tag(experiment_name: str) -> Optional[str]:
    """Return the per-experiment display tag (the ``tag`` field in
    ops_library_map.yaml), or None if the experiment declares no tag.

    Args:
        experiment_name: Full experiment name (e.g. 'ops0138_20260201') or
                         ops prefix (e.g. 'ops0138').
    """
    ops_key = extract_ops_key(experiment_name) or experiment_name
    return _get_tag_map().get(ops_key)


# Label substrings that make a channel a "bad channel" regardless of experiment
# (non-informative / non-marker channels): no-label wells, bleedthrough, and
# autofluorescence-only channels.
BAD_CHANNEL_LABEL_TOKENS = ("no label", "bleedthrough", "autofluorescence")


def get_bad_channels() -> list[dict]:
    """Return explicit channel-level exclusions from the ``bad_channels`` section.

    Each entry is a dict with ``name`` (experiment), ``channel``, optional
    ``marker``, and ``reason``. These are dropped per-(experiment, channel) from
    marker analysis — the experiment itself stays valid. Note that autofluorescence
    / no-label / bleedthrough channels are ALSO bad (detected by label via
    :func:`is_bad_channel`) without needing to be listed here.
    """
    raw = load_bad_experiments().get("bad_channels", []) or []
    return [e for e in raw if isinstance(e, dict)]


def is_bad_channel(
    experiment_name: str, channel_name: str, label: Optional[str] = None
) -> bool:
    """True if this channel should be dropped from per-marker analysis.

    A channel is "bad" if either:
      * its ``label`` is non-informative (no label / bleedthrough /
        autofluorescence), or
      * the (experiment, channel) pair is listed explicitly in ``bad_channels``
        (e.g. ops0101 mCherry FeRhoNox — saturated).
    """
    if label is not None:
        ll = str(label).lower().strip()
        if any(tok in ll for tok in BAD_CHANNEL_LABEL_TOKENS):
            return True

    exp_num = _ops_key_to_number(extract_ops_key(experiment_name))
    chan = str(channel_name).strip().lower()
    for e in get_bad_channels():
        if str(e.get("channel", "")).strip().lower() != chan:
            continue
        name = e.get("name", "")
        if name == experiment_name:
            return True
        if exp_num is not None and _ops_key_to_number(extract_ops_key(name)) == exp_num:
            return True
    return False


def get_category(category: str) -> list:
    """Return the list of values for a category (names or numbers, reasons stripped).

    Args:
        category: One of 'bad', 'iss_only', 'do_not_run', 'non_standard',
                  'positive_control', 'need_rescue'.

    Returns:
        List of experiment names (str) or numbers (int).
    """
    raw = load_bad_experiments().get(category, [])
    return [_normalize_entry(e)[0] for e in raw]


def get_category_with_reasons(category: str) -> list[tuple]:
    """Return list of (value, reason) tuples for a category.

    Args:
        category: Category name.

    Returns:
        List of (name_or_number, reason_or_None) tuples.
    """
    raw = load_bad_experiments().get(category, [])
    return [_normalize_entry(e) for e in raw]


def get_reason(experiment_name: str) -> Optional[str]:
    """Look up the reason an experiment is excluded.

    Searches all categories for a matching experiment name or number.

    Returns:
        Reason string, or None if no reason recorded.
    """
    exp_num = _ops_key_to_number(extract_ops_key(experiment_name))
    data = load_bad_experiments()

    for key, entries in data.items():
        if key == "date_cutoff":
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            value, reason = _normalize_entry(entry)
            if value == experiment_name:
                return reason
            if isinstance(value, int) and value == exp_num:
                return reason

    # Check date cutoff
    cutoff = data.get("date_cutoff", "")
    if cutoff:
        exp_date = _extract_date(experiment_name)
        if exp_date and exp_date < cutoff:
            return f"before date cutoff ({cutoff})"

    return None


def get_excluded_experiment_numbers(
    categories: tuple[str, ...] = DEFAULT_EXCLUDE_CATEGORIES,
) -> set[int]:
    """Return set of experiment numbers to exclude (union of requested categories).

    Handles both full-name lists (extracts number) and raw int lists.
    """
    numbers: set[int] = set()

    for cat in categories:
        for value in get_category(cat):
            if isinstance(value, int):
                numbers.add(value)
            elif isinstance(value, str):
                num = _ops_key_to_number(extract_ops_key(value))
                if num is not None:
                    numbers.add(num)

    return numbers


def get_excluded_experiment_names(
    categories: tuple[str, ...] = DEFAULT_EXCLUDE_CATEGORIES,
) -> set[str]:
    """Return set of full experiment names from string-valued category lists.

    Only includes entries that are full names (strings), not raw numbers.
    """
    names: set[str] = set()

    for cat in categories:
        for value in get_category(cat):
            if isinstance(value, str):
                names.add(value)

    return names


def is_excluded(
    experiment_name: str,
    categories: tuple[str, ...] = DEFAULT_EXCLUDE_CATEGORIES,
    date_cutoff: bool = True,
) -> bool:
    """Check if an experiment should be excluded.

    Args:
        experiment_name: Full experiment name (e.g. 'ops0033_20250429').
        categories: Which category lists to check against.
        date_cutoff: If True, also exclude experiments before the date cutoff.

    Returns:
        True if the experiment should be excluded.
    """
    # Check date cutoff
    if date_cutoff:
        cutoff = get_date_cutoff()
        if cutoff:
            exp_date = _extract_date(experiment_name)
            if exp_date and exp_date < cutoff:
                return True

    # Check by full name
    if experiment_name in get_excluded_experiment_names(categories):
        return True

    # Check by experiment number
    exp_num = _ops_key_to_number(extract_ops_key(experiment_name))
    if exp_num is not None and exp_num in get_excluded_experiment_numbers(categories):
        return True

    return False
