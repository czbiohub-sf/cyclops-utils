from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
import shutil
from typing import Optional, Union
from cyclops_utils.io.zarr_utils import is_precreated_store


def async_delete_path(path: Union[str, Path]) -> Optional[Path]:
    """Delete a file/dir without blocking the caller.

    Renames the target to a sibling ``.trash_*`` (instant on the same
    filesystem) and ``rm -rf``'s it in a detached background process. Use for
    large zarr stores / symlink trees on NFS where a synchronous
    ``shutil.rmtree`` would stall the caller on slow per-file unlinks.

    Returns the trash path (deletion continues in the background), or ``None``
    if the target did not exist.
    """
    p = Path(path)
    if not p.exists() and not p.is_symlink():
        return None
    trash = p.with_name(f".trash_{p.name}_{os.getpid()}_{int(time.time())}")
    p.rename(trash)
    subprocess.Popen(
        ["rm", "-rf", str(trash)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return trash

# Verbose print helper (controlled by CLI --verbose)
VERBOSE: bool = False


def find_monorepo_root(start: Optional[Path] = None) -> Path:
    """Walk up from ``start`` (default: this file) looking for the
    cyclops-monorepo root — the directory that contains both ``cyclops_utils``
    and ``cyclops_model`` as siblings.

    Use this instead of hardcoded ``Path(__file__).parents[N]`` so scripts
    don't break when their location changes within the tree.

    Raises ``FileNotFoundError`` if no such ancestor is found.
    """
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / "cyclops_utils").is_dir() and (p / "cyclops_model").is_dir():
            return p
    raise FileNotFoundError(f"cyclops-monorepo root not found from {start}")


def vprintf(fmt: str, *args) -> None:
    try:
        if VERBOSE:
            if args:
                print(fmt % args)
            else:
                print(fmt)
    except Exception:
        if VERBOSE:
            try:
                print(fmt, *args)
            except Exception:
                pass


# Iterate over experiment configs
def _iter_experiment_configs() -> list[tuple[str, Path]]:
    """Return list of (experiment_name, path) for all available experiment configs.

    Mirrors the logic in orchestrator: files named "*_config.yaml" and experiment
    name derived from the stem with trailing "_config" removed.
    """
    try:
        from cyclops_utils.data.experiment import OpsDataset

        ds = OpsDataset("dummy")
        cfg_dir = ds.config_paths.get("exp_config_dir")
        if cfg_dir is None:
            vprintf("[exp-grids] No exp_config_dir found in OpsDataset.")
            return []
        results: list[tuple[str, Path]] = []
        for cfg in sorted(Path(cfg_dir).glob("*_config.yaml")):
            stem = cfg.stem
            exp_name = stem[:-7] if stem.endswith("_config") else stem
            results.append((exp_name, cfg))
        vprintf(
            "[exp-grids] Found %d experiment configs in %s", len(results), str(cfg_dir)
        )
        return results
    except Exception:
        vprintf("[exp-grids] Exception while listing experiment configs.")
        return []


def extract_ops_key(experiment: str) -> str | None:
    """Extract the canonical ops key (e.g., 'ops0113') from an experiment name.

    Args:
        experiment: Experiment name like "ops0113_20260108", "ops0113", or "OPS0113"

    Returns:
        Canonical ops key like "ops0113", or None if no match
    """
    import re
    match = re.search(r"ops(\d{4})", experiment, re.IGNORECASE)
    if match:
        return f"ops{match.group(1)}".lower()
    return None


def canonicalize_well_path(well_input: str) -> str:
    """Convert well identifier to canonical path format.

    Accepts various well formats and converts to path format (e.g., "A/1").

    Args:
        well_input: Well identifier in various formats:
                   - "1" -> "A/1" (assumes row A)
                   - "A1" -> "A/1"
                   - "B2" -> "B/2"
                   - "A/1" -> "A/1" (already canonical)

    Returns:
        Well path in canonical format "ROW/COL"
    """
    import re

    well_str = str(well_input).strip().upper()

    # Already in path format
    if "/" in well_str:
        return well_str

    # Just a number, assume row A
    if well_str.isdigit():
        return f"A/{well_str}"

    # Format like "A1" or "AA4"
    m = re.match(r"^([A-Za-z]+)(\d+)$", well_str)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # Unknown format, return as-is
    return well_str


def parse_well(well) -> tuple[str, int]:
    """Split a well identifier into (row, col), e.g. "B/2/0" -> ("B", 2).

    Accepts the same formats as canonicalize_well_path (1, "A1", "B2", "A/1").
    """
    row, col = canonicalize_well_path(well).split("/")[:2]
    return row, int(col)


def well_to_prefix(well: str) -> str:
    """Convert well format to position prefix for filtering.

    Converts from experiment config well format (e.g., "A/1/0") to position
    name prefix format (e.g., "A1-") used in position lists.

    Args:
        well: Well identifier in format "ROW/COL/FIELD" (e.g., "A/1/0")

    Returns:
        Position prefix in format "ROWCOL-" (e.g., "A1-")

    Examples:
        >>> well_to_prefix("A/1/0")
        'A1-'
        >>> well_to_prefix("B/2/0")
        'B2-'
    """
    parts = well.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}{parts[1]}-"
    return well


def convert_position_to_hcs(pos_key: str) -> str:
    """Convert 'A1-Site_0' format to 'A/1/0' format.

    Uses canonicalize_well_path to handle well formatting.
    """
    try:
        well_part, site_part = pos_key.split("-Site_")
        well_path = canonicalize_well_path(well_part)
        return f"{well_path}/{site_part}"
    except Exception:
        # Fallback for unexpected formats
        return pos_key


def _select_canonical_experiment(matches: list[str]) -> str | None:
    """Select the canonical experiment name from a list of matches.

    Canonical format is: ops####_YYYYMMDD (e.g., ops0094_20251217)
    This excludes variants with additional suffixes like ops0094_20251217_mark.

    Args:
        matches: List of experiment names to filter

    Returns:
        The canonical experiment name if found, None otherwise.
    """
    import re

    # Pattern: ops followed by 4 digits, underscore, 8-digit date, end of string
    canonical_pattern = re.compile(r"^ops\d{4}_\d{8}$", re.IGNORECASE)

    canonical_matches = [exp for exp in matches if canonical_pattern.match(exp)]

    if len(canonical_matches) == 1:
        return canonical_matches[0]

    return None


def resolve_experiment_name(
    user_input: str,
    verbose: bool = False,
    allow_interactive: bool = False,
    autoselect: bool = False,
) -> str:
    """Resolve shorthand experiment identifier to full experiment name.

    Accepts partial identifiers (e.g., "33" or "ops33") and matches against
    available experiment configs to return the full name (e.g., "ops0033_20250429").

    Args:
        user_input: Experiment name or shorthand (e.g., "33", "ops33", "ops0033_20250429")
        verbose: Whether to print resolution messages
        allow_interactive: If True, prompts user to choose when multiple matches exist
        autoselect: If True, automatically selects the canonical experiment name
                   (format: ops####_YYYYMMDD) when multiple matches exist.
                   Takes precedence over allow_interactive.

    Returns:
        The matched experiment name or the original input if no match found.
    """
    import re

    experiments_list = _iter_experiment_configs()
    if not experiments_list:
        return user_input

    experiments = [name for name, _ in experiments_list]

    # Normalize input
    normalized = user_input.strip().lower()

    # Check for exact match first
    exact_matches = [exp for exp in experiments if exp.lower() == normalized]
    if exact_matches:
        print(f"[experiment] Using: {exact_matches[0]}")
        return exact_matches[0]

    # Try to match experiment number (e.g., "46" -> "ops0046_...")
    # Look for patterns like ops0046, ops46, etc.
    input_digits = re.sub(r"\D", "", normalized)  # Extract just digits
    if input_digits:
        # Try exact experiment number match (e.g., "46" matches "ops0046" but not "ops0033")
        # Match only at the start after 'ops', before any underscore or end
        exp_num_matches = [
            exp
            for exp in experiments
            if re.search(rf"^ops0*{input_digits}(?:_|$)", exp.lower())
        ]

        if len(exp_num_matches) == 1:
            print(f"[experiment] Resolved '{user_input}' -> {exp_num_matches[0]}")
            return exp_num_matches[0]

        if len(exp_num_matches) > 1:
            # Try autoselect first if enabled
            if autoselect:
                canonical = _select_canonical_experiment(exp_num_matches)
                if canonical:
                    print(f"[experiment] Auto-selected canonical: {canonical}")
                    return canonical
            if allow_interactive:
                return _interactive_select_experiment(exp_num_matches, user_input)
            else:
                print(f"[experiment] Multiple matches for '{user_input}':")
                for exp in exp_num_matches:
                    print(f"  - {exp}")
                print(f"[experiment] Using first match: {exp_num_matches[0]}")
                return exp_num_matches[0]

    # Try substring match at start of name (e.g., "ops33" matches "ops0033_20250429")
    start_matches = [exp for exp in experiments if exp.lower().startswith(normalized)]

    if len(start_matches) == 1:
        print(f"[experiment] Resolved '{user_input}' -> {start_matches[0]}")
        return start_matches[0]

    if len(start_matches) > 1:
        # Try autoselect first if enabled
        if autoselect:
            canonical = _select_canonical_experiment(start_matches)
            if canonical:
                print(f"[experiment] Auto-selected canonical: {canonical}")
                return canonical
        if allow_interactive:
            return _interactive_select_experiment(start_matches, user_input)
        else:
            print(f"[experiment] Multiple matches for '{user_input}':")
            for exp in start_matches:
                print(f"  - {exp}")
            print(f"[experiment] Using first match: {start_matches[0]}")
            return start_matches[0]

    # Fall back to general substring match (e.g., "33" anywhere in name)
    substring_matches = [exp for exp in experiments if normalized in exp.lower()]

    if len(substring_matches) == 1:
        print(f"[experiment] Resolved '{user_input}' -> {substring_matches[0]}")
        return substring_matches[0]

    if len(substring_matches) > 1:
        # Try autoselect first if enabled
        if autoselect:
            canonical = _select_canonical_experiment(substring_matches)
            if canonical:
                print(f"[experiment] Auto-selected canonical: {canonical}")
                return canonical
        if allow_interactive:
            return _interactive_select_experiment(substring_matches, user_input)
        else:
            print(f"[experiment] Multiple matches for '{user_input}':")
            for exp in substring_matches:
                print(f"  - {exp}")
            print(f"[experiment] Using first match: {substring_matches[0]}")
            return substring_matches[0]

    # No match found, return original
    print(f"[experiment] No match found for '{user_input}', using as-is")
    return user_input


def _interactive_select_experiment(matches: list[str], user_input: str) -> str:
    """Helper function for interactive selection among multiple experiment matches."""
    print(
        f"Found {len(matches)} experiments matching '{user_input}'. Please choose one:"
    )
    for idx, name in enumerate(matches, start=1):
        print(f"  {idx}. {name}")

    while True:
        choice = input("Enter number (or 'q' to cancel): ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            print("Selection cancelled. Using first match.")
            return matches[0]
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(matches):
            selected = matches[idx - 1]
            print(f"[experiment] Selected: {selected}")
            return selected
        print(f"Please enter a number between 1 and {len(matches)}.")


import yaml


def _extract_channels_from_config(cfg_path: Path) -> list[str]:
    """Extract human-readable channel labels from an experiment YAML config.

    Prefers values from channel_map; falls back to an empty list on errors.
    """
    try:
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
        ch_map = data.get("channel_map", {}) or {}
        # Keep order stable by sorting keys; display values if present, else keys
        labels: list[str] = []
        for k in sorted(ch_map.keys(), key=lambda x: str(x)):
            v = ch_map.get(k)
            labels.append(str(v) if v is not None else str(k))
        return labels
    except Exception:
        return []


def ensure_output_path(
    output_path: Union[str, Path],
    *,
    prompt_user: bool = True,
    overwrite: Optional[bool] = None,
) -> bool:
    """Ensure a clean, writable output path.

    If the path exists, the function will either:
    - Remove it if ``overwrite`` is True
    - Skip and return False if ``overwrite`` is False
    - Prompt the user for confirmation if ``overwrite`` is None and ``prompt_user`` is True

    Returns True if it's safe to proceed (path removed or did not exist), False otherwise.
    """

    out_path = Path(output_path)
    if not out_path.exists():
        return True

    # Explicit non-interactive choices
    if overwrite is True:
        try:
            if out_path.is_dir():
                print(f"Removing directory: {out_path}")
                shutil.rmtree(out_path, ignore_errors=True)
                # Double-check removal and retry with more aggressive method if needed
                if out_path.exists():
                    import subprocess
                    subprocess.run(['rm', '-rf', str(out_path)], check=True)
            else:
                print(f"Removing file: {out_path}")
                out_path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to remove existing output path: {exc}") from exc
        return True

    if overwrite is False:
        print(f"Skipping operation (existing output retained at {out_path}).")
        return False

    # Fall back to interactive prompt if allowed
    if prompt_user:
        resp = (
            input(f"Output path exists at {out_path}. Overwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if resp in ("y", "yes"):
            try:
                if out_path.is_dir():
                    shutil.rmtree(out_path, ignore_errors=True)
                    # Double-check removal and retry with more aggressive method if needed
                    if out_path.exists():
                        import subprocess
                        subprocess.run(['rm', '-rf', str(out_path)], check=True)
                else:
                    out_path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Failed to remove existing output path: {exc}"
                ) from exc
            return True
        print("Skipping operation (existing output retained).")
        return False

    # If we are not allowed to prompt and no explicit overwrite was given, be safe and skip.
    print(
        f"Skipping operation: output path {out_path} exists and no overwrite was specified."
    )
    return False


def prompt_overwrite_resume_skip(
    output_path: Union[str, Path],
    *,
    default: str = "R",
) -> str:
    """Prompt user to Overwrite, Resume, or Skip when an output path exists.

    Returns one of: 'overwrite', 'resume' (i.e. skip the zarr store creation), 'skip'. If 'overwrite' is chosen,
    this function deletes the existing path before returning.
    """
    out_path = Path(output_path)
    if not out_path.exists():
        # Nothing to overwrite; default to resume behavior
        return "resume"

    default = (default or "R").strip().upper()
    if default not in {"O", "R", "S"}:
        default = "R"

    while True:
        resp = (
            input(
                f"Output path exists at {out_path}.\n"
                f"Choose an action: (O)verwrite, (R)esume processing, (S)kip step [{default}]: "
            )
            .strip()
            .lower()
        )
        if resp == "":
            resp = default.lower()
        if resp in ("o", "overwrite"):
            try:
                if out_path.is_dir():
                    shutil.rmtree(out_path)
                else:
                    out_path.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to remove existing output path: {exc}"
                ) from exc
            return "overwrite"
        if resp in ("r", "resume"):
            return "resume"
        if resp in ("s", "skip"):
            return "skip"
        print("Invalid choice. Please enter O, R, or S.")


def decide_overwrite_resume_skip(
    output_path: Path, is_debug: bool, expected_positions: list = None, expected_shapes: dict = None,
    expected_channels: list = None,
) -> str:
    """Return 'create' | 'overwrite' | 'resume' | 'skip' using precreation heuristic.

    Behavior:
    - If output_path does not exist, return 'create'.
    - If store is precreated (all zeros), return 'resume' automatically.
    - If store has partial data, prompt user (default: resume).
    - If store structure is incomplete (missing expected positions), overwrite automatically.
    - If expected_shapes provided and existing shapes don't match, overwrite automatically.
    - If expected_channels provided and existing channels don't match, overwrite automatically.
    - On any error, default to 'resume' to avoid blocking automation.

    Args:
        expected_shapes: dict mapping position -> expected (Y, X) shape tuple
        expected_channels: list of expected channel names
    """
    try:
        p = Path(output_path)
        if not p.exists():
            return "create"

        # Check channel mismatch if expected_channels provided
        if expected_channels and p.exists():
            from iohub import open_ome_zarr
            with open_ome_zarr(p, mode="r") as store:
                existing_channels = list(store.channel_names)
                if existing_channels != list(expected_channels):
                    print(
                        f"Channel mismatch: store has {existing_channels}, "
                        f"expected {list(expected_channels)} - will overwrite"
                    )
                    try:
                        if p.is_dir():
                            shutil.rmtree(p)
                        else:
                            p.unlink(missing_ok=True)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to remove existing output path: {exc}"
                        ) from exc
                    return "overwrite"

        # Check shape mismatch if expected_shapes provided
        if expected_shapes and p.exists():
            from iohub import open_ome_zarr
            with open_ome_zarr(p, mode="r") as store:
                for pos, expected_xy in expected_shapes.items():
                    if pos in [p for p, _ in store.positions()]:
                        existing_xy = store[pos]["0"].shape[-2:]
                        if existing_xy != expected_xy:
                            print(f"Shape mismatch: {pos} has {existing_xy}, expected {expected_xy} - will overwrite")
                            return "overwrite"

        # Check store state: True=all zeros, False=missing positions, None=has data
        store_state = is_precreated_store(p, expected_positions=expected_positions)

        if store_state is True:
            # All zeros = precreated, safe to resume
            print(f"Using existing precreated zarr store at {p}")
            return "resume"

        if store_state is False:
            # Missing expected positions = incomplete structure, must overwrite
            print(f"Store structure incomplete - will overwrite")
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to remove existing output path: {exc}"
                ) from exc
            return "overwrite"

        if store_state is None:
            # Has data = partial or complete data
            if is_debug:
                # Debug mode: auto-overwrite without prompting
                print(f"Store has data - auto-overwriting (debug mode)")
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink(missing_ok=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to remove existing output path: {exc}"
                    ) from exc
                return "overwrite"
            else:
                # Check if running in non-interactive batch job (SLURM)
                import os
                in_batch_job = os.environ.get("SLURM_JOB_ID") is not None

                if in_batch_job:
                    # Batch mode: auto-resume without prompting
                    print(f"Store has partial or filled data - auto-resuming (batch mode)")
                    return "resume"
                else:
                    # Interactive mode: prompt user (default: resume)
                    print(f"Store has partial or filled data - prompting for action")
                    default_choice = "R"  # Default to resume for partial work
                    return prompt_overwrite_resume_skip(p, default=default_choice)

        # Fallback: prompt user
        default_choice = "O" if is_debug else "R"
        return prompt_overwrite_resume_skip(p, default=default_choice)
    except Exception:
        return "resume"


def canonicalize_channel_name(name: str) -> str:
    """Normalize raw channel names to canonical keys used in configs.
    BF/brightfield/phase -> BF, GFP -> GFP, mCherry variants -> mCherry.
    """
    if name is None:
        return ""
    n = str(name).strip()
    low = n.lower()
    if low in {"brightfield", "bf", "bright field", "phase", "bf_phase", "phase"}:
        return "BF"
    if low == "gfp":
        return "GFP"
    if low in {"mcherry", "m-cherry", "cherry"}:
        return "mCherry"
    return n


def build_channel_index_map(
    source_channel_names: list[str] | None,
    dest_channel_list: list[str],
) -> list[int | None]:
    """Return a dest_idx -> source_idx mapping by canonicalized names.

    Tries name-based matching first. If no names match (e.g. generic names
    like 'Channel0'), falls back to positional mapping when channel counts
    are equal. Raises ValueError otherwise.
    """
    n_dest = len(dest_channel_list)

    if not source_channel_names:
        raise ValueError(
            "source_channel_names is empty — cannot map channels without names."
        )
    # Try name-based matching first
    canon_to_src: dict[str, int] = {}
    for i, nm in enumerate(source_channel_names):
        key = canonicalize_channel_name(nm)
        if key and key not in canon_to_src:
            canon_to_src[key] = i
    mapping: list[int | None] = []
    for dnm in dest_channel_list:
        key = canonicalize_channel_name(dnm)
        mapping.append(canon_to_src.get(key))

    # If name-based matching found nothing (generic names like Channel0),
    # use positional mapping but only when counts match exactly.
    if all(idx is None for idx in mapping):
        n_src = len(source_channel_names)
        n_dst = len(dest_channel_list)
        if n_src != n_dst:
            raise ValueError(
                f"Cannot map channels: no name matches between source "
                f"{source_channel_names} and destination {dest_channel_list}, "
                f"and counts differ ({n_src} vs {n_dst})."
            )
        mapping = list(range(n_dst))

    return mapping


def get_experiment_wells(experiment: str, prefix_only: bool = False) -> list[str]:
    """Read wells_to_process from experiment config YAML.

    Args:
        experiment: Experiment name (e.g. "ops0050_20250630")
        prefix_only: If True, return well prefixes like "A/1" instead of "A/1/0"

    Returns:
        List of well strings. Falls back to ["A/1/0", "A/2/0", "A/3/0"] if config not found.
    """
    from cyclops_utils.data.experiment import OpsDataset

    default = ["A/1/0", "A/2/0", "A/3/0"]
    try:
        dataset = OpsDataset(experiment)
        cfg_path = dataset.config_paths.get("exp_config")
        if cfg_path and Path(cfg_path).exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            wells = cfg.get("wells_to_process", [])
            if wells:
                if prefix_only:
                    return ["/".join(w.split("/")[:2]) for w in wells]
                return wells
    except Exception:
        pass
    if prefix_only:
        return ["/".join(w.split("/")[:2]) for w in default]
    return default


def setup_experiment_directories(experiment: str):
    """
    Creates the necessary directory structure for a new experiment based on the
    paths defined in the OpsDataset class.
    """
    from cyclops_utils.data.experiment import OpsDataset

    dataset = OpsDataset(experiment)

    path_sources = [
        dataset.store_paths.values(),
        dataset.result_paths.values(),
        dataset.metrics_paths.values(),
    ]

    all_dirs = set()
    for source_list in path_sources:
        for path in source_list:
            all_dirs.add(path.parent)

    for directory in sorted(list(all_dirs)):
        directory.mkdir(parents=True, exist_ok=True)
