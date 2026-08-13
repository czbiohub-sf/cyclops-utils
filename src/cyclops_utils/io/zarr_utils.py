from collections import defaultdict
from pathlib import Path
from iohub import open_ome_zarr
import dask.array as da
import numpy as np
import random
from typing import Tuple, List, Dict, Any
from iohub.ngff import TransformationMeta

import os
import json


def has_fluorescence_channels_from_config(config: dict) -> bool:
    """
    Determine presence of live-cell fluorescence channels from the config.

    Heuristic: any channel label that is not None/empty and not one of
    {"Phase", "BF", "Brightfield"} implies fluorescence. Channels flagged as
    fixed (cell-painting / 4i panels, listed in ``fixed_channels``) are ignored
    — they aren't part of the live-cell acquisition, so an experiment whose only
    non-fixed channel is BF is phase-only here.
    """
    try:
        ch_map = (config.get("channel_map") or {}) if isinstance(config, dict) else {}
        if not isinstance(ch_map, dict):
            return False
        fixed = set(config.get("fixed_channels") or []) if isinstance(config, dict) else set()
        for name, label in ch_map.items():
            if name in fixed or label is None:
                continue
            label_str = str(label).strip().lower()
            if label_str and label_str not in {"phase", "bf", "brightfield"}:
                return True
        return False
    except Exception:
        return False


def _group_shifts_by_position(all_shifts: dict) -> defaultdict:
    """Group shifts by position for stitching."""

    def get_group(key):
        # Group by full row/col so wells in different rows (A/1 vs B/1) stay distinct.
        return "/".join(key.split("/")[:2])

    grouped_shifts = defaultdict(dict)
    for key, value in all_shifts.items():
        group = get_group(key)
        grouped_shifts[group][key] = value
    return grouped_shifts


def _discover_positions(
    source_path: Path, use_shifts: bool = False, all_shifts: dict = None
) -> list:
    """Discover positions in source store."""
    if use_shifts and all_shifts:
        return list(all_shifts.keys())
    else:
        # Fast discovery using glob
        position_paths = sorted(source_path.glob("*/*/*"))
        from tqdm import tqdm

        position_paths = [
            p
            for p in tqdm(position_paths, desc="Discovering positions")
            if (p / "0").is_dir()
        ]
        if not position_paths:
            raise ValueError(f"No positions found in {source_path}")
        return [str(p.relative_to(source_path)) for p in position_paths]


def _iter_position_paths(source_store: Path | str) -> list[str]:
    try:
        with open_ome_zarr(source_store, mode="r") as store:
            return [p for p, _ in store.positions()]
    except Exception:
        return []


def _create_overlapping_subtile_bounds(
    Y: int, X: int, grid_size: int, blend_pixels: int
) -> List[Tuple[int, int, int, int]]:
    """Calculate overlapping subtile bounds for cross-boundary blending.

    Args:
        Y, X: Full image dimensions
        grid_size: Grid dimension (e.g., 3 for 3x3)
        blend_pixels: Overlap width in pixels

    Returns:
        List of (y_start, y_end, x_start, x_end) tuples with overlaps
    """
    base_height = Y // grid_size
    base_width = X // grid_size

    bounds = []
    for row in range(grid_size):
        for col in range(grid_size):
            # Base bounds without overlap
            y_start = row * base_height
            y_end = (row + 1) * base_height if row < grid_size - 1 else Y
            x_start = col * base_width
            x_end = (col + 1) * base_width if col < grid_size - 1 else X

            # Add overlaps
            if row > 0:  # Not top row
                y_start -= blend_pixels
            if row < grid_size - 1:  # Not bottom row
                y_end += blend_pixels
            if col > 0:  # Not left column
                x_start -= blend_pixels
            if col < grid_size - 1:  # Not right column
                x_end += blend_pixels

            # Clamp to image bounds
            y_start = max(0, y_start)
            y_end = min(Y, y_end)
            x_start = max(0, x_start)
            x_end = min(X, x_end)

            bounds.append((y_start, y_end, x_start, x_end))

    return bounds


def _validate_subtile_grid(n_subtiles: int, Y: int, X: int) -> int:
    """Validate that n_subtiles is a perfect square and the grid size is a factor of image dimensions."""
    sqrt_n = int(np.sqrt(n_subtiles))
    if sqrt_n * sqrt_n != n_subtiles:
        raise ValueError(
            f"n_subtiles ({n_subtiles}) must be a perfect square (4, 9, 16, 25, etc.)"
        )

    grid_size = sqrt_n
    if Y % grid_size != 0 or X % grid_size != 0:
        raise ValueError(
            f"grid_size ({grid_size}) derived from n_subtiles ({n_subtiles}) must be a factor of image dimensions ({Y}x{X})"
        )

    return grid_size


def _validate_output_images(store_path: Path, n_samples: int = 3, raise_on_blank: bool = False):
    """Check a sample of images in the output store to ensure they are not empty.

    Args:
        store_path: Path to zarr store to validate
        n_samples: Number of positions to sample
        raise_on_blank: If True, raise RuntimeError if any blank images are found
    """
    print(
        f"\n[Validation] Checking up to {n_samples} balanced-sampled images in {store_path.name}..."
    )
    try:
        with open_ome_zarr(store_path, mode="r") as ds:
            # Use fast balanced discovery to avoid enumerating all positions
            positions = _discover_positions_fast_balanced(store_path, int(n_samples))
            if not positions:
                print("[Validation] WARNING: No positions found in the output store.")
                return

            num_to_check = min(n_samples, len(positions))
            # Take the first N from the balanced selection
            sampled_positions = positions[:num_to_check]

            failed_positions = []
            for pos in sampled_positions:
                try:
                    # Check all timepoints for this position
                    # A position is only considered failed if ALL timepoints are empty
                    pos_data = ds[pos]["0"]
                    num_timepoints = pos_data.shape[0]
                    
                    # Sample at least one slice from each timepoint
                    all_empty = True
                    for t in range(num_timepoints):
                        image_data = pos_data[t, 0, 0, :, :]
                        if np.any(image_data):
                            all_empty = False
                            break
                    
                    if all_empty:
                        failed_positions.append(f"{pos} (all {num_timepoints} timepoints empty)")
                except Exception as e:
                    print(
                        f"[Validation] WARNING: Could not read or check position {pos}: {e}"
                    )

            if failed_positions:
                print("\n" + "=" * 80)
                print("!!! VALIDATION WARNING !!!")
                print(
                    f"{len(failed_positions)}/{num_to_check} sampled images were found to be all zeros (black)."
                )
                print(
                    "This may indicate a problem with the reconstruction or slice extraction."
                )
                print("Problematic samples:")
                for p in failed_positions:
                    print(f"  - {p}")
                print("=" * 80 + "\n")

                if raise_on_blank:
                    raise RuntimeError(
                        f"VALIDATION FAILED: {len(failed_positions)}/{num_to_check} sampled positions "
                        f"contain all zeros (blank images). Check the output store and re-run if needed."
                    )
            else:
                print(
                    f"[Validation] OK: {num_to_check}/{num_to_check} sampled images contain data."
                )

    except Exception as e:
        print(
            f"\n[Validation] ERROR: Could not perform validation check on {store_path.name}: {e}"
        )


def _validate_all_positions_have_data(
    store_path: Path,
    expected_positions: list[str],
    corner_size: int = 64,
    skip_timepoints: list[int] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Check that ALL expected positions exist and have non-zero data at every timepoint.

    Reads a small corner sample from each position at each timepoint.

    Args:
        store_path: Path to zarr store to validate
        expected_positions: List of position paths expected to have data
        corner_size: Size of the corner sample to read (default 64x64)
        skip_timepoints: List of timepoint indices to skip validation for
            (e.g., [0] to skip t=0 for wells where it's legitimately empty)

    Returns:
        Tuple of (is_complete, missing_positions, empty_positions)
        empty_positions entries are formatted as "pos (t=N)" when a specific
        timepoint is empty.
    """
    missing_positions = []
    empty_positions = []
    _skip = set(skip_timepoints or [])

    with open_ome_zarr(store_path, mode="r") as ds:
        for pos in expected_positions:
            try:
                pos_data = ds[pos]["0"]
            except KeyError:
                missing_positions.append(pos)
                continue

            try:
                cs = min(corner_size, pos_data.shape[-1], pos_data.shape[-2])
                T = pos_data.shape[0]
                for t in range(T):
                    if t in _skip:
                        continue
                    sample = pos_data[t, 0, 0, :cs, :cs]
                    if not np.any(sample):
                        empty_positions.append(f"{pos} (t={t})")
            except Exception:
                empty_positions.append(pos)

    total_incomplete = len(missing_positions) + len(empty_positions)
    is_complete = total_incomplete == 0

    skip_msg = f", skipping t={sorted(_skip)}" if _skip else ""
    print(
        f"[Validation] Checked {len(expected_positions)} positions{skip_msg}: "
        f"{len(expected_positions) - len(missing_positions) - len(set(p.split(' (t=')[0] for p in empty_positions))} complete, "
        f"{len(missing_positions)} missing, {len(empty_positions)} empty timepoints"
    )

    return is_complete, missing_positions, empty_positions


def level_has_data(
    level_path: Path, check_pixels: bool = False, level_index: int = 0
) -> bool:
    """Check that a pyramid level directory contains chunk data.

    Two modes:
    - Filesystem-only (default): Checks the directory is non-empty (has children
      like c/, zarr.json, .zarray, or chunk files). Fast enough to call hundreds
      of times during an audit without reading any array data.
    - Pixel check (check_pixels=True): Also reads sample patches to verify
      non-zero pixel values exist. Catches precreated/zeroed-out levels.
      Sampling adapts to pyramid level: level 0 checks a single center patch,
      higher levels sample multiple locations (center + corners) since data is
      sparser but arrays are smaller so broader sampling is cheap.

    Unlike _validate_output_images / _validate_all_positions_have_data which open
    the full zarr store via iohub, this operates on individual level paths and
    doesn't require the parent store to be opened.

    Args:
        level_path: Path to a pyramid level directory (e.g., store/A/1/0/2/)
        check_pixels: If True, also verify non-zero pixel data (slower but thorough)
        level_index: Pyramid level (0=full res). Higher levels use broader sampling
            to account for sparsity in downsampled data.

    Returns:
        True if the level exists, is non-empty, and (optionally) has non-zero pixels.
    """
    if not level_path.exists():
        return False
    try:
        if not any(level_path.iterdir()):
            return False
    except Exception:
        return False

    # zarr v2: .zarray required. zarr v3: zarr.json required with sharding codec.
    zarray = level_path / ".zarray"
    zarr_json = level_path / "zarr.json"
    if zarray.exists():
        pass  # v2 — metadata present
    elif zarr_json.exists():
        # v3 — verify sharding codec is declared (catches unsharded leftovers)
        try:
            import json as _json
            with open(zarr_json) as _f:
                _meta = _json.load(_f)
            _codecs = _meta.get("codecs", [])
            if not any(c.get("name") == "sharding_indexed" for c in _codecs):
                return False
        except Exception:
            return False
    else:
        return False

    if not check_pixels:
        return True

    try:
        import dask.array as da

        arr = da.from_zarr(str(level_path))
        h, w = arr.shape[-2], arr.shape[-1]

        # Sample a 5x5 grid across the full image extent. Smaller patch at level 0
        # (array is large), bigger at higher levels (array is smaller).
        # Grid sampling is robust for sparse data (segmentation, sparse fluorescence)
        # where cells/signal may be off-center. Short-circuits on first non-zero hit.
        ps = min(64 if level_index == 0 else 128, w, h)
        n = 5
        sample_points = [
            (int(h * (i + 0.5) / n), int(w * (j + 0.5) / n))
            for i in range(n) for j in range(n)
        ]
        n_channels = arr.shape[1] if arr.ndim >= 4 else 1
        # Each channel must have at least one non-zero sample point.
        # This catches cases where some channels have data but others are empty
        # (e.g. Phase2D has data but GFP pyramids are zeros).
        channel_has_data = [False] * n_channels
        for cy, cx in sample_points:
            y0 = max(0, cy - ps // 2)
            y1 = min(h, y0 + ps)
            x0 = max(0, cx - ps // 2)
            x1 = min(w, x0 + ps)
            for c in range(n_channels):
                if channel_has_data[c]:
                    continue  # already confirmed this channel
                if arr.ndim == 5:
                    sample = arr[0, c, 0, y0:y1, x0:x1].compute()
                elif arr.ndim == 4:
                    sample = arr[0, c, y0:y1, x0:x1].compute()
                else:
                    sample = arr[y0:y1, x0:x1].compute()
                if bool(np.any(sample)):
                    channel_has_data[c] = True
            if all(channel_has_data):
                return True  # all channels confirmed — short-circuit
        return all(channel_has_data)
    except Exception:
        # If we can't read pixels, trust the filesystem check
        return True


def is_precreated_store(
    store_path: Path, n_samples: int = 1, expected_positions: list = None
) -> bool | None:
    """Check store state: precreated (all zeros), incomplete structure, or has data.

    Returns:
        True: Positions exist and are all zeros (precreated, auto-resume)
        False: Expected positions missing (incomplete structure, auto-overwrite)
        None: Positions exist with data (partial or complete work, prompt user with default=resume)

    Args:
        expected_positions: List of expected position paths. If provided, samples and
                           validates first, middle, and last positions exist.
                           Returns False if any are missing (incomplete structure).
    """
    try:
        # Determine which positions to check
        if expected_positions:
            # Sample expected positions (first, middle, last) for validation
            sample_indices = {
                0,
                len(expected_positions) // 2,
                len(expected_positions) - 1,
            }
            positions_to_check = [
                expected_positions[i]
                for i in sorted(sample_indices)
                if i < len(expected_positions)
            ]

            # First verify all sampled positions exist
            with open_ome_zarr(store_path, mode="r") as ds:
                for pos in positions_to_check:
                    try:
                        _ = ds[pos]  # Try to access position
                    except (KeyError, IndexError):
                        print(
                            f"⚠️  Expected position {pos} not found (incomplete store)"
                        )
                        return False
        else:
            # Discover last position in each well
            positions_to_check = _discover_last_position_per_well(store_path)

        if not positions_to_check:
            return False

        # Check if positions are all zeros (precreated)
        with open_ome_zarr(store_path, mode="r") as ds:
            for pos in positions_to_check:
                try:
                    comp = str((Path(pos) / "0"))
                    arr = da.from_zarr(str(store_path), component=comp)
                    # Use a tiny window from the first chunk to avoid reading full planes
                    try:
                        y_chunk = int(
                            getattr(arr, "chunks", ((), (), (), (256,), (256,)))[-2][0]
                        )
                        x_chunk = int(
                            getattr(arr, "chunks", ((), (), (), (256,), (256,)))[-1][0]
                        )
                    except Exception:
                        y_chunk, x_chunk = 256, 256
                    cy = max(1, min(64, int(arr.shape[-2]), y_chunk))
                    cx = max(1, min(64, int(arr.shape[-1]), x_chunk))
                    if arr.ndim >= 5:
                        sample = arr[0, 0, 0, 0:cy, 0:cx].compute()
                    elif arr.ndim == 4:
                        sample = arr[0, 0, 0:cy, 0:cx].compute()
                    elif arr.ndim == 3:
                        sample = arr[0, 0:cy, 0:cx].compute()
                    else:
                        sample = arr[0:cy, 0:cx].compute()
                    if np.any(sample):
                        # Has data = partial or complete work, return None
                        return None
                except Exception:
                    return False
            # All sampled positions are zeros = precreated
            return True
    except Exception:
        return False


def _list_sorted_dirnames(p: Path) -> list[str]:
    try:
        return sorted([d.name for d in p.iterdir() if d.is_dir()])
    except Exception:
        return []


def list_wells(store_path: Path) -> list[str]:
    """List wells in an OME-Zarr store from its directory structure.

    Returns well identifiers like ["A/1", "A/2"] by scanning
    row directories (A, B, ...) then column directories (1, 2, ...).
    Fast filesystem-only operation — does not open the zarr store.
    """
    wells = []
    for row in _list_sorted_dirnames(store_path):
        if row.startswith("."):
            continue
        for col in _list_sorted_dirnames(store_path / row):
            if col.startswith("."):
                continue
            wells.append(f"{row}/{col}")
    return wells


def _discover_last_position_per_well(path: Path) -> list[str]:
    """Quickly discover the last position in each well.

    Strategy:
    - List all wells by scanning row directories (e.g., 'A','B',...) then column directories ('1','2',...)
    - For each well, find the last tile (highest numeric/lexicographic value)
    - Returns one position per well found
    - This ensures we validate that ALL wells are complete, not just a few positions
    """
    rows = _list_sorted_dirnames(path)
    if not rows:
        return []

    positions: list[str] = []
    for r in rows:
        cols = _list_sorted_dirnames(path / r)
        for c in cols:
            well_dir = path / r / c
            tiles = _list_sorted_dirnames(well_dir)
            if not tiles:
                continue

            # Sort by numeric component when possible
            try:
                tiles_sorted = sorted(
                    tiles,
                    key=lambda t: int("".join(ch for ch in t if ch.isdigit()) or "0"),
                )
            except Exception:
                tiles_sorted = sorted(tiles)

            # Take the last tile from this well
            last_tile = tiles_sorted[-1]
            positions.append(f"{r}/{c}/{last_tile}")

    return positions


def _discover_positions_fast_balanced(path: Path, debug_n_positions: int) -> list[str]:
    """Quickly select a balanced set of positions without enumerating all fields.

    Strategy:
    - List wells by scanning row directories (e.g., 'A','B',...) then column directories ('1','2',...).
    - Choose a central block of wells of size K (or as many as available if fewer).
    - For each chosen well, list its tile directories and pick the center tile by numeric order.
    This avoids traversing all tiles across all wells.
    """
    if debug_n_positions is None or debug_n_positions <= 0:
        return []

    rows = _list_sorted_dirnames(path)
    wells: list[tuple[str, str]] = []
    for r in rows:
        cols = _list_sorted_dirnames(path / r)
        for c in cols:
            wells.append((r, c))

    if not wells:
        return []

    # Choose a centered block of wells
    K = int(debug_n_positions)
    total_wells = len(wells)
    if K >= total_wells:
        selected_wells = wells
    else:
        mid = total_wells // 2
        start = max(0, mid - (K // 2))
        end = min(total_wells, start + K)
        selected_wells = wells[start:end]

    # For each selected well, pick center tiles and then expand around center until reaching the requested count
    positions: list[str] = []
    if not selected_wells:
        return positions

    # Determine how many tiles to sample per well (top-up strategy)
    per_well = max(
        1, (K + len(selected_wells) - 1) // len(selected_wells)
    )  # ceil(K / num_wells)

    for r, c in selected_wells:
        if len(positions) >= K:
            break
        well_dir = path / r / c
        tiles = _list_sorted_dirnames(well_dir)
        if not tiles:
            continue
        # Sort by numeric component when possible
        try:
            tiles_sorted = sorted(
                tiles, key=lambda t: int("".join(ch for ch in t if ch.isdigit()) or "0")
            )
        except Exception:
            tiles_sorted = tiles

        n = len(tiles_sorted)
        mid = n // 2
        # Spiral out from center: mid, mid-1, mid+1, mid-2, mid+2, ...
        picked = 0
        step = 0
        used_indices: set[int] = set()
        while picked < per_well and len(positions) < K and step < n:
            for delta in (0, -1, 1):
                if delta == 0 and step > 0:
                    continue
                idx = mid + (step if delta == 1 else (-step if delta == -1 else 0))
                if 0 <= idx < n and idx not in used_indices:
                    positions.append(f"{r}/{c}/{tiles_sorted[idx]}")
                    used_indices.add(idx)
                    picked += 1
                    if picked >= per_well or len(positions) >= K:
                        break
            step += 1

    return positions


def _maybe_sample_positions(
    positions: list[str], debug_n_positions: int | None
) -> list[str]:
    if debug_n_positions is None or debug_n_positions <= 0:
        return positions

    # Group by well (first two path parts), sort tiles per well, and pick a centered block per well
    try:
        well_to_tiles: dict[str, list[tuple[int, str]]] = {}
        for p in positions:
            parts = Path(p).parts
            if len(parts) >= 3:
                well = f"{parts[0]}/{parts[1]}"
                tile_str = parts[2]
            else:
                well = "_"
                tile_str = parts[-1]
            # Parse numeric tile index; tolerate leading zeros
            digits = "".join(ch for ch in tile_str if ch.isdigit())
            tile_idx = int(digits) if digits else 0
            well_to_tiles.setdefault(well, []).append((tile_idx, p))

        wells = sorted(well_to_tiles.keys())
        if not wells:
            return positions[: int(debug_n_positions)]

        total_target = int(debug_n_positions)
        per_well_base = max(1, total_target // len(wells))
        remainder = max(0, total_target - per_well_base * len(wells))

        selected: list[str] = []
        for i, well in enumerate(wells):
            items = sorted(well_to_tiles[well], key=lambda t: t[0])
            n = len(items)
            if n == 0:
                continue
            k = per_well_base + (1 if i < remainder else 0)
            k = min(k, n)
            mid = n // 2
            start = max(0, mid - (k // 2))
            end = min(n, start + k)
            block = items[start:end]
            selected.extend([pp for _, pp in block])

        # If we still have fewer than requested (e.g., very small wells), top up from centers again
        if len(selected) < total_target:
            for well in wells:
                if len(selected) >= total_target:
                    break
                items = sorted(well_to_tiles[well], key=lambda t: t[0])
                if not items:
                    continue
                mid = len(items) // 2
                candidate = items[mid][1]
                if candidate not in selected:
                    selected.append(candidate)

        print(
            f"DEBUG: Sampling {min(total_target, len(selected))} centered positions across {len(wells)} wells."
        )
        return selected[:total_target]
    except Exception:
        # Fallback: first N positions
        print(
            f"DEBUG: Sampling {debug_n_positions} positions (fallback head selection)."
        )
        return positions[: int(debug_n_positions)]


def _resolve_output_path_for_debug(
    output_path: Path, debug_n_positions: int | None, debug_output_suffix: str
) -> Path:
    if debug_n_positions is not None and debug_n_positions > 0:
        return output_path.with_name(
            f"{output_path.stem}{debug_output_suffix}{output_path.suffix}"
        )
    return output_path


def _shape_key_from_shape(shape: Tuple[int, int, int, int, int]) -> str:
    t, c, z, y, x = shape
    return f"T{t}C{c}Z{z}Y{y}X{x}"


def _get_or_create_reusable_input_position(
    base_temp_dir: Path,
    shape: Tuple[int, int, int, int, int],
    position_scale,
    input_channel_names: List[str],
    original_metadata: dict | None = None,
) -> Path:
    """Create or reuse a small HCS store for a given (T,C,Z,Y,X) shape and return its position path."""
    pool_dir = base_temp_dir / "input_pool"
    pool_dir.mkdir(exist_ok=True)
    key = _shape_key_from_shape(shape)
    store_path = pool_dir / f"input_{key}.zarr"

    if not store_path.exists():
        with open_ome_zarr(
            store_path,
            layout="hcs",
            mode="w-",
            channel_names=(original_metadata or {}).get(
                "channel_names", input_channel_names
            ),
        ) as input_store:
            try:
                plate_zattrs = (original_metadata or {}).get("plate_zattrs")
                if isinstance(plate_zattrs, dict) and plate_zattrs:
                    z = dict(input_store.zattrs)
                    z.update(plate_zattrs)
                    input_store.zattrs.clear()
                    input_store.zattrs.update(z)
            except Exception:
                pass

            input_pos = input_store.create_position("0", "0", "0")

            try:
                omero_meta = input_pos.zattrs.get("omero", {})
                omero_meta["channels"] = [
                    {"label": name, "color": "FFFFFF", "active": True}
                    for name in (original_metadata or {}).get(
                        "channel_names", input_channel_names
                    )
                ]
                input_pos.zattrs["omero"] = omero_meta
            except Exception:
                pass

            try:
                pos_zattrs = (original_metadata or {}).get("position_zattrs")
                if isinstance(pos_zattrs, dict) and pos_zattrs:
                    z = dict(input_pos.zattrs)
                    z.update(pos_zattrs)
                    input_pos.zattrs.clear()
                    input_pos.zattrs.update(z)
            except Exception:
                pass

            input_pos.create_zeros(
                name="0",
                shape=shape,
                dtype=np.float32,
                transform=[
                    TransformationMeta(
                        type="scale",
                        scale=(original_metadata or {}).get(
                            "position_scale", position_scale
                        ),
                    )
                ],
            )

    return store_path / "0" / "0" / "0"


def _ensure_store_position(
    store_path: Path,
    pos: str,
    T: int,
    Y: int,
    X: int,
    position_scale,
    chunk_size,
    channel_names: List[str],
    num_channels: int = 2,
) -> None:
    """Ensure the given store and position exist with (T, C=num_channels, Z=1)."""
    # Short-circuit: if the position array already exists, skip expensive iohub plate
    # metadata parsing (iohub 0.3.x re-parses all N position metadata on every open call)
    if (store_path / pos / "0").exists():
        return
    # Create store if missing
    if not store_path.exists():
        with open_ome_zarr(
            store_path, layout="hcs", mode="w-", channel_names=channel_names
        ):
            pass
    # Ensure position exists
    with open_ome_zarr(store_path, mode="r+") as store:
        try:
            pos_node = store.create_position(*Path(pos).parts)
            pos_node.create_zeros(
                name="0",
                shape=(int(T), int(num_channels), 1, Y, X),
                dtype=np.float32,
                chunks=chunk_size,
                transform=[TransformationMeta(type="scale", scale=position_scale)],
            )
        except (FileExistsError, ValueError):
            # Already exists
            pass


def _write_plane_to_store(
    store_path: Path, pos: str, plane: np.ndarray, channel_index: int, time_index: int
) -> None:
    """Write a (Y, X) plane into channel `channel_index` at [time_index,channel,0,:,:]."""
    if plane.ndim != 2:
        raise ValueError("plane must be 2D (Y, X)")
    # Use direct zarr access to avoid iohub plate metadata parsing overhead
    # (iohub 0.3.x re-parses all N position metadata on every open_ome_zarr call)
    import zarr
    arr = zarr.open(str(store_path / pos / "0"), mode="r+")
    arr[int(time_index), int(channel_index), 0, :, :] = plane.astype(np.float32, copy=True)


from typing import Optional


def _infer_channel_axis_from_store(source_store: Path | str) -> Optional[int]:
    """
    Infer channel axis from OME-Zarr metadata by matching the number of channel
    names to one of the array dimensions. Falls back to common OME order.
    Returns None for single-channel or if ambiguous.
    """
    with open_ome_zarr(source_store, mode="r") as store:
        channel_names = getattr(store, "channel_names", None)
        if not channel_names:
            # No channel metadata available
            pos_path = next(store.positions())[0]
            shape = store[pos_path].data.shape
            # Heuristic: assume OME (T, C, Z, Y, X)
            return 1 if len(shape) >= 3 and len(shape) <= 5 else None
        n_channels = len(channel_names)
        pos_path = next(store.positions())[0]
        shape = store[pos_path].data.shape
        candidates = [i for i, s in enumerate(shape) if s == n_channels]
        if len(candidates) == 1:
            return candidates[0]
        # Prefer axis 1 for typical OME (T, C, Z, Y, X)
        if 1 in candidates:
            return 1
        # If ambiguous or no match, return None
        return None


# -----------------------------
# Zarr v2/v3 compatibility helpers
# -----------------------------
def create_zarr_array(
    path: str,
    shape: tuple,
    chunks: tuple,
    dtype,
    zarr_format: int = 3,
    shards_ratio: tuple = None,
    fill_value=0,
    overwrite: bool = True,
):
    """
    Create a zarr array with optional sharding for v3 format.

    This helper centralizes zarr array creation logic to ensure consistent
    sharding across all array types (images, segmentations, overlays).

    Args:
        path: Full path to the zarr array (store_path/component)
        shape: Shape of the array
        chunks: Chunk size tuple
        dtype: Data type for the array
        zarr_format: Zarr format version (2 or 3, default: 3)
        shards_ratio: Sharding ratio for v3 format. If None, uses default based on array shape:
                      - 5D arrays: (1, 1, 1, 64, 64)
                      - 4D arrays: (1, 1, 64, 64)
                      - 3D arrays (H, W, C): (64, 64, 1) - only shard spatial dims
                      - 2D arrays: (64, 64)
                      Set to (1, 1, 1) to disable sharding (creates one file per chunk).
                      For small arrays where shards would exceed shape, sharding is disabled.
        fill_value: Fill value for the array (default: 0)
        overwrite: Whether to overwrite existing array (default: True)

    Returns:
        zarr.Array: The created/opened zarr array

    Example:
        # Create a 5D segmentation array with sharding
        arr = create_zarr_array(
            path="/data/store.zarr/A/1/0/labels/seg/1",
            shape=(1, 1, 1, 52224, 52224),
            chunks=(1, 1, 1, 2048, 2048),
            dtype=np.int32,
            zarr_format=3,
        )

        # Create a 3D RGBA overlay array with sharding
        arr = create_zarr_array(
            path="/data/store.zarr/A/1/0/labels/iss_gene_image/0",
            shape=(104448, 104448, 4),
            chunks=(1024, 1024, 4),
            dtype=np.uint8,
            zarr_format=3,
        )
    """
    import zarr
    from pathlib import Path

    path = Path(path)
    array_name = path.name
    parent_path = path.parent

    if zarr_format == 3:
        # Determine default shards_ratio based on array dimensionality
        if shards_ratio is None:
            ndim = len(shape)
            if ndim >= 5:
                shards_ratio = (1, 1, 1, 64, 64)
            elif ndim == 4:
                shards_ratio = (1, 1, 64, 64)
            elif ndim == 3:
                # For 3D arrays like (H, W, 4) RGBA, only shard spatial dims
                shards_ratio = (64, 64, 1)
            else:
                shards_ratio = (64, 64)

        # Ensure shards_ratio matches array dimensions
        if len(shards_ratio) != len(chunks):
            if len(chunks) > len(shards_ratio):
                # Pad with 1s at the beginning
                shards_ratio = (1,) * (len(chunks) - len(shards_ratio)) + shards_ratio
            else:
                # Take last N elements
                shards_ratio = shards_ratio[-len(chunks):]

        # Calculate shards from chunks and ratio
        shards = tuple(c * r for c, r in zip(chunks, shards_ratio))

        # Check if sharding is disabled (all ratios are 1)
        # shards_ratio=(1,1,1) means "don't shard" - each chunk would be its own shard
        sharding_disabled = all(r == 1 for r in shards_ratio)

        # Clamp shard sizes to array dimensions (for small pyramid levels)
        # This creates a single shard when array is smaller than requested shard size.
        # IMPORTANT: Shard dimensions must be multiples of chunk dimensions, so we
        # round up to the nearest chunk boundary after clamping.
        def clamp_shard(shard_size, array_size, chunk_size):
            if shard_size <= array_size:
                return shard_size  # No clamping needed
            # Clamp to array size, but round up to nearest chunk multiple
            # This ensures shard % chunk == 0 (required by zarr sharding codec)
            n_chunks = (array_size + chunk_size - 1) // chunk_size  # ceil division
            return n_chunks * chunk_size

        shards = tuple(
            clamp_shard(s, sh, c) for s, sh, c in zip(shards, shape, chunks)
        )

        # Open or create parent group, then create array with optional sharding
        # This is the correct zarr v3 API - group.create_array() supports shards parameter
        parent_group = zarr.open_group(str(parent_path), mode="a", zarr_format=zarr_format)

        if sharding_disabled:
            # Don't pass shards parameter - creates unsharded array (one file per chunk)
            return parent_group.create_array(
                name=array_name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                overwrite=overwrite,
            )
        else:
            # Enable sharding - groups multiple chunks into shard files
            # For small arrays, shards are clamped to array size (single shard)
            return parent_group.create_array(
                name=array_name,
                shape=shape,
                chunks=chunks,
                shards=shards,
                dtype=dtype,
                fill_value=fill_value,
                overwrite=overwrite,
            )
    else:
        # v2 format - no sharding support, use open_array directly
        mode = "w" if overwrite else "a"
        return zarr.open_array(
            str(path),
            mode=mode,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            zarr_version=2,  # Explicitly force v2 when using zarr 3.x library
        )


def detect_zarr_format(store_path: Path) -> int:
    """
    Detect whether a zarr store is v2 or v3.

    Zarr v2 stores have .zgroup files at each group level.
    Zarr v3 stores have zarr.json files instead.

    Parameters
    ----------
    store_path : Path
        Path to the zarr store root

    Returns
    -------
    int
        2 for zarr v2, 3 for zarr v3
    """
    store_path = Path(store_path)
    # Check root level first
    if (store_path / "zarr.json").exists():
        return 3
    if (store_path / ".zgroup").exists():
        return 2
    # Check one level deeper (in case root has mixed content)
    try:
        for child in store_path.iterdir():
            if child.is_dir():
                if (child / "zarr.json").exists():
                    return 3
                if (child / ".zgroup").exists():
                    return 2
    except Exception:
        pass
    # Default to v2 for backwards compatibility
    return 2


def has_zarr_array_metadata(array_path: Path, zarr_format: int = None) -> bool:
    """
    Check if a zarr array has valid metadata.

    For zarr v2: checks for .zarray file
    For zarr v3: checks for zarr.json file

    Parameters
    ----------
    array_path : Path
        Path to the zarr array directory
    zarr_format : int, optional
        If provided, checks for format-specific metadata.
        If None, checks for either format.

    Returns
    -------
    bool
        True if valid metadata exists
    """
    array_path = Path(array_path)
    if zarr_format is None:
        # Auto-detect from the path
        if (array_path / "zarr.json").exists():
            return True
        if (array_path / ".zarray").exists():
            return True
        return False

    if zarr_format == 3:
        return (array_path / "zarr.json").exists()
    else:
        return (array_path / ".zarray").exists()


def _read_component_attrs(component_dir: Path) -> dict:
    """Read attributes from a zarr component (v2 or v3 format).

    For v2: reads from .zattrs file at root level
    For v3: reads from zarr.json → attributes → custom_metadata

    Returns attributes dict (empty if not found).
    """
    # Try v3 first (zarr.json with attributes)
    zarr_json = component_dir / "zarr.json"
    if zarr_json.exists():
        try:
            with open(zarr_json, "r") as f:
                data = json.load(f)
            # V3 structure: return all attributes (includes both clims_per_level and custom_metadata)
            attrs = data.get("attributes", {})
            # Merge custom_metadata into top-level attrs for backward compatibility
            custom_meta = attrs.get("custom_metadata", {})
            if custom_meta and isinstance(custom_meta, dict):
                # Create merged dict: custom_metadata values at top level, other attrs preserved
                merged = dict(custom_meta)
                for k, v in attrs.items():
                    if k != "custom_metadata":
                        merged[k] = v
                return merged
            # No custom_metadata: return all attributes as-is
            return attrs
        except Exception:
            pass

    # Try v2 (.zattrs file)
    zattrs = component_dir / ".zattrs"
    if zattrs.exists():
        try:
            with open(zattrs, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {}


def read_per_level_clims(source_store: Path, pos: str, level_names: list[str], c_dim: int) -> tuple:
    """Read per-level clims metadata from zarr store (v2 or v3 format).

    For v3: reads clims_per_level dict from position-level zarr.json
    For v2: reads from each pyramid level's .zattrs file

    Returns:
        tuple: (per_level_clims, per_level_clims_per_channel, gamma_per_channel)
    """
    per_level_clims: list = []
    per_level_clims_per_channel: list = []
    gamma_per_channel: list = []

    try:
        # First check for v3-style clims_per_level dict at position level
        pos_dir = Path(source_store) / pos
        # print(f"[DEBUG-READ-CLIMS] pos_dir={pos_dir}")
        pos_attrs = _read_component_attrs(pos_dir)
        # print(f"[DEBUG-READ-CLIMS] pos_attrs keys={list(pos_attrs.keys())}")
        clims_per_level_dict = pos_attrs.get("clims_per_level")
        # print(f"[DEBUG-READ-CLIMS] clims_per_level_dict={clims_per_level_dict is not None}, type={type(clims_per_level_dict)}")

        if clims_per_level_dict and isinstance(clims_per_level_dict, dict):
            # V3 format: clims stored as {"0": {...}, "1": {...}, ...} at position level
            for lvl in level_names:
                lvl_clims = clims_per_level_dict.get(str(lvl), {})

                cl = lvl_clims.get("contrast_limits")
                if isinstance(cl, (list, tuple)) and len(cl) == 2:
                    try:
                        per_level_clims.append((float(cl[0]), float(cl[1])))
                    except Exception:
                        per_level_clims.append(None)
                else:
                    per_level_clims.append(None)

                clpc = lvl_clims.get("contrast_limits_per_channel")
                lvl_pc = None
                if isinstance(clpc, (list, tuple)):
                    lvl_pc = []
                    for i in range(int(c_dim)):
                        try:
                            entry = clpc[i]
                            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                                lvl_pc.append((float(entry[0]), float(entry[1])))
                            else:
                                lvl_pc.append(None)
                        except Exception:
                            lvl_pc.append(None)
                per_level_clims_per_channel.append(lvl_pc)

            # Read gamma_per_channel from position level
            gpc = pos_attrs.get("gamma_per_channel")
            if isinstance(gpc, (list, tuple)):
                gamma_per_channel = [float(g) for g in gpc]

        else:
            # V2 format: clims stored in each pyramid level's .zattrs
            for lvl in level_names:
                comp_dir = Path(source_store) / pos / str(lvl)
                attrs = _read_component_attrs(comp_dir)

                cl = attrs.get("contrast_limits")
                if isinstance(cl, (list, tuple)) and len(cl) == 2:
                    try:
                        per_level_clims.append((float(cl[0]), float(cl[1])))
                    except Exception:
                        per_level_clims.append(None)
                else:
                    per_level_clims.append(None)

                clpc = attrs.get("contrast_limits_per_channel")
                lvl_pc = None
                if isinstance(clpc, (list, tuple)):
                    lvl_pc = []
                    for i in range(int(c_dim)):
                        try:
                            entry = clpc[i]
                            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                                lvl_pc.append((float(entry[0]), float(entry[1])))
                            else:
                                lvl_pc.append(None)
                        except Exception:
                            lvl_pc.append(None)
                per_level_clims_per_channel.append(lvl_pc)

                # Read gamma_per_channel from first level only
                if not gamma_per_channel:
                    gpc = attrs.get("gamma_per_channel")
                    if isinstance(gpc, (list, tuple)):
                        gamma_per_channel = [float(g) for g in gpc]

    except Exception:
        per_level_clims = [None] * len(level_names)
        per_level_clims_per_channel = [None] * len(level_names)

    return per_level_clims, per_level_clims_per_channel, gamma_per_channel


def write_component_attrs(component_dir: Path, updates: dict) -> None:
    """Merge and write .zattrs for a component directory.

    - Reads existing .zattrs (if present), updates with provided keys, then writes back.
    - Silently no-ops on I/O errors to keep pipelines resilient.
    """
    try:
        attrs_path = component_dir / ".zattrs"
        try:
            with open(attrs_path, "r") as f:
                attrs = json.load(f)
        except Exception:
            attrs = {}
        if not isinstance(attrs, dict):
            attrs = {}
        attrs.update(updates or {})
        with open(attrs_path, "w") as f:
            json.dump(attrs, f)
    except Exception:
        pass


def list_numeric_levels(source_store: Path | str, pos: str) -> list[str]:
    """Return sorted numeric level keys for a position using iohub."""
    with open_ome_zarr(Path(source_store) / pos, layout="fov", mode="r") as fov:
        keys = [k for k in getattr(fov, "array_keys", lambda: [])() if str(k).isdigit()]
        return sorted(keys, key=lambda s: int(s))


def get_level0_shape(source_store: Path | str, pos: str) -> tuple[int, int]:
    """Return (Y,X) shape for level 0 of a position."""
    with open_ome_zarr(Path(source_store) / pos, layout="fov", mode="r") as fov:
        y0, x0 = int(fov["0"].shape[-2]), int(fov["0"].shape[-1])
        return y0, x0


def get_channel_dim(source_store: Path | str, pos: str) -> int:
    """Return number of channels for a position inferred from iohub shape."""
    with open_ome_zarr(Path(source_store) / pos, layout="fov", mode="r") as fov:
        return int(fov.data.shape[1]) if fov.data.ndim >= 2 else 1


def ensure_pyramid_levels(source_store: Path | str, pos: str, levels: int, force: bool = False) -> None:
    """Initialize pyramid levels for a position, optionally deleting existing levels first.

    NOTE: This should be called ONCE per position before parallel jobs, not from
    within parallel workers. For parallel pyramid building, pre-initialize all
    positions first, then have workers just write data.

    Args:
        source_store: Path to the zarr store
        pos: Position path (e.g., "A/1/0")
        levels: Number of pyramid levels to create
        force: If True, delete existing pyramid levels before reinitializing
    """
    import shutil

    source_path = Path(source_store)
    pos_path = source_path / pos

    # If force mode, delete existing pyramid levels BEFORE opening store
    # This must be done outside the iohub context to avoid stale metadata
    if force:
        for level in range(1, levels):
            level_path = pos_path / str(level)
            if level_path.exists():
                shutil.rmtree(level_path, ignore_errors=True)
    else:
        # Check if all pyramid levels already exist
        all_levels_exist = True
        for level in range(1, levels):
            level_path = pos_path / str(level)
            # Check for zarr.json (v3) or .zarray (v2)
            if not (level_path / "zarr.json").exists() and not (level_path / ".zarray").exists():
                all_levels_exist = False
                break

        if all_levels_exist:
            return

    # Now open fresh and initialize (zarr will see clean state after deletion)
    with open_ome_zarr(source_store, mode="r+") as store:
        fov = store[pos]
        fov.initialize_pyramid(levels=levels)


def _register_pyramid_levels_in_multiscales(
    pos_path: Path, levels: int, factor: int = 2
) -> None:
    """Ensure the position's multiscales `datasets` list registers every on-disk
    pyramid level. The v3 unsharded init creates level *arrays* but not metadata;
    NGFF readers (napari) discover levels from this list, so without it a built
    pyramid renders as a single level. Idempotent. Derives each level's scale
    from level 0 by multiplying spatial axes by factor**level (matches iohub
    `initialize_pyramid`); leaves time/channel axes untouched.
    """
    import json

    for fname in ("zarr.json", ".zattrs"):
        meta_path = pos_path / fname
        if not meta_path.exists():
            continue
        try:
            d = json.loads(meta_path.read_text())
        except Exception:
            return
        attrs = d.get("attributes", d)
        ome = attrs.get("ome") or attrs
        ms = ome.get("multiscales")
        if not ms or not ms[0].get("datasets"):
            return
        axes = ms[0].get("axes", [])
        template = ms[0]["datasets"][0]
        try:
            base_scale = next(
                ct["scale"] for ct in template["coordinateTransformations"]
                if ct.get("type") == "scale"
            )
        except (KeyError, StopIteration):
            return
        ndim = len(base_scale)
        # Spatial axis indices: prefer axis metadata, else assume last 3 (TCZYX)
        # or last 2 (YX-only) are spatial.
        if axes and len(axes) == ndim:
            spatial = [i for i, a in enumerate(axes) if a.get("type") == "space"]
        else:
            spatial = list(range(max(0, ndim - 3), ndim))

        new_datasets = []
        for level in range(levels):
            if not (pos_path / str(level)).exists():
                continue
            scale_vec = [
                base_scale[i] * (factor ** level) if i in spatial else base_scale[i]
                for i in range(ndim)
            ]
            new_datasets.append({
                "path": str(level),
                "coordinateTransformations": [{"type": "scale", "scale": scale_vec}],
            })
        if not new_datasets:
            return
        # No-op if already complete with matching scales.
        if ms[0]["datasets"] == new_datasets:
            return
        ms[0]["datasets"] = new_datasets
        meta_path.write_text(json.dumps(d, indent=2))
        return


def ensure_pyramid_levels_unsharded(
    source_store: Path | str,
    pos: str,
    levels: int,
    force: bool = False,
    factor: int = 2,
    chunks: tuple = (1, 1, 1, 512, 512),
) -> None:
    """Initialize pyramid levels with NO sharding for parallel-safe writes.

    This creates unsharded arrays (one file per chunk) which allows multiple
    SLURM workers to write to different (t, c) indices without lock contention.
    After all parallel workers complete, call reshard_zarr_array() to consolidate.

    Args:
        source_store: Path to the zarr store
        pos: Position path (e.g., "A/1/0")
        levels: Number of pyramid levels to create
        force: If True, delete existing pyramid levels before reinitializing
        factor: Downsampling factor between levels (default: 2)
        chunks: Chunk size tuple for pyramid levels (default: (1, 1, 1, 512, 512))
    """
    import shutil
    import threading
    import time as _t
    import zarr

    source_path = Path(source_store)
    pos_path = source_path / pos
    zarr_format = detect_zarr_format(source_path)

    # Force: rename existing levels (instant) and rmtree in a daemon thread so
    # init isn't blocked by NFS file deletion. Also sweep stale temp/trash dirs
    # left over from killed prior runs.
    def _async_rmtree(p):
        threading.Thread(target=shutil.rmtree, args=(p,), kwargs={"ignore_errors": True}, daemon=True).start()
    if force:
        ts = int(_t.time() * 1000)
        for stale in list(pos_path.glob("*_resharding_temp")) + list(pos_path.glob("*_old_trash")) + list(pos_path.glob(".__trash_*")):
            _async_rmtree(stale)
        for level in range(1, levels):
            level_path = pos_path / str(level)
            if level_path.exists():
                trash = pos_path / f".__trash_{level}_{ts}"
                os.rename(str(level_path), str(trash))
                _async_rmtree(trash)
    else:
        # Check if all pyramid levels already exist
        all_levels_exist = True
        for level in range(1, levels):
            level_path = pos_path / str(level)
            if not (level_path / "zarr.json").exists() and not (level_path / ".zarray").exists():
                all_levels_exist = False
                break

        if all_levels_exist:
            # Arrays present but metadata may not list them — register and return.
            _register_pyramid_levels_in_multiscales(pos_path, levels, factor)
            return

    # Get base array (level 0) shape and dtype
    level0_path = pos_path / "0"
    level0_arr = zarr.open(str(level0_path), mode="r")
    base_shape = level0_arr.shape
    dtype = level0_arr.dtype

    # Create each pyramid level with NO sharding (parallel-safe)
    for level in range(1, levels):
        level_path = pos_path / str(level)

        # Calculate downsampled shape for this level
        scale = factor ** level
        # Only downsample spatial dimensions (last 2), keep T, C, Z the same
        level_shape = tuple(
            s if i < len(base_shape) - 2 else max(1, s // scale)
            for i, s in enumerate(base_shape)
        )

        # Use shards_ratio=(1,1,1,1,1) for parallel-safe writes (one file per chunk)
        create_zarr_array(
            path=str(level_path),
            shape=level_shape,
            chunks=chunks,
            dtype=dtype,
            zarr_format=zarr_format,
            shards_ratio=(1,) * len(chunks),  # NO sharding - parallel safe
            fill_value=0,
            overwrite=True,
        )

    # Register all created levels in the position multiscales metadata so NGFF
    # readers (napari) discover them — array creation alone doesn't do this.
    _register_pyramid_levels_in_multiscales(pos_path, levels, factor)


def add_missing_zarr_metadata(
    source_store: Path | str,
    pos: str,
    array_name: str = "seg",
    level: int = 0,
    reference_array: str = "0"
) -> bool:
    """
    Add missing .zarray metadata for a zarr array by inferring from reference array.

    Parameters
    ----------
    source_store : Path | str
        Path to the zarr store
    pos : str
        Position path (e.g., "A/1/0")
    array_name : str
        Name of the array that's missing metadata (e.g., "seg", "nuclear_seg")
    level : int
        Pyramid level (default: 0)
    reference_array : str
        Reference array to infer shape from (default: "0" for main image)

    Returns
    -------
    bool
        True if metadata was successfully added, False otherwise
    """
    import zarr

    source_store = Path(source_store)
    target_path = source_store / pos / array_name / str(level)
    zarray_path = target_path / ".zarray"

    # Check if metadata already exists
    if zarray_path.exists():
        return True

    # Check if the directory exists
    if not target_path.exists():
        return False

    # Check if any chunk files exist
    chunk_files = list(target_path.glob("*"))
    if not chunk_files:
        print(f"WARNING: No data chunks found in {target_path}")
        return False

    try:
        # Infer shape from reference array - only read metadata, not data
        import zarr as zarr_lib

        # Open reference zarr array to read shape from metadata only
        ref_path = source_store / pos / reference_array
        ref_zarr = zarr_lib.open_array(str(ref_path), mode='r')
        inferred_shape = tuple(ref_zarr.shape)

        # Determine appropriate chunk size based on dimensionality
        if len(inferred_shape) >= 5:
            chunks = (1, 1, 1, min(2048, inferred_shape[-2]), min(2048, inferred_shape[-1]))
        elif len(inferred_shape) == 4:
            chunks = (1, 1, min(2048, inferred_shape[-2]), min(2048, inferred_shape[-1]))
        else:
            chunks = (min(2048, inferred_shape[-2]), min(2048, inferred_shape[-1]))

        # Create zarr array metadata by writing .zarray file directly
        print(f"Reconstructing {array_name} metadata:")
        print(f"  Shape: {inferred_shape}")
        print(f"  Chunks: {chunks}")
        print(f"  Dtype: int32")

        # Write .zarray metadata file
        zarray_metadata = {
            "chunks": list(chunks),
            "compressor": {
                "id": "blosc",
                "cname": "lz4",
                "clevel": 5,
                "shuffle": 1
            },
            "dtype": "<i4",  # int32 little-endian
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "shape": list(inferred_shape),
            "zarr_format": 2
        }

        with open(zarray_path, 'w') as f:
            json.dump(zarray_metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"ERROR: Failed to reconstruct metadata for {target_path}: {e}")
        return False


# Build fine-grained tasks over (position, time, channel, level)
def enumerate_units(
    source_store: Path | str,
    pos_paths: list[str],
    t_indices: "list[int] | None" = None,
) -> list[tuple[str, int, int]]:
    """List (position, t, c) units to process.

    `t_indices` (optional) restricts to those timepoints; useful when
    parallelizing per-(pos, t) — pyramid shards are (1, C, 1, Y, X), so
    different t's write to different shard files and can run concurrently.
    """
    units: list[tuple[str, int, int]] = []
    with open_ome_zarr(source_store, mode="r") as store_local:
        for pos_path in pos_paths:
            fov = store_local[pos_path]
            t_dim = fov.data.shape[0] if fov.data.ndim >= 1 else 1
            c_dim = get_channel_dim(source_store, pos_path)
            t_range = range(int(t_dim)) if t_indices is None else [int(t) for t in t_indices if 0 <= int(t) < int(t_dim)]
            for t in t_range:
                for c in range(int(c_dim)):
                    units.append((pos_path, int(t), int(c)))
    return units


def ensure_position_array(
    store,
    hsc_name: str,
    shape: tuple,
    chunk_size,
    dtype,
    scale,
) -> None:
    """Ensure Zarr position `hsc_name` and array '0' exist (idempotent)."""
    from pathlib import Path as _P

    # Create position group if missing
    try:
        pos = store.create_position(*_P(hsc_name).parts)
    except Exception:
        pos = store[hsc_name]
    # Create primary array if missing
    try:
        pos.create_zeros(
            name="0",
            shape=shape,
            chunks=chunk_size,
            dtype=dtype,
            transform=[TransformationMeta(type="scale", scale=scale)],
        )
    except (FileExistsError, ValueError, Exception):
        pass


def _calculate_central_roi(shape: tuple, n_tiles: int, tile_size: int = 2048) -> tuple:
    """Calculate centered ROI slices for spatial dimensions based on tile grid.

    Args:
        shape: Full array shape (T, Y, X) or (T, C, Z, Y, X)
        n_tiles: Number of tiles per side (e.g., 4 creates a 4×4 grid = 16 tiles total)
        tile_size: Size of each tile in pixels (default: 2048)

    Returns:
        Tuple of slices that selects central region, e.g.:
        - For 3D (T,Y,X): (slice(None), slice(y_start, y_end), slice(x_start, x_end))
        - For 5D (T,C,Z,Y,X): (slice(None), slice(None), slice(None), slice(y_start, y_end), slice(x_start, x_end))

    Example:
        n_tiles=4, tile_size=2048 → ROI is 8192×8192 pixels (4×4 grid of 2048×2048 tiles)
    """
    roi_y = roi_x = n_tiles * tile_size

    # Assume last two dimensions are always Y, X
    full_y, full_x = shape[-2], shape[-1]

    # Calculate centered crop
    y_start = max(0, (full_y - roi_y) // 2)
    y_end = min(full_y, y_start + roi_y)
    x_start = max(0, (full_x - roi_x) // 2)
    x_end = min(full_x, x_start + roi_x)

    # Build slice tuple: keep all leading dimensions, crop YX
    n_leading_dims = len(shape) - 2
    slices = tuple([slice(None)] * n_leading_dims) + (
        slice(y_start, y_end),
        slice(x_start, x_end),
    )

    return slices


def _get_three_roi_configs(
    full_shape: tuple,
    roi_size: int = 500,
    edge_distance: int = 7000,
) -> List[tuple]:
    """Calculate 3 ROI configurations (center, mid-radius, edge) for sampling.

    Args:
        full_shape: Full array spatial shape (Y, X)
        roi_size: Size of each ROI in pixels (default: 500)
        edge_distance: Distance from edge for the third ROI in pixels (default: 7000)

    Returns:
        List of tuples: [(y_center, x_center, label, y_start, y_end, x_start, x_end), ...]

    Example:
        >>> roi_configs = _get_three_roi_configs((30000, 30000), roi_size=500)
        >>> for y_c, x_c, label, y_s, y_e, x_s, x_e in roi_configs:
        ...     print(f"{label}: {y_s}:{y_e}, {x_s}:{x_e}")
    """
    cy, cx = full_shape[0] // 2, full_shape[1] // 2

    # Calculate distances from center
    max_radius = min(cy, cx)
    mid_distance = max_radius // 2
    calculated_edge_distance = max_radius - edge_distance

    # Define 3 ROI locations: center, mid-radius, near-edge
    roi_centers = [
        (cy, cx, "center"),
        (cy - mid_distance, cx, "mid"),
        (cy - calculated_edge_distance, cx, "edge"),
    ]

    roi_configs = []
    for roi_y, roi_x, label in roi_centers:
        # Calculate ROI bounds
        y_start = max(0, roi_y - roi_size // 2)
        y_end = min(full_shape[0], y_start + roi_size)
        x_start = max(0, roi_x - roi_size // 2)
        x_end = min(full_shape[1], x_start + roi_size)

        roi_configs.append((roi_y, roi_x, label, y_start, y_end, x_start, x_end))

    return roi_configs


def _save_roi_coords_to_yaml(
    roi_configs: List[tuple],
    output_path: Path,
    well: str,
    full_shape: tuple,
    roi_size: int,
    edge_distance: int,
):
    """Save ROI coordinates to a YAML file.

    Args:
        roi_configs: List of (y_center, x_center, label, y_start, y_end, x_start, x_end) tuples
        output_path: Path to save YAML file
        well: Well identifier (e.g., "A/1/0")
        full_shape: Full image shape (Y, X)
        roi_size: Size of each ROI in pixels
        edge_distance: Distance from edge for edge ROI
    """
    import yaml

    roi_coords_data = {
        "well": well,
        "full_shape": {"y": int(full_shape[0]), "x": int(full_shape[1])},
        "roi_size": roi_size,
        "edge_distance": edge_distance,
        "rois": [],
    }

    for roi_y, roi_x, label, y_start, y_end, x_start, x_end in roi_configs:
        roi_coords_data["rois"].append(
            {
                "label": label,
                "center": {"y": int(roi_y), "x": int(roi_x)},
                "bounds": {
                    "y_start": int(y_start),
                    "y_end": int(y_end),
                    "x_start": int(x_start),
                    "x_end": int(x_end),
                },
            }
        )

    with open(output_path, "w") as f:
        yaml.dump(roi_coords_data, f, default_flow_style=False, sort_keys=False)


# --------- Fast Zarr Writing Utilities ---------


def reshard_zarr_array(
    source_path: Path | str,
    dest_path: Path | str = None,
    chunks: tuple = None,
    shards_ratio: tuple = None,
    tile_size: int = 4096,
    show_progress: bool = True,
) -> Path:
    """
    Reshard a zarr v3 array to a different chunking/sharding configuration.

    This is useful when you need to:
    - Convert from unsharded (one file per chunk) to sharded for storage efficiency
    - Change shard sizes for different access patterns
    - Match chunking to an existing array (e.g., match seg group format)

    The function reads the source array tile-by-tile and writes to the destination
    with the new chunking/sharding configuration. This is memory-efficient as it
    only loads one tile at a time.

    Args:
        source_path: Path to source zarr array (e.g., "/data/store.zarr/A/1/0/labels/cell_seg/0")
        dest_path: Path to destination zarr array. If None, creates a temp array and
                   replaces the source (in-place resharding).
        chunks: Target chunk size. If None, uses (1, 1, 1, 512, 512) for 5D arrays.
        shards_ratio: Sharding ratio for target. If None, uses (1, 1, 1, 32, 32) for
                      ~16k x 16k shards (~1GB shard files for int32).
        tile_size: Size of tiles to read/write at a time (default: 4096).
                   Larger = faster but more memory.
        show_progress: Whether to show progress bar (default: True).

    Returns:
        Path to the resharded array (dest_path or source_path if in-place).

    Example:
        # Reshard an unsharded array to match seg group format
        reshard_zarr_array(
            source_path="/data/store.zarr/A/1/0/labels/cell_seg/0",
            chunks=(1, 1, 1, 512, 512),
            shards_ratio=(1, 1, 1, 32, 32),
        )

        # Reshard to a different location
        reshard_zarr_array(
            source_path="/data/store.zarr/A/1/0/labels/cell_seg_unsharded/0",
            dest_path="/data/store.zarr/A/1/0/labels/cell_seg/0",
            chunks=(1, 1, 1, 512, 512),
            shards_ratio=(1, 1, 1, 32, 32),
        )
    """
    import zarr
    import shutil
    import tempfile
    from tqdm import tqdm

    source_path = Path(source_path)
    in_place = dest_path is None

    # Open source array and get metadata
    source_arr = zarr.open(str(source_path), mode="r")
    shape = source_arr.shape
    dtype = source_arr.dtype
    ndim = len(shape)

    # Default chunks: preserve source chunks if not specified
    if chunks is None:
        chunks = source_arr.chunks

    # Default shards_ratio based on dimensionality
    if shards_ratio is None:
        if ndim >= 5:
            shards_ratio = (1, 1, 1, 32, 32)  # 16384x16384 shards
        elif ndim == 4:
            shards_ratio = (1, 1, 32, 32)
        elif ndim == 3:
            shards_ratio = (1, 32, 32)
        else:
            shards_ratio = (32, 32)

    # Calculate shards and clamp to array dimensions (for small pyramid levels)
    # IMPORTANT: Shard dimensions must be multiples of chunk dimensions (zarr requirement)
    shards = tuple(c * r for c, r in zip(chunks, shards_ratio))

    def clamp_shard(shard_size, array_size, chunk_size):
        if shard_size <= array_size:
            return shard_size  # No clamping needed
        # Clamp to array size, but round up to nearest chunk multiple
        n_chunks = (array_size + chunk_size - 1) // chunk_size  # ceil division
        return n_chunks * chunk_size

    shards = tuple(clamp_shard(s, sh, c) for s, sh, c in zip(shards, shape, chunks))

    # Determine destination path
    if in_place:
        # Create temp destination next to source
        temp_dir = source_path.parent
        dest_path = temp_dir / f"{source_path.name}_resharding_temp"
    else:
        dest_path = Path(dest_path)

    # Rename + async rmtree to avoid NFS "Directory not empty" races on stale leftovers.
    if dest_path.exists():
        import threading, time as _t
        trash = dest_path.with_name(f"{dest_path.name}.__trash_{int(_t.time() * 1000)}")
        os.rename(str(dest_path), str(trash))
        threading.Thread(target=shutil.rmtree, args=(trash,), kwargs={"ignore_errors": True}, daemon=True).start()

    # Create destination array with new chunking/sharding
    dest_group = zarr.open_group(str(dest_path.parent), mode="a", zarr_format=3)
    dest_arr = dest_group.create_array(
        name=dest_path.name,
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        fill_value=0,
        overwrite=True,
    )

    print(f"  Resharding: {source_path.name}")
    print(f"    Source shape: {shape}, dtype: {dtype}")
    print(f"    Target chunks: {chunks}, shards: {shards}")

    # Copy data in parallel shard-sized tiles (one task per destination
    # shard). Benchmarked against shard-row strips on a 40 GB int32
    # labels zarr: 7 row-strips × 7-way parallel = 63 s; 7 × 7 shard
    # tiles × 16-way parallel = 51.5 s (18 % faster). The gain comes
    # from keeping the thread pool saturated for longer — shard-row
    # strips starve as fast workers finish first. Worker cap is
    # OPS_RESHARD_WORKERS, default 16.
    from concurrent.futures import ThreadPoolExecutor
    import os

    y_dim, x_dim = shape[-2], shape[-1]
    shard_y = shards[-2]
    shard_x = shards[-1]
    n_y = max(1, (y_dim + shard_y - 1) // shard_y)
    n_x = max(1, (x_dim + shard_x - 1) // shard_x)
    n_tasks = n_y * n_x
    max_workers = int(os.environ.get("OPS_RESHARD_WORKERS", str(min(n_tasks, 16))))
    max_workers = max(1, min(max_workers, n_tasks))
    print(f"    Copying data in {n_y}×{n_x}={n_tasks} shard tiles using {max_workers} workers...")

    def _copy_shard_tile(task_idx):
        """Copy one destination shard. Each thread opens its own zarr handles."""
        import zarr as _zarr
        _src = _zarr.open(str(source_path), mode="r")
        _dst_group = _zarr.open_group(str(dest_path.parent), mode="r+", zarr_format=3)
        _dst = _dst_group[dest_path.name]

        iy, ix = divmod(task_idx, n_x)
        y0 = iy * shard_y
        y1 = min(y_dim, (iy + 1) * shard_y)
        x0 = ix * shard_x
        x1 = min(x_dim, (ix + 1) * shard_x)
        if ndim == 5:
            slc = (slice(None), slice(None), slice(None), slice(y0, y1), slice(x0, x1))
        elif ndim == 4:
            slc = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
        elif ndim == 3:
            slc = (slice(None), slice(y0, y1), slice(x0, x1))
        else:
            slc = (slice(y0, y1), slice(x0, x1))
        _dst[slc] = np.asarray(_src[slc])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_copy_shard_tile, range(n_tasks)))

    # If in-place, swap via rename (atomic) then delete old data in background.
    # os.rename is instant on the same filesystem vs shutil.rmtree on thousands of files.
    if in_place:
        trash_path = source_path.parent / f"{source_path.name}_old_trash"
        if trash_path.exists():
            shutil.rmtree(trash_path)
        os.rename(str(source_path), str(trash_path))
        os.rename(str(dest_path), str(source_path))
        # Validate that zarr.json survived the move
        zarr_json = source_path / "zarr.json"
        if not zarr_json.exists():
            raise RuntimeError(
                f"Resharding failed: zarr.json missing after in-place swap at {source_path}. "
                f"The array data may be present but metadata was lost."
            )
        # Delete old unsharded data (non-blocking — ok if this is slow)
        shutil.rmtree(trash_path, ignore_errors=True)
        print(f"    In-place resharding complete: {source_path.name}")
        return source_path
    else:
        print(f"    Resharding complete: {dest_path}")
        return dest_path


def write_zarr_slice_direct(
    store_path, component_path: str, data, t: int, c: int
) -> None:
    """Write data directly to a zarr array slice, bypassing iohub overhead.

    This is significantly faster than iohub writes (3-4x speedup) because it:
    - Uses zarr.open() directly without metadata processing
    - Performs direct array indexing without coordinate transformations
    - Avoids validation and safety checks on each write

    Args:
        store_path: Path to zarr store
        component_path: Component path within store (e.g., "A/1/0/1")
        data: Data array to write (spatial dimensions only, e.g., (Z, Y, X) or (Y, X))
        t: Time index
        c: Channel index

    Example:
        >>> write_zarr_slice_direct(
        ...     store_path="/path/to/store.zarr",
        ...     component_path="A/1/0/1",
        ...     data=my_array,  # shape: (1, 52309, 52374)
        ...     t=0, c=0
        ... )
    """
    import zarr

    zarr_store = zarr.open(str(store_path), mode="r+")
    zarr_arr = zarr_store[component_path]

    # Expand data dims to match the zarr array exactly.
    # Caller provides spatial-only data (Y,X) or (Z,Y,X); we prepend singleton
    # dims until we match the number of spatial dimensions, then add T and C.
    spatial_dims_needed = zarr_arr.ndim - 2  # dims beyond T and C
    while data.ndim < spatial_dims_needed:
        data = data[np.newaxis, ...]
    data_with_tc = data[np.newaxis, np.newaxis, ...]

    # Direct assignment to zarr array slice
    zarr_arr[t : t + 1, c : c + 1] = data_with_tc


def write_zarr_slices_parallel(store_path, writes, max_workers: int = 4):
    """Write multiple zarr slices in parallel for maximum throughput.

    Parallelizing writes provides significant speedup (~4x) because:
    - Network I/O latency dominates write time
    - Multiple parallel writes saturate network bandwidth
    - Smaller arrays benefit most (less overhead per MB written)

    Args:
        store_path: Path to zarr store
        writes: List of (component_path, data, t, c) tuples to write
        max_workers: Maximum number of parallel write threads (default: 4)

    Returns:
        List of (component_path, elapsed_seconds, mb_written) tuples

    Example:
        >>> writes = [
        ...     ("A/1/0/1", level1_data, 0, 0),
        ...     ("A/1/0/2", level2_data, 0, 0),
        ...     ("A/1/0/3", level3_data, 0, 0),
        ... ]
        >>> results = write_zarr_slices_parallel("/path/to/store.zarr", writes)
        >>> for path, time, mb in results:
        ...     print(f"{path}: {time:.1f}s ({mb:.1f} MB, {mb/time:.1f} MB/s)")
    """
    from joblib import Parallel, delayed
    import time

    def write_single(component_path: str, data, t: int, c: int):
        """Execute one write and return timing stats."""
        t_start = time.time()
        mb_size = data.nbytes / 1024 / 1024

        write_zarr_slice_direct(store_path, component_path, data, t, c)

        elapsed = time.time() - t_start
        return component_path, elapsed, mb_size

    # Execute writes in parallel
    n_workers = min(len(writes), max_workers)
    results = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(write_single)(comp, data, t, c) for comp, data, t, c in writes
    )

    return results


def add_missing_zgroups(store_path: Path, verbose: bool = True) -> int:
    """
    Add missing .zgroup files to a zarr v2 store.

    This fixes stores where directories are valid zarr groups but are missing
    the .zgroup metadata file. This is needed for zarr v3 library compatibility
    when reading zarr v2 stores.

    Parameters
    ----------
    store_path : Path or str
        Path to the zarr store root
    verbose : bool
        If True, print each .zgroup file added

    Returns
    -------
    int
        Number of .zgroup files added

    Example
    -------
    >>> from cyclops_utils.io.zarr_utils import add_missing_zgroups
    >>> add_missing_zgroups("/path/to/cell_segmentation.zarr")
    Added .zgroup to A/1/0
    Added .zgroup to A/2/0
    Added 2 .zgroup files to cell_segmentation.zarr
    """
    store_path = Path(store_path)

    if not store_path.exists():
        print(f"Store not found: {store_path}")
        return 0

    count = 0
    # Walk through all directories
    for dirpath in store_path.rglob("*"):
        if not dirpath.is_dir():
            continue

        # Skip if it's an array directory (has .zarray)
        if (dirpath / ".zarray").exists():
            continue

        # Skip if already has .zgroup or zarr.json
        if (dirpath / ".zgroup").exists() or (dirpath / "zarr.json").exists():
            continue

        # Check if it looks like a zarr group (has subdirs that are arrays or groups)
        has_zarr_children = any(
            (child / ".zarray").exists() or (child / ".zgroup").exists() or child.name.isdigit()
            for child in dirpath.iterdir() if child.is_dir()
        )

        if has_zarr_children:
            zgroup_path = dirpath / ".zgroup"
            zgroup_path.write_text('{"zarr_format": 2}')
            if verbose:
                print(f"  Added .zgroup to {dirpath.relative_to(store_path)}")
            count += 1

    print(f"Added {count} .zgroup files to {store_path.name}")
    return count
