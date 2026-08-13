"""
HCS-OME-Zarr store creation backed by iohub.

## Usage

```python
from pathlib import Path
from cyclops_utils.io.zarr_precreate import create_hcs_store_fast
import numpy as np

create_hcs_store_fast(
    store_path=Path("experiment.zarr"),
    positions=["A/1/000001", "A/2/000001", ...],
    shape=(10, 3, 5, 2048, 2048),  # (T, C, Z, Y, X)
    chunks=(1, 1, 1, 256, 256),
    dtype=np.uint16,
    scale=(1.0, 1.0, 0.5, 0.325, 0.325),
    channel_names=["DAPI", "GFP", "RFP"],
)
```
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
from iohub.ngff import TransformationMeta, open_ome_zarr


def create_hcs_store_fast(
    store_path: Path,
    positions: List[str],
    shape: Tuple[int, int, int, int, int],
    chunks: Tuple[int, int, int, int, int],
    dtype: np.dtype,
    scale: Tuple[float, float, float, float, float],
    channel_names: List[str],
    acquisition_id: int = 0,
    omero_window_defaults: dict | None = None,
    shards_ratio: Tuple[int, ...] | None = None,
    version: Literal["0.4", "0.5"] = "0.4",
    mode: Literal["r+", "a", "w", "w-"] = "w",
) -> None:
    """Create an HCS-OME-Zarr store.

    TODO(v3 follow-up): Once ``iohub.ngff.utils.create_empty_plate`` (and the
    paired per-position writer ``iohub.ngff.utils.process_single_position``)
    is released and stable, replace this local helper with the iohub
    equivalents. biahub#215 did the same swap and removed ~1800 lines of
    in-repo plate/empty-create code; adopting iohub's helpers means future
    OME-Zarr 0.5+ fixes (channel-name handling, sharding defaults, glob/
    zarr.json traversal) flow in for free instead of being re-debugged here.
    See royerlab/ops_process#96 review thread for context and
    czbiohub-sf/biahub#215 for the reference migration.

    Args:
        store_path: Path to output zarr store
        positions: List of position paths in HCS format: "row/col/field"
        shape: 5D array shape (T, C, Z, Y, X)
        chunks: 5D chunk dimensions (T, C, Z, Y, X)
        dtype: Numpy data type
        scale: Coordinate transformation scale (T, C, Z, Y, X) in physical units
        channel_names: List of channel names
        acquisition_id: Acquisition index (default: 0)
        omero_window_defaults: Unused, kept for API compatibility
        shards_ratio: Per-dimension shard/chunk ratio for zarr v3 sharding (v3 only).
            Each shard contains the product of ratios number of chunks per dimension.
        version: OME-NGFF version — "0.4" writes zarr v2, "0.5" writes zarr v3
        mode: Store open mode passed to open_ome_zarr.
            "w"  — create new, overwrite if exists (default).
            "w-" — create new, raise if exists.
            "a"  — create if not exists, append positions if exists.
            "r+" — read/write, must exist.
    """
    if shards_ratio is not None and version != "0.5":
        raise ValueError(
            f"shards_ratio requires version='0.5' (zarr v3). Got version='{version}'."
        )
    shape = tuple(int(s) for s in shape)
    chunks = tuple(int(c) for c in chunks)
    pos_specs = [
        (*Path(p).parts[:3], None, None, acquisition_id)
        for p in positions
    ]
    transform = [TransformationMeta(type="scale", scale=list(scale))]

    with open_ome_zarr(
        store_path, layout="hcs", mode=mode,
        channel_names=channel_names, version=version,
    ) as plate:
        created = plate.create_positions(pos_specs)

        def _init_array(pos):
            pos.create_zeros("0", shape=shape, dtype=dtype, chunks=chunks,
                             transform=transform, shards_ratio=shards_ratio)

        with ThreadPoolExecutor(max_workers=min(32, len(positions))) as pool:
            list(pool.map(_init_array, created))
