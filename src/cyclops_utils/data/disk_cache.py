"""Disk-backed parquet cache for any DataFrame.

One function: `df_cache`. Wraps a builder callable in a parquet sidecar
under `~/.cache/cyclops_utils/<namespace>/<key>.parquet` (override via
`$OPS_UTILS_CACHE_DIR`). Mtime-keyed when a `source_path` is given, so
re-exporting the upstream file auto-busts the cache.

Designed for places we currently re-read the same big CSV / re-filter
the same `linked_results` over and over: attention CSVs, NTC pools per
(experiment, well), captions tables, channel-score summaries, anything
that's slow to compute and used across multiple scripts.

Example — read-only big CSV with a gene filter on warm reads::

    df = df_cache(
        namespace="attention_csv",
        key="pma_top_phase_cells_v3.csv",
        builder=lambda: pd.read_csv(csv_path),
        source_path=csv_path,
        read_kwargs={"filters": [("gene", "==", "ABCE1")]},
    )

Example — derived per-(exp, well) NTC pool::

    pool = df_cache(
        namespace="ntc_pool",
        key=f"{exp}__{well}",
        builder=lambda: build_ntc_pool(exp, well),
        source_path=linked_results_csv,
    )

Deliberate non-features (keep it small): no locking, no TTL, no pickle
fallback, no compression knob. DataFrames only. Last-writer-wins on
concurrent rebuilds — parquet write is atomic via tmp+rename so readers
never see a half-file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from cyclops_utils.paths import BASE_PATH

# Shared cache root on the ICD-OPS project filesystem so every user
# benefits from each other's cache builds. The attention-atlas v3
# parquet alone is ~600 MB and takes ~30 s to build — paying that cost
# once per cluster instead of once per user is the whole point.
# Namespaces (e.g. "attention_csv", "ntc_pool") sit directly under this
# root, matching the existing `cache/phase_reconstruction/` convention.
# Override via $OPS_CACHE_DIR (e.g. point at /tmp during testing).
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "OPS_CACHE_DIR",
        f"{BASE_PATH}/cache",
    )
)


def _cache_path(namespace: str, key: str, cache_root: Path | None) -> Path:
    root = cache_root if cache_root is not None else DEFAULT_CACHE_ROOT
    # Replace path-unsafe chars in key so callers can use slashes / colons
    # in their natural keys without leaking subdir structure.
    safe_key = key.replace("/", "__").replace(":", "_").replace(" ", "_")
    return root / namespace / f"{safe_key}.parquet"


def _is_fresh(cache_path: Path, source_path: Path | None) -> bool:
    if not cache_path.exists():
        return False
    if source_path is None:
        return True
    try:
        if not source_path.exists():
            # Source vanished; cache is the only copy we have. Trust it.
            return True
        return cache_path.stat().st_mtime >= source_path.stat().st_mtime
    except OSError:
        # stat failed (permissions, broken link, etc.) — treat as stale
        # so we go through the builder path and report there if needed.
        return False


def df_cache(
    namespace: str,
    key: str,
    builder: Callable[[], pd.DataFrame | None],
    *,
    source_path: Path | str | None = None,
    read_kwargs: dict[str, Any] | None = None,
    cache_root: Path | str | None = None,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """In-memory-free, disk-backed parquet cache for one DataFrame.

    1. If a fresh parquet exists at the computed path, read and return it
       (with `read_kwargs` passed to `pd.read_parquet`, e.g. for filter
       pushdown).
    2. Otherwise call `builder()`. If it returns None, return None
       (nothing is written; subsequent calls will retry the build).
    3. Otherwise write the result to the cache path atomically (tmp +
       rename) and return it. Write failures are logged but never raise
       — the builder's value is still returned.

    Freshness:
        With `source_path`, the cache is fresh iff the parquet's mtime
        is >= the source's mtime. Without `source_path`, any existing
        cache file is considered fresh — the caller is responsible for
        invalidation (e.g. by deleting the file or changing `key`).

    Args:
        namespace: Logical bucket; becomes a subdir of the cache root.
            Use it to group related cached artifacts ("ntc_pool",
            "attention_csv", "captions", ...).
        key: Unique identifier within the namespace. Slashes, spaces and
            colons are normalized so callers can pass natural keys.
        builder: Zero-arg callable that returns the DataFrame to cache,
            or `None` to indicate "nothing to cache" (e.g. missing file).
        source_path: Optional upstream file whose mtime invalidates the
            cache. Pass the original CSV / linked_results path here.
        read_kwargs: Forwarded to `pd.read_parquet` on warm reads. Use
            `{"filters": [("col", "==", val)]}` for pushdown.
        cache_root: Override the default cache root. Defaults to
            `$OPS_UTILS_CACHE_DIR` or `~/.cache/cyclops_utils/`.
        verbose: Print a one-line status when the builder runs (cold) or
            the cache is read (warm).
    """
    src = Path(source_path) if source_path is not None else None
    root = Path(cache_root) if cache_root is not None else None
    path = _cache_path(namespace, key, root)

    if _is_fresh(path, src):
        try:
            df = pd.read_parquet(path, **(read_kwargs or {}))
            if verbose:
                print(f"  [df_cache] hit  {namespace}/{key} ({len(df)} rows)")
            return df
        except Exception as e:
            # Stale or corrupt parquet — fall through to builder. Don't
            # crash the caller over a bad cache file.
            print(f"  [df_cache] read failed {path}: {e} — rebuilding")

    if verbose:
        print(f"  [df_cache] miss {namespace}/{key} — building")

    df = builder()
    if df is None:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:
        # Persistence is best-effort. The user still gets the data.
        print(f"  [df_cache] write failed {path}: {e}")

    return df
