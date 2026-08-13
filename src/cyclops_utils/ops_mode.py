"""Operational vs. research mode for OPS pipeline runs.

Operational mode dual-writes logs to a central location and uses real
data output paths. The central log root is OPS_LOG_ROOTDIR when set,
otherwise $OPS_BASE_PATH/logs. Research mode keeps logs local only and
redirects data outputs to a /rerun/ subdirectory so test runs cannot
overwrite production data.

Selection:
    OPS_MODE=research|operational         (env var; inherits into SLURM children)
    python -m cyclops_process.processes.run --mode operational

Precedence in run.py: explicit CLI flag > pre-set env var > default (research).
"""
from __future__ import annotations

import enum
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from cyclops_utils.paths import BASE_PATH


class OPSMode(str, enum.Enum):
    RESEARCH = "research"
    OPERATIONAL = "operational"


_MANUAL_COPY_FILENAME = "MANUAL_COPY_NEEDED.txt"


def get_mode() -> OPSMode:
    """Current OPS mode; defaults to RESEARCH if OPS_MODE is unset or unknown."""
    raw = os.environ.get("OPS_MODE", "").strip().lower()
    if raw == OPSMode.OPERATIONAL.value:
        return OPSMode.OPERATIONAL
    return OPSMode.RESEARCH


def is_operational() -> bool:
    return get_mode() is OPSMode.OPERATIONAL


def central_log_root() -> Path:
    """Root directory for dual-written logs.

    OPS_LOG_ROOTDIR when set — point it at a shared monitoring tree to collect
    logs from every account in one place — otherwise $OPS_BASE_PATH/logs.
    """
    override = os.environ.get("OPS_LOG_ROOTDIR")
    if override:
        return Path(override)
    return Path(f"{BASE_PATH}/logs")


def resolved_output_base_dir(base: str | Path) -> Path:
    """Append `/rerun` to `base` in research mode; return unchanged in operational.

    Idempotent: if the path already ends with 'rerun', it is returned unchanged.
    """
    p = Path(base)
    if is_operational():
        return p
    if p.name == "rerun":
        return p
    return p / "rerun"


def _manual_copy_file() -> Path:
    return Path(os.getcwd()) / _MANUAL_COPY_FILENAME


def register_manual_copy(
    src: str | Path,
    dst: str | Path,
    reason: str,
    *,
    context: str = "",
) -> None:
    """Record a failed dual-write so the user can reconcile later.

    Appends an rsync-ready line to MANUAL_COPY_NEEDED.txt in the cwd. Never
    raises — bookkeeping must not fail a pipeline.
    """
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        ctx = f"{context} " if context else ""
        line = f"rsync -a {src} {dst}    # {ts} {ctx}reason: {reason}\n"
        with _manual_copy_file().open("a") as f:
            f.write(line)
    except Exception as e:
        print(f"[ops_mode] warn: could not record manual-copy entry: {e}", file=sys.stderr)


def run_timestamp() -> str:
    """Stable timestamp for the current pipeline run, shared across all steps.

    Reads OPS_RUN_TS (set by run.py at startup and forwarded to SLURM children)
    so every step in the same run writes to a common central subdirectory.
    If unset — e.g. for an entry point other than run.py — initializes it once
    on first call and caches via the env var.
    """
    ts = os.environ.get("OPS_RUN_TS")
    if not ts:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        os.environ["OPS_RUN_TS"] = ts
    return ts


def central_path_for_step(experiment: str, step_name: str) -> Path:
    """Central log destination for (experiment, step) under the current run.

    Layout: ${OPS_LOG_ROOTDIR}/<experiment>/<step_name>/<OPS_RUN_TS>/
    """
    return central_log_root() / experiment / step_name / run_timestamp()


def _update_latest_symlink(experiment: str, step_name: str, target: Path) -> None:
    """Maintain a `latest` symlink alongside the timestamped run subdirs.

    Uses a relative target so the link survives mount-point changes.
    Failures are swallowed — a missing symlink is cosmetic.
    """
    link = central_log_root() / experiment / step_name / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.name)
    except Exception:
        pass


def mirror_slurm_log_dir(
    local_dir: str | Path,
    experiment: str,
    step_name: str,
    *,
    job_id: str | None = None,
) -> Path | None:
    """Copy a completed SLURM step's log directory into the central store.

    No-op in research mode. In operational mode, copies every file in
    `local_dir` (flat; no recursion) into
    ``${OPS_LOG_ROOTDIR}/<experiment>/<step_name>/<OPS_RUN_TS>/`` and updates
    a `latest` symlink alongside.

    Never raises. Any failure is recorded via `register_manual_copy` so the
    user can reconcile later, and the pipeline continues.

    Returns the destination path on success, None otherwise.
    """
    if not is_operational():
        return None

    src = Path(local_dir)
    dst = central_path_for_step(experiment, step_name)
    ctx = f"step={step_name} job_id={job_id or '?'}"

    if not src.exists():
        register_manual_copy(src, dst, reason="source directory not found", context=ctx)
        return None

    try:
        dst.mkdir(parents=True, exist_ok=True)
        for entry in sorted(src.iterdir()):
            if entry.is_file():
                shutil.copy2(entry, dst / entry.name)
        _update_latest_symlink(experiment, step_name, dst)
        return dst
    except Exception as e:
        register_manual_copy(src, dst, reason=f"{type(e).__name__}: {e}", context=ctx)
        return None


def mirror_experiment_logs(experiment: str, base_dir: str | Path | None = None) -> Path | None:
    """End-of-run sweep: copy the experiment's aggregate ``function_call_log.yaml``
    and the ``logs/`` directory of timestamped per-step copies into the central
    store under ``<exp>/_aggregate/<OPS_RUN_TS>/``.

    This is what gets called once at end-of-run (typically via ``atexit``) so
    Grafana / debuggers can find the full per-run history of the aggregate yaml
    that the decorator builds up over the run. Per-step ``slurm_stats.yaml``
    files written by ``mirror_step_stats`` already live under
    ``<exp>/<step>/<OPS_RUN_TS>/`` — those are unchanged.

    No-op in research mode. Never raises — failures are recorded via
    ``register_manual_copy``. Skips silently if either the aggregate yaml or
    the ``logs/`` dir is missing.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g. ``"ops0147_20260422"``).
    base_dir : str or Path, optional
        Override for the experiment root directory. Defaults to
        ``OPS_OUTPUT_BASE_DIR/<experiment>``. The decorator writes its
        aggregate yaml there.

    Returns
    -------
    Path or None
        Destination directory on success, ``None`` otherwise.
    """
    if not is_operational():
        return None

    if base_dir is None:
        base = os.environ.get("OPS_OUTPUT_BASE_DIR")
        if not base:
            return None
        base_dir = Path(base) / experiment
    else:
        base_dir = Path(base_dir)

    src_yaml = base_dir / "function_call_log.yaml"
    src_logs = base_dir / "logs"

    if not src_yaml.exists() and not src_logs.exists():
        return None

    dst_dir = central_log_root() / experiment / "_aggregate" / run_timestamp()
    ctx = f"experiment={experiment}"

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        if src_yaml.exists():
            shutil.copy2(src_yaml, dst_dir / src_yaml.name)
        if src_logs.exists() and src_logs.is_dir():
            dst_logs = dst_dir / "logs"
            if dst_logs.exists():
                shutil.rmtree(dst_logs)
            shutil.copytree(src_logs, dst_logs)
        _update_latest_symlink(experiment, "_aggregate", dst_dir)
        return dst_dir
    except Exception as e:
        register_manual_copy(
            base_dir, dst_dir,
            reason=f"{type(e).__name__}: {e}", context=ctx,
        )
        return None


def mirror_step_stats(
    experiment: str,
    step_name: str,
    log_key: str,
    entry: dict,
) -> Path | None:
    """Mirror one entry from `function_call_log.yaml` into the central
    per-step `slurm_stats.yaml`.

    Layout: ``${OPS_LOG_ROOTDIR}/<experiment>/<step_name>/<OPS_RUN_TS>/slurm_stats.yaml``

    Multiple invocations of the same step within one run (e.g. per-well
    fanout) accumulate into the same yaml — keyed by `log_key`, which the
    decorator builds from func name + process + method + well. File
    locking matches the aggregate writer so concurrent jobs don't corrupt
    the dict.

    No-op in research mode. Never raises — failures are recorded via
    `register_manual_copy` so the user can reconcile later.

    Returns the destination path on success, None otherwise.
    """
    if not is_operational():
        return None

    try:
        import fcntl
        import yaml as _yaml
    except ImportError:
        return None

    dst_dir = central_path_for_step(experiment, step_name)
    dst = dst_dir / "slurm_stats.yaml"
    ctx = f"step={step_name} log_key={log_key}"

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        lock_path = str(dst) + ".lock"
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                if dst.exists():
                    with open(dst, "r") as f:
                        existing = _yaml.safe_load(f) or {}
                else:
                    existing = {}
                existing[log_key] = entry
                with open(dst, "w") as f:
                    _yaml.dump(existing, f)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        _update_latest_symlink(experiment, step_name, dst_dir)
        return dst
    except Exception as e:
        register_manual_copy(
            Path("function_call_log entry"), dst,
            reason=f"{type(e).__name__}: {e}", context=ctx,
        )
        return None


def print_manual_copy_hint_if_any() -> None:
    """If any dual-writes failed, print a visible summary pointing the user at the file."""
    f = _manual_copy_file()
    if not f.exists():
        return
    bar = "=" * 72
    print("", file=sys.stderr)
    print(bar, file=sys.stderr)
    print("  SOME LOGS COULD NOT BE COPIED TO CENTRAL STORE", file=sys.stderr)
    print(f"  See {f}", file=sys.stderr)
    print("  Run the listed rsync commands manually to reconcile.", file=sys.stderr)
    print(bar, file=sys.stderr)
    try:
        print(f.read_text(), file=sys.stderr)
    except Exception:
        pass
