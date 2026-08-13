"""Phase tracker for DAG-integrated inner-job tracking.

When the DAG runner executes a launcher step locally, it sets a PhaseTracker
in a context variable.  Utility functions (``wait_for_multiple_job_arrays``,
``submit_parallel_jobs``, ``monitor_jobs_with_retry``) check this variable
and transparently delegate job-array waiting to the DAG runner's
``_track_inner_jobs``, which updates the live DAG table with real-time
progress.

Step functions require **zero changes** — the tracking is completely
transparent.  CLI invocations (where no PhaseTracker is set) continue to
use internal waiting logic as before.
"""

import contextvars
from typing import Callable

_current_phase_tracker: contextvars.ContextVar["PhaseTracker | None"] = (
    contextvars.ContextVar("phase_tracker", default=None)
)


class _JobRef:
    """Minimal job reference for ``_track_inner_jobs`` compatibility.

    ``_track_inner_jobs`` accesses ``job_info["job"].job_id``. This shim
    lets us build a LauncherResult from plain SLURM job-ID strings
    (as used by ``monitor_jobs_with_retry``).
    """

    def __init__(self, job_id: str):
        self.job_id = job_id


def launcher_result_from_job_ids(
    job_ids: list[str],
    job_specs: list[dict],
    label: str = "",
):
    """Build a ``LauncherResult`` from plain SLURM job-ID strings.

    Used by the ``monitor_jobs_with_retry`` hook so that ``_track_inner_jobs``
    can poll sacct for jobs originally submitted outside of
    ``submit_parallel_jobs``.
    """
    from cyclops_utils.hpc.launcher_result import JobArray, LauncherResult

    submitted_jobs = [
        {"name": spec.get("name", f"job_{jid}"), "job": _JobRef(jid)}
        for jid, spec in zip(job_ids, job_specs)
    ]

    # Derive base_job_id (strip array index suffix like "12345_0" → "12345")
    base_id = job_ids[0].split("_")[0] if job_ids else ""

    arr = JobArray(
        base_job_id=base_id,
        submitted_jobs=submitted_jobs,
        label=label,
    )
    return LauncherResult(job_arrays=[arr], total_jobs=len(job_ids))


class PhaseTracker:
    """Bridges utility wait calls with DAG runner inner-job tracking.

    Parameters
    ----------
    step_name : str
        Name of the step being executed (for display updates).
    track_fn : callable
        Bound method ``DAGRunner._track_inner_jobs(name, lr)`` that polls
        sacct, updates the DAG display, and returns
        ``(n_failed, completed_names, failed_names)``.
    """

    def __init__(self, step_name: str, track_fn: Callable, total_phases: int = 1):
        self.step_name = step_name
        self.track_fn = track_fn
        self.total_phases = total_phases
        self._phase_num = 0

    def _next_phase_label(self, label: str = "") -> str:
        """Increment phase counter and return a display label like '1/2 pyramids'."""
        self._phase_num += 1
        phase_prefix = f"{self._phase_num}/{self.total_phases}" if self.total_phases > 1 else ""
        if phase_prefix and label:
            return f"{phase_prefix} {label}"
        return phase_prefix or label or ""

    # ------------------------------------------------------------------
    # Drop-in for wait_for_multiple_job_arrays
    # ------------------------------------------------------------------
    def wait_for_arrays(
        self,
        job_arrays: list[dict],
        experiment: str = "",
        verbose: bool = True,
        label: str = "",
        **kwargs,
    ) -> dict:
        """Track job arrays via DAG display instead of internal waiting.

        Returns a dict compatible with ``wait_for_multiple_job_arrays``.
        """
        from cyclops_utils.hpc.launcher_result import LauncherResult

        lr = LauncherResult()
        for arr in job_arrays:
            lr.add_array(
                {
                    "base_job_id": arr["base_job_id"],
                    "submitted_jobs": arr["submitted_jobs"],
                },
                label=arr.get("label", ""),
            )

        # Build name→array_label mapping so we can reconstruct array_results
        # from the flat completed/failed lists returned by track_fn.
        name_to_array_label: dict[str, str] = {}
        for arr in job_arrays:
            arr_label = arr.get("label", "")
            for job_info in arr.get("submitted_jobs", []):
                name_to_array_label[job_info["name"]] = arr_label

        phase_label = self._next_phase_label(label)
        n_failed, completed, failed = self.track_fn(self.step_name, lr, phase_label=phase_label)

        # Reconstruct per-array results so callers (e.g. submit_conversion_batch)
        # that iterate over array_results["label"]["failed"] see real failures.
        array_results: dict[str, dict] = {arr.get("label", ""): {"completed": [], "failed": []} for arr in job_arrays}
        for name in completed:
            arr_label = name_to_array_label.get(name, "")
            if arr_label in array_results:
                array_results[arr_label]["completed"].append(name)
        for name in failed:
            arr_label = name_to_array_label.get(name, "")
            if arr_label in array_results:
                array_results[arr_label]["failed"].append(name)

        return {
            "total_failed": n_failed,
            "completed": completed,
            "failed": failed,
            "array_results": array_results,
        }

    # ------------------------------------------------------------------
    # Drop-in for submit_parallel_jobs(wait_for_completion=True)
    # ------------------------------------------------------------------
    def wait_for_result(
        self,
        submit_result: dict,
        label: str = "",
    ) -> tuple[list[str], list[str]]:
        """Track a single ``submit_parallel_jobs`` result.

        Returns
        -------
        tuple[list[str], list[str]]
            (completed_names, failed_names) — same shape as ``_wait_for_jobs``.
        """
        from cyclops_utils.hpc.launcher_result import LauncherResult

        lr = LauncherResult.from_submit_result(submit_result, label=label)
        phase_label = self._next_phase_label(label)
        _, completed, failed = self.track_fn(self.step_name, lr, phase_label=phase_label)
        return completed, failed

    # ------------------------------------------------------------------
    # Drop-in for monitor_jobs_with_retry
    # ------------------------------------------------------------------
    def wait_for_job_ids(
        self,
        job_ids: list[str],
        job_specs: list[dict],
        label: str = "",
    ) -> dict:
        """Track jobs by ID string via DAG display.

        Returns a dict compatible with ``monitor_jobs_with_retry``.
        Note: retry logic is not available through PhaseTracker —
        if jobs fail, the step fails and the user can re-run.
        """
        lr = launcher_result_from_job_ids(job_ids, job_specs, label=label)
        phase_label = self._next_phase_label(label)
        n_failed, completed, failed = self.track_fn(self.step_name, lr, phase_label=phase_label)

        return {
            "final_job_ids": job_ids,
            "all_succeeded": n_failed == 0,
            "failed_jobs": failed,
            "retry_count": 0,
        }
