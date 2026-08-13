"""
Shared utilities for SLURM batch job submission.

Provides reusable components for submitting parallel SLURM jobs with:
- Job submission and tracking
- Progress monitoring with live timers
- Resource utilization reporting
- Job manifest generation (YAML and Markdown)
- Results aggregation
- Experiment detection and batch processing

Used by various pipeline processing scripts that need to parallelize
work across multiple wells or configurations.
"""

import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Callable, Any
import yaml
import submitit

from cyclops_utils.hpc.slurm_utils import (
    format_time,
    print_slurm_job_stats,
    parse_time_to_seconds,
    parse_memory_value,
)
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.paths import BASE_PATH

# All terminal SLURM job states (sacct will never transition out of these).
# See: https://slurm.schedmd.com/sacct.html#SECTION_JOB-STATE
_SLURM_TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "DEADLINE",
    "BOOT_FAIL", "REVOKED",
}


def _is_terminal_state(state: str) -> bool:
    """Check if a SLURM job state is terminal (job will not change state)."""
    return state in _SLURM_TERMINAL_STATES or state.startswith("CANCELLED")


def check_step_dependencies_satisfied(
    dataset: OpsDataset,
    step_name: str,
    config: dict,
) -> tuple[bool, str]:
    """
    Check if a step's dependencies are satisfied by verifying input files exist.

    Uses the dependency information from slurm_task_config.yaml to determine
    which upstream steps must be complete before this step can run.

    Args:
        dataset: OpsDataset instance for the experiment
        step_name: Name of the step to check (e.g., "build_pyramids")
        config: Experiment configuration dict

    Returns:
        Tuple of (is_ready, reason) where reason explains why if not ready

    Example:
        >>> dataset = OpsDataset("ops0042_20250520")
        >>> ready, reason = check_step_dependencies_satisfied(
        ...     dataset, "build_pyramids", config
        ... )
        >>> if not ready:
        ...     print(f"Cannot run: {reason}")
    """
    # Load SLURM task config to get dependencies
    slurm_config_path = dataset.config_paths["slurm_task_config"]
    try:
        with open(slurm_config_path, "r") as f:
            slurm_config = yaml.safe_load(f) or {}
    except Exception as e:
        return True, f"Could not load SLURM config: {e}"

    # Get dependencies for this step
    step_config = slurm_config.get(step_name, {})
    dependencies = step_config.get("dependencies")

    # If no dependencies or None, step is always ready
    if not dependencies:
        return True, "No dependencies"

    # Check each dependency
    missing_deps = []
    for dep_step in dependencies:
        dep_outputs = dataset.get_output_files_for_step(dep_step, config)
        if dep_outputs is None:
            # No output check defined for this dependency. The step's author
            # explicitly marked it as uncheckable (e.g. `fix_v3_stores` is
            # idempotent fix-up with no single output file). Treat as
            # satisfied so it doesn't block downstream steps.
            continue

        # Check if all dependency outputs exist
        missing = [str(p) for p in dep_outputs if not Path(p).exists()]
        if missing:
            missing_deps.append(
                f"{dep_step}: {len(missing)}/{len(dep_outputs)} missing"
            )

    if missing_deps:
        return False, f"Missing dependencies: {'; '.join(missing_deps)}"

    return True, "All dependencies satisfied"


def detect_experiments_needing_processing(
    input_checker: Callable[[OpsDataset], bool],
    output_checker: Callable[[OpsDataset, Any], list[Path]],
    wells: list[int] = [1, 2, 3],
    force: bool = False,
    verbose: bool = True,
    experiment_filter: Callable[[str], bool] | None = None,
    extra_data_extractor: Callable[[OpsDataset, list[int]], dict] | None = None,
) -> tuple[list[tuple[str, int, int, dict]], list[tuple[str, int, int, dict]]]:
    """
    Generic experiment detection for batch processing.

    Scans $OPS_OUTPUT_BASE_DIR (default $OPS_BASE_PATH) to find experiments
    that need processing based on custom input/output checks.

    Parameters
    ----------
    input_checker : Callable[[OpsDataset], bool]
        Function that checks if experiment has required inputs.
        Should return True if experiment is valid for processing.
    output_checker : Callable[[OpsDataset, Any], list[Path]]
        Function that returns list of expected output paths.
        Should return paths for all expected outputs.
    wells : list[int]
        Wells to check (default: [1, 2, 3])
    force : bool
        If True, include all experiments with valid inputs even if outputs exist
    verbose : bool
        Print progress during scan
    experiment_filter : Callable[[str], bool] | None
        Optional filter function for experiment names. Return True to include.
    extra_data_extractor : Callable[[OpsDataset, list[int]], dict] | None
        Optional function to extract extra metadata (e.g., quality metrics)

    Returns
    -------
    tuple[list, list]
        (experiments_to_process, experiments_completed)
        Each list contains tuples of (experiment_name, n_completed, n_expected, extra_data)

    Examples
    --------
    # For tracking:
    >>> def check_tracking_input(ds):
    ...     return ds.store_paths["lc_5x_phase_2d_stitched_v3"].exists()
    >>> def get_tracking_outputs(ds, wells):
    ...     return [ds.append_well("tracking_geff", f"A/{w}/0") for w in wells]
    >>> detect_experiments_needing_processing(
    ...     input_checker=check_tracking_input,
    ...     output_checker=get_tracking_outputs,
    ...     wells=[1, 2, 3]
    ... )

    # For registration:
    >>> def check_registration_input(ds):
    ...     return (ds.store_paths["lc_5x_segmentation"].exists() and
    ...             ds.store_paths["iss_segmentation"].exists())
    >>> def get_registration_outputs(ds, wells):
    ...     return [ds.append_well("auto_iss_register", f"A/{w}/0") for w in wells]
    >>> detect_experiments_needing_processing(
    ...     input_checker=check_registration_input,
    ...     output_checker=get_registration_outputs,
    ...     wells=[1, 2, 3]
    ... )
    """
    ops_dir = Path(os.environ.get('OPS_OUTPUT_BASE_DIR', f'{BASE_PATH}'))
    # Real experiments match `opsNNNN_YYYYMMDD` (4-digit ID + underscore + 8-digit date).
    # The looser "starts with ops" filter pulled in adjacent project dirs like
    # `ops-paper-analysis`, `ops_data_report`, `ops_reruns`, etc. — inflating
    # the scan count and forcing per-dir input checks that always fail.
    import re
    # `opsNNNN_YYYYMMDD` prefix, with optional arbitrary suffix(es) like
    # `_mark`, `_v2`, `_test_run`, etc.
    _EXP_RE = re.compile(r"^ops\d{4}_\d{8}(_.+)?$")
    experiments = sorted(
        d.name for d in ops_dir.iterdir()
        if d.is_dir() and _EXP_RE.match(d.name)
    )

    experiments_to_process = []
    experiments_completed = []

    if verbose:
        print(f"\nScanning {len(experiments)} experiments...")
        print(f"Wells to check: {wells}")
        print(f"{'='*60}\n")

    for experiment in experiments:
        try:
            # Apply optional experiment filter
            if experiment_filter and not experiment_filter(experiment):
                continue

            dataset = OpsDataset(experiment)

            # Check if experiment has required inputs
            if not input_checker(dataset):
                continue

            # Get expected outputs
            expected_outputs = output_checker(dataset, wells)
            existing_outputs = [p for p in expected_outputs if p.exists()]

            n_completed = len(existing_outputs)
            n_expected = len(expected_outputs)

            # Extract extra metadata if provided
            extra_data = {}
            if extra_data_extractor:
                try:
                    extra_data = extra_data_extractor(dataset, wells)
                except Exception:
                    pass

            # Categorize experiment
            if force and n_expected > 0:
                experiments_to_process.append((experiment, n_completed, n_expected, extra_data))
            elif n_completed < n_expected:
                experiments_to_process.append((experiment, n_completed, n_expected, extra_data))
            elif n_expected > 0 and n_completed == n_expected:
                experiments_completed.append((experiment, n_completed, n_expected, extra_data))

        except Exception as e:
            if verbose:
                print(f"  ✗ Error checking {experiment}: {e}")
            continue

    # Sort experiments numerically by ops number
    def get_ops_number(item: tuple) -> int:
        """Extract numeric ops number from experiment tuple."""
        exp_name = item[0]
        try:
            return int(exp_name.split("_")[0].replace("ops", ""))
        except (ValueError, IndexError):
            return 9999  # Put malformed names at end

    experiments_to_process.sort(key=get_ops_number)
    experiments_completed.sort(key=get_ops_number)

    return experiments_to_process, experiments_completed


def _partition_node_sizes(partition: str, constraint: str | None) -> list[tuple[int, int]]:
    """(cores, mem_GB) per node in ``partition``, restricted to nodes whose features
    satisfy ``constraint`` (e.g. "[a100_80|h100|h200]").

    ``-N`` is required: without it a heterogeneous partition collapses to a single
    "64+ 500000+" row and every node looks like the smallest one. Falls back to a
    single conservative node size if sinfo is unavailable.
    """
    import subprocess

    try:
        out = subprocess.run(["sinfo", "-h", "-N", "-p", partition, "-o", "%c|%m|%f"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception as e:
        print(f"  [dispersion] sinfo query failed ({e}); using fallback node size")
        return [(32, 250)]

    wanted = {t for t in re.split(r"[|&,\[\]()]+", constraint or "") if t.strip()}
    sizes, filtered = [], []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        try:
            entry = (int(parts[0].rstrip("+")), int(parts[1].rstrip("+")) // 1024)
        except ValueError:
            continue
        sizes.append(entry)
        feats = {f.strip() for f in (parts[2] if len(parts) > 2 else "").split(",")}
        if not wanted or feats & wanted:
            filtered.append(entry)
    if filtered:
        return filtered
    return sizes or [(32, 250)]


def _apply_node_dispersion(slurm_params: dict, spread: bool, max_tasks_per_node: int | None) -> None:
    """Raise cpus_per_task/mem so SLURM can't best-fit-pack many array tasks onto one
    node (which saturates that node's CPU/IO/RAM). ``max_tasks_per_node`` (precise)
    sizes the request against the largest node the job's constraint allows, so at most
    K tasks fit per node; ``spread`` (imprecise) applies a fixed bump with no ``sinfo``
    query. Only ever raises the request; no-op unless one is set; max_tasks_per_node
    takes precedence.
    """
    if not spread and not max_tasks_per_node:
        return
    import math

    def _mem_gb(v):
        s = str(v).upper().strip()
        for suf, mul in (("GB", 1), ("G", 1), ("MB", 1 / 1024), ("M", 1 / 1024), ("TB", 1024), ("T", 1024)):
            if s.endswith(suf):
                try:
                    return float(s[: -len(suf)]) * mul
                except ValueError:
                    return 0.0
        try:
            return float(s) / 1024  # bare number is MB in SLURM
        except ValueError:
            return 0.0

    cur_cpus = int(slurm_params.get("cpus_per_task", 1) or 1)
    cur_mem = _mem_gb(slurm_params.get("mem", slurm_params.get("slurm_mem", 0)))
    if max_tasks_per_node:
        K = max(1, int(max_tasks_per_node))
        sizes = _partition_node_sizes(
            slurm_params.get("slurm_partition", ""),
            slurm_params.get("slurm_constraint"),
        )
        max_c, max_m = max(c for c, _ in sizes), max(m for _, m in sizes)
        min_c, min_m = min(c for c, _ in sizes), min(m for _, m in sizes)
        # Exceed largest_node/(K+1) so a (K+1)th task cannot fit anywhere, then
        # clamp to the smallest node so the request stays schedulable there.
        want_c, want_m = math.floor(max_c / (K + 1)) + 1, math.floor(max_m / (K + 1)) + 1
        cpus, mem_gb = min(want_c, min_c), max(1, min(want_m, min_m))
        note = (f"max_tasks_per_node={K} (eligible nodes "
                f"{min_c}c/{min_m}G..{max_c}c/{max_m}G)")
        if want_c > min_c and want_m > min_m:
            note += f"; clamped to smallest node, >{K} may still pack on the largest"
    else:
        cpus, mem_gb, note = 8, 48, "spread=True (fixed 8c/48G)"
    new_cpus, new_mem = max(cur_cpus, cpus), max(int(cur_mem), mem_gb)
    slurm_params["cpus_per_task"] = new_cpus
    slurm_params["mem"] = f"{new_mem}GB"
    print(f"  [dispersion] {note} -> {new_cpus} cpus, {new_mem}GB per task")


def submit_parallel_jobs(
    jobs_to_submit: list[dict],
    experiment: str,
    slurm_params: dict,
    log_dir: str | Path = "slurm_logs",
    manifest_prefix: str = "batch",
    dry_run: bool = False,
    wait_for_completion: bool = True,
    verbose: bool = True,
    post_completion_callback: Callable[[list[dict], str], Any] | None = None,
    print_resource_summary: bool = True,
    print_success: bool = True,
    step_name: str | None = None,
    spread: bool = False,
    max_tasks_per_node: int | None = None,
) -> dict:
    """
    Submit parallel SLURM jobs with monitoring and reporting.

    Parameters
    ----------
    jobs_to_submit : list[dict]
        List of job specifications, each containing:
        - name: Job display name
        - func: Function to execute
        - kwargs: Dict of keyword arguments to pass to func
        - metadata: Optional dict with additional tracking info (e.g., type, well)
    experiment : str
        Experiment name for logging and reporting
    slurm_params : dict
        SLURM parameters (timeout_min, mem, cpus_per_task, slurm_partition, etc.)
    log_dir : str or Path
        Directory for SLURM logs (default: "slurm_logs")
    manifest_prefix : str
        Prefix for manifest files (default: "batch")
    dry_run : bool
        If True, print plan without submitting (default: False)
    wait_for_completion : bool
        If True, wait for all jobs to complete (default: True)
    verbose : bool
        Print detailed progress (per-job completion messages) (default: True)
    post_completion_callback : Callable[[list[dict], str], Any] | None
        Optional callback to run after jobs complete. Receives:
        - submitted_jobs: List of job info dicts with results
        - experiment: Experiment name
    print_resource_summary : bool
        Print SLURM resource utilization summary at the end (default: True)
    step_name : str, optional
        Parent step identifier for central log mirror. When set and OPS_MODE
        is 'operational', the SLURM log directory is copied to
        ``${OPS_LOG_ROOTDIR}/<experiment>/<step_name>/<OPS_RUN_TS>/`` after
        all jobs reach a terminal state (success or failure). No-op when
        step_name is None or in research mode.
    spread : bool
        Node-dispersion (imprecise). SLURM best-fit-**packs** array tasks, so many
        small jobs can pile onto one node and saturate its CPU/IO. When True, bump
        cpus_per_task/mem to fixed fractions (8 cpus, 48GB) so fewer tasks fit per
        node and they spread. Only raises the request, never lowers a larger one.
        Off by default. Ignored if ``max_tasks_per_node`` is set. Tradeoff: bigger
        requests may queue more when the partition is busy.
    max_tasks_per_node : int, optional
        Node-dispersion (precise). Cap how many array tasks may share a node: the
        util queries ``sinfo`` for the smallest node in the partition and sizes
        cpus_per_task/mem so at most K tasks pack per node. Only raises the request.
        Off by default; takes precedence over ``spread``.

    Returns
    -------
    dict
        Job submission results with:
        - success: bool
        - base_job_id: str (SLURM array job ID)
        - jobs: list of job metadata
        - manifest_yaml: Path to YAML manifest
        - manifest_md: Path to Markdown manifest
        - completed: list of completed job names (if wait_for_completion)
        - failed: list of failed job names (if wait_for_completion)
        - all_completed: bool (if wait_for_completion)
    """
    if not jobs_to_submit:
        print("No jobs to submit!")
        return {"success": False, "error": "No jobs to submit"}

    # Prepend slurm_logs/ to log_dir to keep all SLURM logs in a central location
    log_dir = f"slurm_logs/{log_dir}"

    # Optional node dispersion: raise per-task cpus/mem so SLURM spreads array tasks
    # across nodes instead of packing them (off by default).
    _apply_node_dispersion(slurm_params, spread, max_tasks_per_node)

    # Print submission plan
    print(f"\n{'='*60}")
    print(f"SLURM Batch Submission")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Total jobs: {len(jobs_to_submit)}")
    print(f"\nSLURM Resources (per job):")
    print(f"  Timeout: {slurm_params['timeout_min']} min")
    print(f"  Memory: {slurm_params.get('mem', slurm_params.get('slurm_mem', 'N/A'))}")
    print(f"  CPUs: {slurm_params['cpus_per_task']}")
    print(f"  Partition: {slurm_params['slurm_partition']}")
    if slurm_params.get('slurm_gres'):
        print(f"  GPUs: {slurm_params['slurm_gres']}")
    if slurm_params.get('slurm_constraint'):
        print(f"  Constraint: {slurm_params['slurm_constraint']}")
    print(f"\nJobs to submit:")
    max_display = 3
    for i, job in enumerate(jobs_to_submit[:max_display], 1):
        print(f"  {i}. {job['name']}")
    if len(jobs_to_submit) > max_display:
        print(f"  ... and {len(jobs_to_submit) - max_display} more")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN: No jobs submitted")
        return {"dry_run": True, "jobs": jobs_to_submit}

    # Set up submitit executor
    log_path = Path(log_dir) / "%j"
    # Extract slurm_python if provided (needs to go to constructor, not update_parameters)
    slurm_python = slurm_params.pop("slurm_python", None)
    if slurm_python:
        executor = submitit.AutoExecutor(folder=str(log_path), slurm_python=slurm_python)
    else:
        executor = submitit.AutoExecutor(folder=str(log_path))
    # Apply QoS if set via --slurm-tag or OPS_SLURM_QOS env var
    slurm_qos = os.environ.get("OPS_SLURM_QOS")
    if slurm_qos and "slurm_qos" not in slurm_params:
        slurm_params["slurm_qos"] = slurm_qos

    # Exclude bad nodes via env var (comma-separated, e.g. OPS_SLURM_EXCLUDE="gpu-h-8,gpu-h-1")
    # TODO(2026-04-11): TEMPORARY hardcoded exclude for gpu-h-8 (slow NFS I/O causing timeouts).
    # Remove this block once the node is fixed. The env var override still works independently.
    # if "slurm_exclude" not in slurm_params:
    #     slurm_params["slurm_exclude"] = "gpu-h-5,gpu-h-8"
    #     print("  [WARN] Excluding gpu-h-5,gpu-h-8 (known slow nodes — remove hardcode when fixed)")
    slurm_exclude = os.environ.get("OPS_SLURM_EXCLUDE")
    if slurm_exclude and "slurm_exclude" not in slurm_params:
        slurm_params["slurm_exclude"] = slurm_exclude

    executor.update_parameters(**slurm_params)

    # Set descriptive job name so squeue shows experiment info instead of "submitit"
    job_name = experiment[:128]  # SLURM truncates at 128 chars
    executor.update_parameters(slurm_job_name=job_name)

    # Disable CPU binding to avoid srun binding failures on some nodes
    executor.update_parameters(slurm_srun_args=["--cpu-bind=none"])

    # Submit jobs (single vs array)
    submitted_jobs = []
    single_mode = len(jobs_to_submit) == 1

    # Route through slurm_worker_run so each per-job SLURM worker has an
    # active ExperimentNotifier (shared thread_ts via JSON cache) — required
    # for any @notify_step decorator inside the submitted func to fire its
    # `attachments` resolver into the experiment's existing Slack thread.
    # Imported here rather than at module scope so callers that only want the
    # batch helpers don't pull in slack_sdk and the notifier's import-time
    # auth_test(); see the note in profiling/decorators.py.
    from cyclops_utils.profiling.slack_notifier import slurm_worker_run as _slurm_worker_run

    if single_mode:
        print(f"Submitting 1 job...")
        job_info = jobs_to_submit[0]
        job = executor.submit(
            _slurm_worker_run, job_info["func"], experiment, (), dict(job_info["kwargs"])
        )
        submitted_jobs.append({
            "job": job,
            "name": job_info["name"],
            "metadata": job_info.get("metadata", {}),
        })
    else:
        print(f"Submitting {len(jobs_to_submit)} jobs as batch array...")
        with executor.batch():
            for job_info in jobs_to_submit:
                job = executor.submit(
                    _slurm_worker_run, job_info["func"], experiment, (), dict(job_info["kwargs"])
                )
                submitted_jobs.append({
                    "job": job,
                    "name": job_info["name"],
                    "metadata": job_info.get("metadata", {}),
                })

    # Extract job IDs and save manifest
    if submitted_jobs:
        first_job_id = str(submitted_jobs[0]["job"].job_id)
        base_job_id = first_job_id if single_mode else first_job_id.split("_")[0]

        # Save job manifest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_dir = Path("slurm_logs/slurm_job_manifests")
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Prepare manifest data
        manifest_data: dict = {
            "master_job_id": base_job_id,
            "submitted": timestamp,
            "experiment": experiment,
            "total_jobs": len(submitted_jobs),
            "resources": {
                "timeout_min": slurm_params["timeout_min"],
                "mem": slurm_params.get("mem", slurm_params.get("slurm_mem", "N/A")),
                "cpus": slurm_params["cpus_per_task"],
                "partition": slurm_params["slurm_partition"],
            },
        }

        if single_mode:
            manifest_jobs = [
                {
                    "job_id": base_job_id,
                    "name": j["name"],
                    **j["metadata"],
                }
                for j in submitted_jobs
            ]
        else:
            manifest_jobs = [
                {
                    "array_index": idx,
                    "job_id": f"{base_job_id}_{idx}",
                    "name": j["name"],
                    **j["metadata"],
                }
                for idx, j in enumerate(submitted_jobs)
            ]
        manifest_data["jobs"] = manifest_jobs

        # Save as YAML
        yaml_file = manifest_dir / f"{manifest_prefix}_{base_job_id}_{timestamp}.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

        # Save as Markdown
        md_file = manifest_dir / f"{manifest_prefix}_{base_job_id}_{timestamp}.md"
        with open(md_file, "w") as f:
            if single_mode:
                f.write(f"# SLURM Job: {base_job_id}\n\n")
            else:
                f.write(f"# SLURM Batch: {base_job_id}\n\n")
            f.write(f"**Submitted:** {timestamp}\n\n")
            f.write(f"**Experiment:** {experiment}\n\n")
            mem_val = slurm_params.get("mem", slurm_params.get("slurm_mem", "N/A"))
            f.write(
                f"**Resources:** {slurm_params['timeout_min']}min, {mem_val}, {slurm_params['cpus_per_task']} CPUs\n\n"
            )
            f.write(f"**Total Jobs:** {len(submitted_jobs)}\n\n")
            f.write("## Job Details\n\n")
            if single_mode:
                f.write("| Job ID | Name |\n")
                f.write("|--------|------|\n")
                for j in submitted_jobs:
                    f.write(f"| {base_job_id} | {j['name']} |\n")
            else:
                f.write("| Array Index | Job ID | Name |\n")
                f.write("|-------------|--------|------|\n")
                for idx, j in enumerate(submitted_jobs):
                    f.write(f"| {idx} | {base_job_id}_{idx} | {j['name']} |\n")
            f.write("\n## Management Commands\n\n")
            if single_mode:
                f.write(f"- **Check status:** `squeue -j {base_job_id}`\n")
                f.write(f"- **Cancel:** `scancel {base_job_id}`\n")
            else:
                f.write(f"- **Check status:** `squeue -u $USER | grep {base_job_id}`\n")
                f.write(f"- **Cancel all:** `scancel {base_job_id}`\n")
                f.write(f"- **Cancel one:** `scancel {base_job_id}_<array_index>`\n")
                f.write(f"- **View logs:** `ls {log_dir}/`\n")

        # Print summary
        print(f"\n{'='*60}")
        if single_mode:
            print(f"✓ Submitted 1 job")
            print(f"  Job ID: {base_job_id}")
        else:
            print(
                f"✓ Submitted {len(submitted_jobs)} jobs as array under master job: {base_job_id}"
            )
            print(f"  Job IDs: {base_job_id}_0 to {base_job_id}_{len(submitted_jobs)-1}")
        print(f"  Experiment: {experiment}")
        mem_display = slurm_params.get("mem", slurm_params.get("slurm_mem", "N/A"))
        print(
            f"  Resources: {slurm_params['timeout_min']}min, {mem_display}, {slurm_params['cpus_per_task']} CPUs"
        )
        print(f"\nJob manifest saved:")
        print(f"  YAML: {yaml_file}")
        print(f"  Markdown: {md_file}")
        print(f"\nManagement:")
        if single_mode:
            print(f"  Check status: squeue -j {base_job_id}")
            print(f"  Cancel: scancel {base_job_id}")
        else:
            print(f"  Check status: squeue -u $USER | grep {base_job_id}")
            print(f"  Cancel all: scancel {base_job_id}")
            print(f"  Cancel one: scancel {base_job_id}_<array_index>")
        print(f"  View logs: ls {log_dir}/")
        print(f"{'='*60}\n")

        # Wait for all jobs to complete if requested
        if wait_for_completion:
            # If running in DAG context, delegate waiting to PhaseTracker
            # for live progress in the DAG table
            _tracker = None
            try:
                from cyclops_utils.hpc.phase_tracker import _current_phase_tracker
                _tracker = _current_phase_tracker.get(None)
            except (ImportError, LookupError):
                pass

            if _tracker:
                # Build a minimal submit result dict for the tracker
                _submit_result = {
                    "base_job_id": base_job_id,
                    "submitted_jobs": submitted_jobs,
                }
                completed, failed = _tracker.wait_for_result(
                    _submit_result, label=experiment,
                )
            else:
                completed, failed = _wait_for_jobs(
                    submitted_jobs=submitted_jobs,
                    base_job_id=base_job_id,
                    slurm_params=slurm_params,
                    experiment=experiment,
                    verbose=verbose,
                    print_resource_summary=print_resource_summary,
                    print_success=print_success,
                )

            # Run post-completion callback if provided
            if post_completion_callback is not None:
                try:
                    post_completion_callback(submitted_jobs, experiment)
                except Exception as e:
                    print(f"\n⚠️  Post-completion callback failed: {e}")

            # Phase 4: mirror the SLURM log directory to the central store
            # when running in operational mode. No-op otherwise. Wrapped in
            # try/except — never let log-mirror failures break the caller.
            if step_name:
                try:
                    from cyclops_utils.ops_mode import mirror_slurm_log_dir
                    mirror_slurm_log_dir(
                        log_dir, experiment, step_name, job_id=str(base_job_id),
                    )
                except Exception as _e:
                    print(f"[ops_mode] warn: fanout central log mirror "
                          f"failed for {step_name}: {_e}", file=sys.stderr)

            # Roll up per-subtask GPU/CPU metrics into a per-step summary.
            # Each subtask writes its own log_key (e.g.
            # ``segment_single_position_posA_1_006006``) via the
            # ``@versioned_function`` decorator; this aggregates them under
            # ``<step_name>__summary`` so reporting tools don't have to walk
            # every per-task entry. Best-effort: never let a metrics roll-up
            # failure crash the actual work.
            if step_name and jobs_to_submit:
                worker_func = jobs_to_submit[0].get("func")
                worker_name = getattr(worker_func, "__name__", None) if worker_func else None
                if worker_name:
                    try:
                        from cyclops_utils.profiling.decorators import (
                            emit_subtask_summary,
                        )
                        emit_subtask_summary(
                            experiment,
                            child_key_prefix=worker_name,
                            summary_key=step_name,
                        )
                    except Exception as _e:
                        print(f"[summary] warn: emit_subtask_summary for "
                              f"{step_name} failed: {_e}", file=sys.stderr)

            return {
                "success": True,
                "base_job_id": base_job_id,
                "jobs": manifest_data["jobs"],
                "manifest_yaml": str(yaml_file),
                "manifest_md": str(md_file),
                "completed": completed,
                "failed": failed,
                "all_completed": len(failed) == 0,
            }

        return {
            "success": True,
            "base_job_id": base_job_id,
            "jobs": manifest_data["jobs"],
            "submitted_jobs": submitted_jobs,  # Include job objects for manual waiting
            "manifest_yaml": str(yaml_file),
            "manifest_md": str(md_file),
            "completed": None,
            "failed": None,
            "all_completed": None,
        }

    return {"success": False, "error": "No jobs submitted"}


def _print_batch_resource_summary(
    job_list: list[dict],
    experiment: str,
    slurm_params: dict | None,
    failed_jobs: list,
) -> None:
    """
    Print a single aggregated resource summary table for batches with >10 jobs.

    Instead of individual per-job tables, queries sacct for all completed jobs
    and displays min-max ranges for key metrics in one table.
    """
    import time as time_module

    # Build failed name set
    failed_names = set()
    for f in failed_jobs:
        failed_names.add(f[0] if isinstance(f, tuple) else f)

    # Collect job IDs for completed jobs
    completed_ids = set()
    for job_info in job_list:
        if job_info["name"] not in failed_names:
            completed_ids.add(str(job_info["job"].job_id))

    n_total = len(job_list)
    n_failed = len(failed_jobs)
    n_completed = n_total - n_failed

    if not completed_ids:
        return

    time_module.sleep(2)

    # Get unique base IDs for efficient sacct queries
    base_ids = set()
    for jid in completed_ids:
        base_ids.add(jid.split("_")[0] if "_" in jid else jid)

    fields = [
        "JobID", "Partition", "NodeList", "Elapsed", "Timelimit",
        "AllocCPUS", "MaxRSS", "MaxDiskRead", "MaxDiskWrite",
        "TotalCPU", "CPUTime", "ReqMem", "State",
    ]

    # Batch query sacct by base ID and merge step data
    all_data = {}
    for base_id in base_ids:
        try:
            cmd = [
                "sacct", "-j", base_id,
                "--format", ",".join(fields),
                "--parsable2", "--noheader",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                continue

            # Group records by task ID (strip .batch/.extern/.0 suffixes)
            task_records = {}
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) < len(fields):
                    continue
                record = dict(zip(fields, parts))
                task_id = record["JobID"].split(".")[0]
                if task_id not in task_records:
                    task_records[task_id] = []
                task_records[task_id].append(record)

            # Merge step data into main record (same logic as print_slurm_job_stats)
            for task_id, records in task_records.items():
                if task_id not in completed_ids:
                    continue
                data = records[0]
                step = None
                for r in records:
                    if ".0" in r["JobID"]:
                        step = r
                        break
                if not step:
                    for r in records:
                        if ".batch" in r["JobID"]:
                            step = r
                            break
                if step:
                    for key in ["MaxRSS", "MaxDiskRead", "MaxDiskWrite", "TotalCPU"]:
                        if step.get(key, "").strip():
                            data[key] = step[key]
                all_data[task_id] = data

        except Exception:
            continue

    if not all_data:
        print(f"\n⚠️  Could not retrieve resource stats for batch jobs")
        return

    # Collect per-job metrics and aggregate
    partitions = set()
    nodes = set()
    cpus_set = set()
    job_metrics = []

    for data in all_data.values():
        m = {}
        m["elapsed"] = parse_time_to_seconds(data.get("Elapsed", ""))
        m["timelimit"] = parse_time_to_seconds(data.get("Timelimit", ""))
        m["cpu_used"] = parse_time_to_seconds(data.get("TotalCPU", ""))
        m["cpu_avail"] = parse_time_to_seconds(data.get("CPUTime", ""))
        m["max_rss"] = parse_memory_value(data.get("MaxRSS", ""))

        req_str = data.get("ReqMem", "")
        if req_str and len(req_str) > 1 and req_str[-1].isalpha():
            if req_str[-2].isalpha():
                req_str = req_str[:-1]
        m["req_mem"] = parse_memory_value(req_str)
        m["disk_read"] = parse_memory_value(data.get("MaxDiskRead", ""))
        m["disk_write"] = parse_memory_value(data.get("MaxDiskWrite", ""))

        p = data.get("Partition", "").strip()
        if p:
            partitions.add(p)
        n = data.get("NodeList", "").strip()
        if n:
            nodes.add(n)
        c = data.get("AllocCPUS", "").strip()
        if c:
            cpus_set.add(c)

        job_metrics.append(m)

    # Formatting helpers
    def fmt_time(s):
        h, m, sec = int(s) // 3600, (int(s) % 3600) // 60, int(s) % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def fmt_range(key, formatter):
        vals = [m[key] for m in job_metrics if m[key] > 0]
        if not vals:
            return "-"
        mn, mx = min(vals), max(vals)
        return formatter(mn) if mn == mx else f"{formatter(mn)} - {formatter(mx)}"

    def fmt_range_with_zero(key, formatter):
        vals = [m[key] for m in job_metrics]
        if not vals:
            return "-"
        mn, mx = min(vals), max(vals)
        return formatter(mn) if mn == mx else f"{formatter(mn)} - {formatter(mx)}"

    def fmt_gb(b):
        return f"{b / (1024**3):.1f}GB" if b > 0 else "0.0GB"

    def pct_range(num_key, denom_key):
        pcts = []
        for m in job_metrics:
            if m[denom_key] > 0 and m[num_key] > 0:
                pcts.append(m[num_key] / m[denom_key] * 100)
        if not pcts:
            return ""
        mn, mx = min(pcts), max(pcts)
        return f"{mn:.1f}%" if abs(mn - mx) < 0.1 else f"{mn:.1f}-{mx:.1f}%"

    # Print table
    print(f"\n{'='*60}")
    print(f"SLURM Batch Resource Summary ({len(all_data)} jobs)")
    print(f"{'='*60}")

    print("\n┌─────────────────────────┬──────────────────────────┬──────────────┐")
    print("│ Batch Resource Summary  │ Value                    │ Utilization  │")
    print("├─────────────────────────┼──────────────────────────┼──────────────┤")

    def row(label, value, util=""):
        v = str(value)[:24]
        u = str(util)[:12]
        print(f"│ {label:<23} │ {v:<24} │ {u:<12} │")

    row("Experiment", experiment)
    row("Total Jobs", n_total)
    row("Completed", n_completed)
    if n_failed > 0:
        row("Failed", n_failed)

    parts_str = ", ".join(sorted(partitions)) or "-"
    row("Partition(s)", parts_str[:24])

    nodes_str = ", ".join(sorted(nodes)) if len(nodes) <= 3 else f"{len(nodes)} nodes"
    row("Node(s)", nodes_str[:24])

    cpus_str = ", ".join(sorted(cpus_set)) or "-"
    row("Allocated CPUs", cpus_str)

    row("Elapsed Time", fmt_range("elapsed", fmt_time),
        pct_range("elapsed", "timelimit"))
    row("Time Limit", fmt_range("timelimit", fmt_time))
    row("CPU Time Used (Max)", fmt_range("cpu_used", fmt_time),
        pct_range("cpu_used", "cpu_avail"))
    row("Max RSS Memory", fmt_range("max_rss", fmt_gb),
        pct_range("max_rss", "req_mem"))
    row("Memory Requested", fmt_range("req_mem", fmt_gb))
    row("Max Disk Read", fmt_range_with_zero("disk_read", fmt_gb))
    row("Max Disk Write", fmt_range_with_zero("disk_write", fmt_gb))

    print("└─────────────────────────┴──────────────────────────┴──────────────┘")

    # Print failed jobs
    if failed_jobs:
        print(f"\n⚠️  {len(failed_jobs)} job(s) failed:")
        for failed_item in failed_jobs:
            if isinstance(failed_item, tuple):
                name, job_id = failed_item
                print(f"  - {name} [job: {job_id}]")
            else:
                print(f"  - {failed_item}")
        print(f"\nCheck logs for details\n")


def _print_resource_summary(
    job_list: list[dict],
    job_stats: dict,
    experiment: str,
    slurm_params: dict | None,
    failed_jobs: list[str],
) -> None:
    """
    Print SLURM resource utilization summary for completed jobs.

    Args:
        job_list: List of job info dicts with 'name', 'job', and optionally 'array_label' and 'slurm_params'
        job_stats: Dict mapping job names to their stats
        experiment: Experiment name
        slurm_params: Default SLURM parameters for timeout display (used if job doesn't have its own)
        failed_jobs: List of failed job names
    """
    # For large batches (>3 jobs), print a single aggregated summary table
    if len(job_list) > 3:
        _print_batch_resource_summary(job_list, experiment, slurm_params, failed_jobs)
        return

    if job_stats:
        print(f"{'='*60}")
        print(f"SLURM Resource Utilization Summary")
        print(f"{'='*60}\n")

        for job_info in job_list:
            name = job_info["name"]
            job_id = str(job_info["job"].job_id)

            # Only print stats if we have them
            if name in job_stats:
                # Check if this is from a multi-array monitoring (has array_label)
                if "array_label" in job_info:
                    array_label = job_info["array_label"]
                    print(f"Job: {name} ({array_label}) (ID: {job_id})")
                else:
                    print(f"Job: {name} (ID: {job_id})")

                print(f"-" * 60)

                # Use job-specific slurm_params if available, otherwise use default
                job_slurm_params = job_info.get("slurm_params", slurm_params)

                print_slurm_job_stats(
                    job_id=job_id,
                    experiment=experiment,
                    slurm_params=job_slurm_params,
                    queue_time_sec=0,
                )
                print()  # Blank line between jobs

    # Print failed jobs summary if any
    if failed_jobs:
        print(f"\n⚠️  {len(failed_jobs)} job(s) failed:")
        for failed_item in failed_jobs:
            # Handle both (name, job_id) tuples and plain names for backwards compatibility
            if isinstance(failed_item, tuple):
                name, job_id = failed_item
                print(f"  - {name} [job: {job_id}]")
            else:
                print(f"  - {failed_item}")
        print(f"\nCheck logs for details\n")


def _print_progress_update(
    completed_count: int,
    failed_count: int,
    total_jobs: int,
    start_time: float | None,
    submission_time: float,
    timeout_min: int,
) -> None:
    """
    Print live progress update for job monitoring.

    Args:
        completed_count: Number of completed jobs
        failed_count: Number of failed jobs
        total_jobs: Total number of jobs
        start_time: Time when first job started running (None if still queued)
        submission_time: Time when jobs were submitted
        timeout_min: Timeout in minutes for display purposes
    """
    current_time = time.time()
    n_done = completed_count + failed_count

    # Calculate percentage
    pct = (n_done / total_jobs) * 100 if total_jobs > 0 else 0

    # Build timer string - different display if jobs haven't started yet
    if start_time is None:
        # Jobs are still queued
        queue_time = int(current_time - submission_time)
        timer_str = f"⏳ Progress: {n_done}/{total_jobs} ({pct:.0f}%) | Queued for {format_time(queue_time)} (waiting for jobs to start...)"
    else:
        # Jobs are running - show elapsed runtime
        total_elapsed = int(current_time - start_time)

        timeout_sec = timeout_min * 60
        elapsed_pct = (total_elapsed / timeout_sec) * 100 if timeout_sec > 0 else 0

        # Estimate time remaining based on average job completion time
        if n_done > 0:
            avg_time_per_job = total_elapsed / n_done
            remaining_jobs = total_jobs - n_done
            est_remaining = int(avg_time_per_job * remaining_jobs)
            eta_str = f" | ETA: {format_time(est_remaining)}"
        else:
            eta_str = ""

        timer_str = (
            f"⏳ Progress: {n_done}/{total_jobs} ({pct:.0f}%) | "
            f"Runtime: {format_time(total_elapsed)} of {timeout_min}min ({elapsed_pct:.0f}%){eta_str}"
        )

    sys.stdout.write(f"\r{timer_str}")
    sys.stdout.flush()


def wait_for_multiple_job_arrays(
    job_arrays: list[dict],
    experiment: str,
    verbose: bool = True,
    print_resource_summary: bool = True,
    print_success: bool = True,
) -> dict:
    """
    Wait for multiple job arrays to complete with unified progress monitoring.

    Parameters
    ----------
    job_arrays : list[dict]
        List of job array info, each containing:
        - submitted_jobs: list of job dicts with 'job' and 'name'
        - base_job_id: array job ID
        - label: descriptive label (e.g., "base images", "segmentation")
        - slurm_params: SLURM parameters dict
    experiment : str
        Experiment name
    verbose : bool
        Print detailed progress (per-job completion messages)
    print_resource_summary : bool
        Print SLURM resource utilization summary at the end (default: True)

    Returns
    -------
    dict
        Results with 'completed' and 'failed' lists for each array
    """
    # If running in DAG context, delegate to PhaseTracker for live progress display
    try:
        from cyclops_utils.hpc.phase_tracker import _current_phase_tracker
        tracker = _current_phase_tracker.get(None)
        if tracker:
            # Build a phase label from the array labels (e.g. "image+seg+overlay")
            array_labels = [a.get("label", "") for a in job_arrays if a.get("label")]
            phase_label = "+".join(array_labels) if array_labels else ""
            return tracker.wait_for_arrays(
                job_arrays, experiment=experiment, verbose=verbose,
                label=phase_label,
            )
    except (ImportError, LookupError):
        pass

    # Combine all jobs with array labels and their specific slurm_params
    all_jobs = []
    for array_info in job_arrays:
        for job_info in array_info["submitted_jobs"]:
            all_jobs.append({
                **job_info,
                "array_label": array_info["label"],
                "array_id": array_info["base_job_id"],
                "slurm_params": array_info["slurm_params"],
            })

    total_jobs = len(all_jobs)

    # Build array info string
    array_info_str = ", ".join([f"{arr['label']} ({arr['base_job_id']})" for arr in job_arrays])

    print(f"\nWaiting for {total_jobs} jobs across {len(job_arrays)} arrays to complete...")
    print(f"Arrays: {array_info_str}")
    print(f"Press Ctrl+C to abort waiting (jobs will continue running)\n")

    submission_time = time.time()
    start_time = None
    completed = []  # List of completed job names
    completed_set = set()  # Set of job_ids for fast lookup (not names — names can collide
                           # across arrays when the same position+group appears in multiple stores)
    failed = []  # List of (name, job_id) tuples for output
    failed_set = set()  # Set of job_ids for fast lookup
    job_stats = {}
    last_update = 0
    update_interval = 5
    jobs_running = set()

    # Track per-array results
    array_results = {arr["label"]: {"completed": [], "failed": []} for arr in job_arrays}

    # Build job_id -> job_info mapping for fast lookup
    job_id_to_info = {str(j["job"].job_id): j for j in all_jobs}

    # Get base job ID for batch queries (all array jobs share the base)
    base_job_ids = set()
    for arr in job_arrays:
        base_job_ids.add(arr["base_job_id"])

    try:
        while len(completed) + len(failed) < total_jobs:
            # Batch query SLURM for all job states at once (much faster than individual queries)
            job_states = {}
            for base_id in base_job_ids:
                try:
                    cmd = ["sacct", "-j", base_id, "--format=JobID,State", "-n", "-P"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split("\n"):
                            if "|" in line:
                                parts = line.split("|")
                                jid = parts[0].split(".")[0]  # Remove .batch suffix
                                state = parts[1]
                                job_states[jid] = state
                except Exception:
                    pass

            # Check if any jobs started running (for timer)
            if start_time is None:
                for jid, state in job_states.items():
                    if state == "RUNNING":
                        start_time = time.time()
                        print("⏱️  First job started running, timer started...\n")
                        break

            # Process completed/failed jobs
            for job_info in all_jobs:
                job = job_info["job"]
                name = job_info["name"]
                array_label = job_info["array_label"]
                job_id = str(job.job_id)

                if job_id in completed_set or job_id in failed_set:
                    continue

                # Check status from batch query
                status = job_states.get(job_id, "")
                if _is_terminal_state(status):
                    # Clear the progress line before printing completion
                    sys.stdout.write("\r" + " " * 120 + "\r")
                    sys.stdout.flush()

                    if status == "COMPLETED":
                        completed.append(name)
                        completed_set.add(job_id)
                        array_results[array_label]["completed"].append(name)
                        # Get job stats (skip for speed, can be done post-hoc)
                        try:
                            sacct_result = job.get_info()
                            job_stats[name] = sacct_result
                        except Exception:
                            pass

                        if print_success:
                            total_str = format_time(int(time.time() - submission_time))
                            run_str = format_time(int(time.time() - start_time)) if start_time else "queued"
                            print(f"  ✓ {name} ({array_label}) completed ({len(completed)}/{total_jobs}) [job: {job_id}] [walltime: {total_str}, runtime: {run_str}]")
                    else:
                        failed.append((name, job_id))  # Store tuple with job_id
                        failed_set.add(job_id)
                        array_results[array_label]["failed"].append(name)
                        total_str = format_time(int(time.time() - submission_time))
                        run_str = format_time(int(time.time() - start_time)) if start_time else "queued"
                        print(f"  ✗ {name} ({array_label}) FAILED ({len(failed)}/{total_jobs}) [job: {job_id}] [walltime: {total_str}, runtime: {run_str}]")

            # Progress update
            current_time = time.time()
            if current_time - last_update >= update_interval:
                # Get timeout for display - use first array's timeout
                timeout_min = job_arrays[0]["slurm_params"].get("timeout_min", 20)
                _print_progress_update(
                    completed_count=len(completed),
                    failed_count=len(failed),
                    total_jobs=total_jobs,
                    start_time=start_time,
                    submission_time=submission_time,
                    timeout_min=timeout_min,
                )
                last_update = current_time

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  User interrupted waiting. Jobs will continue running.")
        for arr in job_arrays:
            print(f"  Check status: squeue -u $USER | grep {arr['base_job_id']}")
            print(f"  Cancel all: scancel {arr['base_job_id']}")
        return {"interrupted": True, "array_results": array_results}

    # Final summary
    runtime_seconds = time.time() - start_time if start_time else 0
    runtime_minutes = runtime_seconds / 60

    print(f"\n{'='*60}")
    if failed:
        print(f"Jobs finished in {runtime_minutes:.1f} minutes runtime")
    else:
        print(f"All jobs finished in {runtime_minutes:.1f} minutes runtime")

    for arr_label, results in array_results.items():
        print(f"  {arr_label}: {len(results['completed'])}/{len(results['completed']) + len(results['failed'])} completed")

    print(f"{'='*60}\n")

    # Print resource utilization for all jobs
    if print_resource_summary:
        _print_resource_summary(
            job_list=all_jobs,
            job_stats=job_stats,
            experiment=experiment,
            slurm_params=None,  # Each job has its own slurm_params stored
            failed_jobs=failed,
        )

    return {"array_results": array_results, "all_completed": len(failed) == 0}


def monitor_slurm_arrays(
    arrays: list[dict],
    poll_interval: int = 15,
    verbose: bool = True,
) -> dict:
    """
    Watch already-submitted SLURM (array) jobs by job ID until all reach a
    terminal state, printing live aggregate progress.

    Unlike ``submit_parallel_jobs`` / ``wait_for_multiple_job_arrays``, this does
    NOT need submitit ``Job`` objects — it polls ``sacct`` by base job ID, so it
    works for jobs submitted via raw ``sbatch``, submitit ``map_array``, or a
    dependency chain (array -> concat -> convert). Use it to "watch the arrays"
    after a fire-and-forget submission instead of the process silently exiting.

    Parameters
    ----------
    arrays : list[dict]
        One entry per (array) job to watch. Each dict:
        - ``base_job_id`` (str | int): array master id (e.g. ``"34310229"``) or a
          single job id. Any ``_taskidx`` suffix is stripped.
        - ``label`` (str): display label. Defaults to the base id.
        - ``total`` (int): expected number of array tasks. Defaults to 1.
          Needed because ``sacct`` collapses not-yet-started array tasks into a
          single compressed range row (``12345_[6-99%100]``), so they can't be
          counted individually until they start.
    poll_interval : int
        Seconds between ``sacct`` polls (default 15).
    verbose : bool
        Print per-task completion lines for small arrays (``total`` <= 20).
        Failures are always printed regardless of size.

    Returns
    -------
    dict
        ``{label: {"completed": int, "failed": int, "total": int,
        "failed_ids": [...]}}`` plus a top-level ``"all_completed"`` bool.
        Includes ``"interrupted": True`` if the user Ctrl+C'd out.
    """
    # Non-terminal states that mean "task exists but hasn't finished" (vs RUNNING).
    _PENDING_STATES = {"PENDING", "REQUEUED", "RESIZING", "SUSPENDED", "REQUEUE_HOLD"}

    state = {}
    for a in arrays:
        base = str(a["base_job_id"]).split("_")[0]
        label = a.get("label", base)
        state[label] = {
            "base": base,
            "total": int(a.get("total", 1)),
            "completed": set(),  # terminal task JobIDs that COMPLETED
            "failed": set(),     # terminal task JobIDs that did not COMPLETE
            "running": 0,        # from most recent poll
            "done": False,       # array reached a terminal whole
        }

    summary = ", ".join(
        f"{lbl} ({s['base']}, {s['total']} task{'s' if s['total'] != 1 else ''})"
        for lbl, s in state.items()
    )
    print(f"\nWatching {len(state)} SLURM job(s): {summary}")
    print("Press Ctrl+C to stop watching (jobs keep running)\n")

    submission_time = time.time()
    start_time = None
    last_update = 0.0

    try:
        while not all(s["done"] for s in state.values()):
            for label, s in state.items():
                if s["done"]:
                    continue
                try:
                    cmd = ["sacct", "-j", s["base"], "--format=JobID,State", "-n", "-P"]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                except Exception:
                    continue
                if res.returncode != 0:
                    continue

                running = 0
                pending_present = False
                for line in res.stdout.strip().split("\n"):
                    if "|" not in line:
                        continue
                    jid, st = line.split("|", 1)
                    st = st.split("|")[0]
                    # Skip job substeps (.batch / .extern / .0).
                    if "." in jid:
                        continue
                    # Compressed pending-range row -> tasks queued but not started.
                    if "[" in jid:
                        pending_present = True
                        continue
                    if st == "RUNNING":
                        running += 1
                        if start_time is None:
                            start_time = time.time()
                            sys.stdout.write("\r" + " " * 200 + "\r")
                            print("⏱️  First task started running, timer started...\n")
                    elif st in _PENDING_STATES:
                        pending_present = True

                    if jid in s["completed"] or jid in s["failed"]:
                        continue
                    if _is_terminal_state(st):
                        sys.stdout.write("\r" + " " * 200 + "\r")
                        sys.stdout.flush()
                        if st == "COMPLETED":
                            s["completed"].add(jid)
                            if verbose and s["total"] <= 20:
                                done = len(s["completed"]) + len(s["failed"])
                                print(f"  ✓ {label} {jid} completed ({done}/{s['total']})")
                        else:
                            s["failed"].add(jid)
                            print(f"  ✗ {label} {jid} FAILED [{st}]")

                s["running"] = running
                done_n = len(s["completed"]) + len(s["failed"])
                # Done when we've accounted for every expected task, OR when
                # nothing is left pending/running and we've seen at least one
                # terminal task (guards against an over-counted `total`).
                if done_n >= s["total"] or (
                    done_n > 0 and running == 0 and not pending_present
                ):
                    s["done"] = True

            now = time.time()
            if now - last_update >= 5:
                parts = []
                for label, s in state.items():
                    done = len(s["completed"]) + len(s["failed"])
                    seg = f"{label} {done}/{s['total']}"
                    if s["failed"]:
                        seg += f" ✗{len(s['failed'])}"
                    if s["running"]:
                        seg += f" ⟳{s['running']}"
                    parts.append(seg)
                if start_time is None:
                    head = f"⏳ Queued {format_time(int(now - submission_time))}"
                else:
                    head = f"⏳ Runtime {format_time(int(now - start_time))}"
                sys.stdout.write(f"\r{(head + ' | ' + ' | '.join(parts))[:200]}")
                sys.stdout.flush()
                last_update = now

            time.sleep(poll_interval)

        sys.stdout.write("\r" + " " * 200 + "\r")
        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped watching. Jobs continue running.")
        for label, s in state.items():
            print(f"  {label}: status `squeue -j {s['base']}`  |  cancel `scancel {s['base']}`")
        out = {
            label: {
                "completed": len(s["completed"]),
                "failed": len(s["failed"]),
                "total": s["total"],
                "failed_ids": sorted(s["failed"]),
            }
            for label, s in state.items()
        }
        out["interrupted"] = True
        return out

    total_failed = sum(len(s["failed"]) for s in state.values())
    runtime = (time.time() - start_time) / 60 if start_time else 0
    print(f"\n{'='*60}")
    print(f"All watched jobs finished in {runtime:.1f} min runtime")
    for label, s in state.items():
        msg = f"  {label}: {len(s['completed'])}/{s['total']} completed"
        if s["failed"]:
            msg += f", {len(s['failed'])} failed"
        print(msg)
    print(f"{'='*60}\n")

    out = {
        label: {
            "completed": len(s["completed"]),
            "failed": len(s["failed"]),
            "total": s["total"],
            "failed_ids": sorted(s["failed"]),
        }
        for label, s in state.items()
    }
    out["all_completed"] = total_failed == 0
    return out


def _wait_for_jobs(
    submitted_jobs: list[dict],
    base_job_id: str,
    slurm_params: dict,
    experiment: str,
    verbose: bool = True,
    print_resource_summary: bool = True,
    print_success: bool = True,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Wait for all jobs to complete with live progress monitoring.

    Returns tuple of (completed_names, failed_jobs) where failed_jobs is a list of (name, job_id) tuples.
    """
    print(f"Waiting for all {len(submitted_jobs)} jobs to complete...")
    print(f"Press Ctrl+C to abort waiting (jobs will continue running)\n")

    submission_time = time.time()
    start_time = None  # Will be set when first job starts running
    completed = []  # List of completed job names
    completed_set = set()  # Set of job_ids for fast lookup (not names — names can collide)
    failed = []  # List of (name, job_id) tuples for output
    failed_set = set()  # Set of job_ids for fast lookup
    job_stats = {}  # Track job statistics
    last_update = 0
    update_interval = 5  # Update timer every 5 seconds

    # Get base job ID for batch queries
    first_job_id = str(submitted_jobs[0]["job"].job_id)
    base_job_id_for_query = first_job_id.split("_")[0] if "_" in first_job_id else first_job_id

    try:
        while len(completed) + len(failed) < len(submitted_jobs):
            # Batch query SLURM for all job states at once (much faster than individual queries)
            job_states = {}
            try:
                cmd = ["sacct", "-j", base_job_id_for_query, "--format=JobID,State", "-n", "-P"]
                sacct_result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if sacct_result.returncode == 0:
                    for line in sacct_result.stdout.strip().split("\n"):
                        if "|" in line:
                            parts = line.split("|")
                            jid = parts[0].split(".")[0]  # Remove .batch suffix
                            state = parts[1]
                            job_states[jid] = state
            except Exception:
                pass

            # Check if any jobs started running (for timer)
            if start_time is None:
                for jid, state in job_states.items():
                    if state == "RUNNING":
                        start_time = time.time()
                        print(f"⏱️  First job started running, timer started...\n")
                        break

            # Process completed/failed jobs
            for job_info in submitted_jobs:
                job = job_info["job"]
                name = job_info["name"]
                job_id = str(job.job_id)

                # Skip if already tracked
                if job_id in completed_set or job_id in failed_set:
                    continue

                # Check status from batch query
                status = job_states.get(job_id, "")
                if _is_terminal_state(status):
                    # Clear the live timer line before printing completion
                    sys.stdout.write("\r" + " " * 100 + "\r")
                    sys.stdout.flush()

                    elapsed = time.time() - start_time if start_time else 0

                    if status == "COMPLETED":
                        completed.append(name)
                        completed_set.add(job_id)

                        # Store job stats for later display
                        job_stats[name] = {
                            "job_id": job_id,
                            "status": "completed",
                            "elapsed": elapsed,
                        }

                        total_elapsed_job = time.time() - submission_time
                        time_str = f"[walltime: {format_time(int(total_elapsed_job))}, runtime: {format_time(int(elapsed))}]" if start_time else f"[walltime: {format_time(int(total_elapsed_job))}, queued]"
                        if print_success:
                            print(f"  ✓ {name} completed ({len(completed)}/{len(submitted_jobs)}) [job: {job_id}] {time_str}")
                    else:
                        failed.append((name, job_id))  # Store tuple with job_id
                        failed_set.add(job_id)

                        # Store failure info
                        job_stats[name] = {
                            "job_id": job_id,
                            "status": "failed",
                            "elapsed": elapsed,
                            "error": status,
                        }

                        total_elapsed_job = time.time() - submission_time
                        time_str = f"[walltime: {format_time(int(total_elapsed_job))}, runtime: {format_time(int(elapsed))}]" if start_time else f"[walltime: {format_time(int(total_elapsed_job))}, queued]"
                        # Suppress detailed error messages - just show job name and count
                        # Full error details are in SLURM logs: slurm_logs/<log_dir>/<job_id>_0_log.err
                        print(f"  ✗ {name} FAILED ({len(failed)} failed) [job: {job_id}] {time_str}")

            # Update live timer display
            current_time = time.time()
            if current_time - last_update >= update_interval:
                timeout_min = slurm_params.get("timeout_min", 20)
                _print_progress_update(
                    completed_count=len(completed),
                    failed_count=len(failed),
                    total_jobs=len(submitted_jobs),
                    start_time=start_time,
                    submission_time=submission_time,
                    timeout_min=timeout_min,
                )
                last_update = current_time

            # Sleep briefly if not all done yet
            if len(completed) + len(failed) < len(submitted_jobs):
                time.sleep(1)

        # Clear the timer line when all done
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()

        # Calculate total time (runtime if started, otherwise just queued time)
        if start_time:
            total_elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"All jobs finished in {total_elapsed/60:.1f} minutes runtime")
            print(f"  Completed: {len(completed)}/{len(submitted_jobs)}")
            print(f"  Failed: {len(failed)}/{len(submitted_jobs)}")
            print(f"{'='*60}\n")
        else:
            queue_time = time.time() - submission_time
            print(f"\n{'='*60}")
            print(f"All jobs finished (queued for {queue_time/60:.1f} minutes, but never started running)")
            print(f"  Completed: {len(completed)}/{len(submitted_jobs)}")
            print(f"  Failed: {len(failed)}/{len(submitted_jobs)}")
            print(f"{'='*60}\n")

        # Get SLURM utilization stats for all jobs
        if print_resource_summary:
            print()  # Blank line before resource summary
            _print_resource_summary(
                job_list=submitted_jobs,
                job_stats=job_stats,
                experiment=experiment,
                slurm_params=slurm_params,
                failed_jobs=failed,
            )

    except KeyboardInterrupt:
        print(f"\n\n⚠️  User interrupted waiting. Jobs will continue running.")
        print(f"  Check status: squeue -u $USER | grep {base_job_id}")
        print(f"  Cancel all: scancel {base_job_id}\n")

    return completed, failed


def handle_batch_mode_cli(
    detect_func: Callable,
    job_builder_func: Callable,
    args: Any,
    slurm_params: dict,
    log_dir: str,
    manifest_prefix: str,
    step_description: str = "processing",
) -> int:
    """
    Handle --all batch mode CLI logic with confirmation prompts and job submission.

    This consolidates the common batch mode pattern used across slurm submission scripts.

    Parameters
    ----------
    detect_func : Callable
        Function that returns (experiments_to_process, experiments_completed)
    job_builder_func : Callable
        Function that builds job dict for a given experiment
        Signature: (experiment: str, args: Any) -> dict
    args : argparse.Namespace
        Parsed CLI arguments with: force, quiet, dry_run, yes, no_wait
    slurm_params : dict
        SLURM parameters dict
    log_dir : str
        Log directory path pattern (e.g., "slurm_tracking_logs/%j")
    manifest_prefix : str
        Prefix for manifest files (e.g., "tracking_batch")
    step_description : str
        Description for output messages (e.g., "tracking", "conversion")

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    # Detect experiments
    experiments_to_process, experiments_completed = detect_func(
        force=args.force,
        verbose=not args.quiet,
    )

    if not experiments_to_process:
        print(f"\n✓ All experiments are complete! No {step_description} jobs needed.\n")
        if not args.quiet and experiments_completed:
            print(f"Completed experiments ({len(experiments_completed)}):")
            for exp, n_done, n_total, _ in experiments_completed:
                print(f"  ✓ {exp}")
        return 0

    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch {step_description.title()} Submission: {len(experiments_to_process)} experiments")
    print(f"{'='*60}\n")

    for exp, n_done, n_total, _ in experiments_to_process:
        status = f"{n_done}/{n_total}" if n_total > 0 else "pending"
        print(f"  • {exp}: {status}")

    if experiments_completed and not args.quiet:
        print(f"\nAlready completed ({len(experiments_completed)}):")
        for exp, n_done, n_total, _ in experiments_completed:
            print(f"  ✓ {exp}")

    # Exit early for dry run (before building expensive job lists)
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN: Job Submission Plan")
        print(f"{'='*60}\n")
        print(f"Would submit jobs for {len(experiments_to_process)} experiments")
        print(f"SLURM Resources (per job):")
        print(f"  Timeout: {slurm_params['timeout_min']} min")
        print(f"  Memory: {slurm_params.get('mem', slurm_params.get('slurm_mem', 'N/A'))}")
        print(f"  CPUs: {slurm_params['cpus_per_task']}")
        print(f"  Partition: {slurm_params['slurm_partition']}")
        if slurm_params.get('gpus_per_node'):
            print(f"  GPUs: {slurm_params['gpus_per_node']}")
        if slurm_params.get('slurm_gres'):
            print(f"  GPUs: {slurm_params['slurm_gres']}")
        if slurm_params.get('slurm_constraint'):
            print(f"  Constraint: {slurm_params['slurm_constraint']}")
        print(f"\nDRY RUN: No jobs submitted\n")
        return 0

    # Build job list (only if not dry run)
    all_jobs = []
    for experiment, n_done, n_total, _ in experiments_to_process:
        jobs = job_builder_func(experiment, args)
        if jobs:
            # Support job builders that return either a single dict or a list of dicts
            if isinstance(jobs, dict):
                all_jobs.append(jobs)
            elif isinstance(jobs, list):
                all_jobs.extend(jobs)

    # Show job plan
    print(f"\n{'='*60}")
    print(f"Job Submission Plan")
    print(f"{'='*60}\n")
    print(f"Total jobs to submit: {len(all_jobs)}")
    print(f"SLURM Resources (per job):")
    print(f"  Timeout: {slurm_params['timeout_min']} min")
    print(f"  Memory: {slurm_params.get('mem', slurm_params.get('slurm_mem', 'N/A'))}")
    print(f"  CPUs: {slurm_params['cpus_per_task']}")
    print(f"  Partition: {slurm_params['slurm_partition']}")
    if slurm_params.get('gpus_per_node'):
        print(f"  GPUs: {slurm_params['gpus_per_node']}")
    if slurm_params.get('slurm_gres'):
        print(f"  GPUs: {slurm_params['slurm_gres']}")
    if slurm_params.get('slurm_constraint'):
        print(f"  Constraint: {slurm_params['slurm_constraint']}")

    print(f"\nExperiments:")
    for job in all_jobs:
        print(f"  {job['metadata']['experiment']}")

    print(f"\n{'='*60}\n")

    # Prompt for confirmation
    if not args.yes:
        try:
            response = input(f"Submit {len(all_jobs)} jobs to SLURM? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\nCancelled by user. No jobs submitted.\n")
                return 0
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled by user. No jobs submitted.\n")
            return 0
        print()
    else:
        print("Proceeding with submission (--yes flag provided)...\n")

    # Submit jobs
    result = submit_parallel_jobs(
        jobs_to_submit=all_jobs,
        experiment=f"batch_{manifest_prefix}_{len(experiments_to_process)}_experiments",
        slurm_params=slurm_params,
        log_dir=log_dir,
        manifest_prefix=manifest_prefix,
        dry_run=args.dry_run,
        wait_for_completion=not args.no_wait,
        verbose=not args.quiet,
        post_completion_callback=None,
    )

    # Return exit code
    if result.get("dry_run"):
        return 0
    elif result.get("success"):
        if result.get("all_completed") is not None:
            return 0 if result.get("all_completed") else 1
        else:
            return 0
    else:
        return 1


def handle_single_experiment_cli(
    submit_func: Callable,
    args: Any,
    slurm_params: dict,
) -> int:
    """
    Handle single experiment mode CLI logic with experiment name resolution.

    Parameters
    ----------
    submit_func : Callable
        Function to submit the job
        Signature: (experiment: str, slurm_params: dict, args: Any) -> dict
    args : argparse.Namespace
        Parsed CLI arguments with: experiment, dry_run, no_wait, quiet
    slurm_params : dict
        SLURM parameters dict

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    from cyclops_utils.data.filesystem import resolve_experiment_name

    # Resolve experiment name
    resolved_name = resolve_experiment_name(
        args.experiment,
        allow_interactive=True
    )

    if resolved_name is None:
        print("No experiment selected or found. Exiting.")
        return 1

    # Submit job
    result = submit_func(
        experiment=resolved_name,
        slurm_params=slurm_params,
        args=args,
    )

    # Return exit code
    if result.get("dry_run"):
        return 0
    elif result.get("success"):
        if result.get("all_completed") is not None:
            return 0 if result.get("all_completed") else 1
        else:
            return 0
    else:
        return 1
