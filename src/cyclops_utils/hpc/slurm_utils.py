import os
import subprocess
import datetime
import re


def print_slurm_job_header(title: str = "SLURM Job", extra_info: dict = None) -> None:
    """Print SLURM job environment info header at job startup.

    Args:
        title: Title to display in header
        extra_info: Optional dict of additional key-value pairs to display
    """
    job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    job_name = os.environ.get("SLURM_JOB_NAME", "N/A")
    partition = os.environ.get("SLURM_JOB_PARTITION", "N/A")
    n_nodes = os.environ.get("SLURM_JOB_NUM_NODES", "N/A")
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "N/A")
    n_tasks = os.environ.get("SLURM_NPROCS", os.environ.get("SLURM_NTASKS", "N/A"))
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU", None)
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE", None)
    time_limit = os.environ.get("SLURM_TIMELIMIT", "N/A")
    gpus = os.environ.get("SLURM_GPUS", os.environ.get("SLURM_JOB_GPUS", None))

    # Format memory string
    if mem_per_cpu:
        mem_str = f"{mem_per_cpu}/CPU"
    elif mem_per_node:
        mem_str = f"{mem_per_node}/node"
    else:
        mem_str = "N/A"

    print("=" * 60)
    print(f"{title} - Job {job_id}")
    print("=" * 60)
    print(f"  Job Name:    {job_name}")
    print(f"  Partition:   {partition}")
    print(f"  Nodes:       {n_nodes} ({nodelist})")
    print(f"  Tasks:       {n_tasks}")
    print(f"  CPUs/task:   {cpus_per_task}")
    print(f"  Memory:      {mem_str}")
    print(f"  Time limit:  {time_limit}")
    if gpus:
        print(f"  GPUs:        {gpus}")

    # Print extra info if provided
    if extra_info:
        for key, value in extra_info.items():
            print(f"  {key}:".ljust(14) + f" {value}")

    print("=" * 60, flush=True)


def parse_time_to_seconds(time_str: str) -> float:
    """Parse time string (e.g., '01:23:45' or '1-02:34:56') to seconds."""
    if not time_str or time_str == "N/A":
        return 0

    try:
        days = 0
        if "-" in time_str:
            parts = time_str.split("-")
            days = int(parts[0])
            time_str = parts[1]

        time_parts = time_str.split(":")
        if len(time_parts) == 3:
            hours, minutes, seconds = map(float, time_parts)
            return days * 86400 + hours * 3600 + minutes * 60 + seconds
        elif len(time_parts) == 2:
            minutes, seconds = map(float, time_parts)
            return days * 86400 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0
    return 0


def get_gpu_allocation_info(job_id: str) -> tuple[str, str]:
    """Query SLURM for GPU allocation details by checking job and node info.

    Returns:
        tuple: (gpu_count, gpu_type) where gpu_type is GPU type (e.g., "H100")
    """
    gpu_count = None
    gpu_type = None

    try:
        # Query job information
        cmd = ["scontrol", "show", "job", str(job_id)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            job_output = result.stdout

            # Parse GPU count from AllocTRES (e.g., "AllocTRES=...gres/gpu=2")
            alloc_tres_match = re.search(r"AllocTRES=.*gres/gpu=(\d+)", job_output)
            if alloc_tres_match:
                gpu_count = alloc_tres_match.group(1)

            # Get node name to query GPU type
            node_match = re.search(r"NodeList=(\S+)", job_output)
            if node_match:
                node_name = node_match.group(1)

                # Query node for GPU type
                node_cmd = ["scontrol", "show", "node", node_name]
                node_result = subprocess.run(
                    node_cmd, capture_output=True, text=True, timeout=5
                )
                if node_result.returncode == 0:
                    # Look for Gres=gpu:TYPE:COUNT in node info (e.g., "Gres=gpu:h100:8")
                    node_gpu_match = re.search(
                        r"Gres=gpu:([a-z0-9]+):", node_result.stdout, re.IGNORECASE
                    )
                    if node_gpu_match:
                        gpu_type = node_gpu_match.group(1).upper()

    except Exception:
        pass

    return gpu_count, gpu_type


def parse_gpu_metrics_from_tres(tres_usage: str) -> tuple[float | None, float | None]:
    """Parse GPU utilization and memory from TRES usage string.

    Args:
        tres_usage: TRES string like "cpu=...,gres/gpuutil=108,gres/gpumem=72378M,..."

    Returns:
        tuple: (gpu_util_pct, gpu_mem_gb) - utilization percentage and memory in GB
    """
    gpu_util_pct = None
    gpu_mem_gb = None

    if not tres_usage:
        return gpu_util_pct, gpu_mem_gb

    for part in tres_usage.split(","):
        if "gres/gpuutil=" in part:
            try:
                val = part.split("=")[-1]
                if val and val.replace(".", "").isdigit():
                    gpu_util_pct = float(val)
            except (ValueError, IndexError):
                pass
        elif "gres/gpumem=" in part:
            try:
                val = part.split("=")[-1]
                # Convert to GB (format is typically like "72378M")
                mem_bytes = parse_memory_value(val)
                if mem_bytes > 0:
                    gpu_mem_gb = mem_bytes / (1024**3)
            except (ValueError, IndexError):
                pass

    return gpu_util_pct, gpu_mem_gb

def parse_memory_value(mem_str: str) -> float:
    """Parse memory string (e.g., '123456K', '123M', '12G') to bytes."""
    if not mem_str or mem_str == "N/A":
        return 0

    mem_str = mem_str.strip()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

    for suffix, multiplier in multipliers.items():
        if mem_str.endswith(suffix):
            try:
                return float(mem_str[:-1]) * multiplier
            except ValueError:
                return 0

    # Try to parse as plain number (bytes)
    try:
        return float(mem_str)
    except ValueError:
        return 0



def format_time(seconds: int) -> str:
    """Format seconds as human-readable time (e.g., '2h:10m:30s' or '15m:45s').

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h:{minutes:02d}m:{secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m:{secs:02d}s"
    else:
        return f"{secs}s"
        
def format_size_gb(value_str: str) -> str:
    """Format size value to GB (e.g., '1.5GB', '234.0GB')."""
    if not value_str or value_str == "N/A" or value_str.strip() == "":
        return "-"

    # Parse to bytes first
    bytes_val = parse_memory_value(value_str)
    if bytes_val == 0:
        return "-"

    # Always show in GB
    gb_val = bytes_val / (1024**3)
    return f"{gb_val:.1f}GB"

def print_slurm_job_stats(
    job_id: str,
    experiment: str = None,
    slurm_params: dict = None,
    queue_time_sec: int = 0,
) -> None:
    """Query sacct and display resource utilization stats for the completed job.

    Args:
        job_id: Final job ID that completed successfully
        experiment: Experiment name
        slurm_params: Slurm parameters used
        queue_time_sec: Queue time for initial job (0 if not captured)
    """
    try:
        import time as time_module

        # Query sacct for key resource metrics
        fields = [
            "JobID",
            "JobName",
            "Partition",
            "NodeList",
            "Submit",
            "Start",
            "Elapsed",
            "Timelimit",
            "AllocCPUS",
            "MaxRSS",
            "AveRSS",
            "MaxVMSize",
            "MaxDiskRead",
            "MaxDiskWrite",
            "AveDiskRead",
            "AveDiskWrite",
            "TotalCPU",
            "AveCPU",
            "CPUTime",
            "ReqMem",
            "State",
            "ExitCode",
            "AllocTRES",
            "TRESUsageInMax",
            "TRESUsageInAve",
            "TRESUsageOutMax",
        ]

        # Wait briefly for Slurm accounting to update, then query with retries
        time_module.sleep(2)

        for attempt in range(3):
            cmd = [
                "sacct",
                "-j",
                str(job_id),
                "--format",
                ",".join(fields),
                "--parsable2",
                "--noheader",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return

            lines = result.stdout.strip().split("\n")
            if not lines or not lines[0]:
                return

            # Parse all records (main job + job steps)
            records = []
            for line in lines:
                parts = line.split("|")
                if len(parts) >= len(fields):
                    records.append(dict(zip(fields, parts)))

            if not records:
                return

            # Use main job record for metadata, but merge in actual job step stats
            # Job steps: .batch = batch script, .extern = setup, .0/.1/etc = srun tasks
            # For GPU jobs, the actual work is typically in .0 (srun step)
            data = records[0]
            step_data = None

            # Priority: look for .0 step (srun), then .batch, for actual resource usage
            for record in records:
                job_id_field = record.get("JobID", "")
                if ".0" in job_id_field:
                    step_data = record
                    break

            if not step_data:
                for record in records:
                    job_id_field = record.get("JobID", "")
                    if ".batch" in job_id_field:
                        step_data = record
                        break

            # Merge resource usage fields from the appropriate step
            if step_data:
                resource_keys = [
                    "MaxRSS",
                    "AveRSS",
                    "MaxVMSize",
                    "MaxDiskRead",
                    "MaxDiskWrite",
                    "AveDiskRead",
                    "AveDiskWrite",
                    "TotalCPU",
                    "AveCPU",
                    "TRESUsageInMax",
                    "TRESUsageInAve",
                    "TRESUsageOutMax",
                ]
                for key in resource_keys:
                    if step_data.get(key, "").strip():
                        data[key] = step_data[key]

            # Check if job is completed and has usage stats
            state = data.get("State", "")
            has_stats = data.get("MaxRSS", "").strip() != ""

            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT") and has_stats:
                break

            # If still running or stats not ready, wait and retry
            if attempt < 2:
                time_module.sleep(3)

        # Continue with whatever data we have (show partial stats if still running)

        # Get GPU allocation info (count and type from node)
        gpu_count, gpu_type = get_gpu_allocation_info(job_id)

        # Fallback: Parse TRES if helper didn't find GPU count
        if not gpu_count and "AllocTRES" in data and data["AllocTRES"]:
            tres_parts = data["AllocTRES"].split(",")
            for part in tres_parts:
                if "gres/gpu=" in part:
                    gpu_count = part.split("=")[-1]
                    break

        # Parse GPU metrics from TRESUsageInMax and TRESUsageInAve
        gpu_util_max_pct, gpu_mem_max_gb = parse_gpu_metrics_from_tres(
            data.get("TRESUsageInMax", "")
        )
        gpu_util_ave_pct, gpu_mem_ave_gb = parse_gpu_metrics_from_tres(
            data.get("TRESUsageInAve", "")
        )

        # Calculate percentages where possible
        elapsed_sec = parse_time_to_seconds(data.get("Elapsed", ""))
        timelimit_sec = parse_time_to_seconds(data.get("Timelimit", ""))
        time_usage_pct = (elapsed_sec / timelimit_sec * 100) if timelimit_sec > 0 else 0

        # CPU efficiency: TotalCPU (actual CPU time used) / CPUTime (allocated CPU time)
        cpu_time_used_sec = parse_time_to_seconds(data.get("TotalCPU", ""))
        cpu_time_avail_sec = parse_time_to_seconds(data.get("CPUTime", ""))
        cpu_usage_pct = (
            (cpu_time_used_sec / cpu_time_avail_sec * 100)
            if cpu_time_avail_sec > 0
            else 0
        )

        # Memory metrics
        max_rss_bytes = parse_memory_value(data.get("MaxRSS", ""))
        ave_rss_bytes = parse_memory_value(data.get("AveRSS", ""))
        req_mem_str = data.get("ReqMem", "")
        # ReqMem format can be like "400Gn" or "400Gc" - strip trailing letter
        if req_mem_str and req_mem_str[-1].isalpha() and len(req_mem_str) > 1:
            if req_mem_str[-2].isalpha():
                req_mem_str = req_mem_str[:-1]
        req_mem_bytes = parse_memory_value(req_mem_str)
        mem_usage_pct = (
            (max_rss_bytes / req_mem_bytes * 100) if req_mem_bytes > 0 else 0
        )

        # Helper to format values (empty string -> "-")
        def fmt(val):
            return val if val and str(val).strip() else "-"

        # Calculate queue wait time
        submit_time = fmt(data.get("Submit"))
        start_time = fmt(data.get("Start"))
        queue_wait = "-"
        if submit_time != "-" and start_time != "-":
            try:
                submit_dt = datetime.strptime(submit_time, "%Y-%m-%dT%H:%M:%S")
                start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                wait_seconds = (start_dt - submit_dt).total_seconds()
                queue_wait = f"{int(wait_seconds // 60)}m {int(wait_seconds % 60)}s"
            except:
                pass

        # Display the stats table with three columns
        print("\n┌─────────────────────────┬──────────────────────────┬──────────────┐")
        print("│ Slurm Resource Usage    │ Value                    │ Utilization  │")
        print("├─────────────────────────┼──────────────────────────┼──────────────┤")
        if experiment:
            print(f"│ Experiment              │ {experiment:<24} │              │")
        print(f"│ Job ID                  │ {str(job_id):<24} │              │")
        print(
            f"│ Job Name                │ {fmt(data.get('JobName')):<24} │              │"
        )
        print(
            f"│ Partition               │ {fmt(data.get('Partition')):<24} │              │"
        )
        print(
            f"│ Node                    │ {fmt(data.get('NodeList')):<24} │              │"
        )

        # Display queue time (from timer if captured, otherwise from sacct)
        if queue_time_sec > 0:
            queue_hours = queue_time_sec // 3600
            queue_mins = (queue_time_sec % 3600) // 60
            queue_secs = queue_time_sec % 60
            if queue_hours > 0:
                queue_display = f"{queue_hours}h:{queue_mins:02d}m:{queue_secs:02d}s"
            elif queue_mins > 0:
                queue_display = f"{queue_mins}m:{queue_secs:02d}s"
            else:
                queue_display = f"{queue_secs}s"
            print(f"│ Queue Wait Time         │ {queue_display:<24} │              │")
        else:
            queue_display = queue_wait
            print(f"│ Queue Wait Time         │ {queue_display:<24} │              │")

        # Format CPU efficiency (max)
        cpu_efficiency_max = f"{cpu_usage_pct:.1f}%" if cpu_usage_pct > 0 else "-"

        # Format memory displays in GB
        def format_bytes_gb(bytes_val):
            if not bytes_val or bytes_val <= 0:
                return "-"
            return f"{bytes_val / (1024**3):.1f}GB"

        max_rss_display = format_bytes_gb(max_rss_bytes)
        ave_rss_display = format_bytes_gb(ave_rss_bytes)

        # Format disk I/O in GB
        max_disk_read = format_size_gb(data.get("MaxDiskRead", ""))
        ave_disk_read = format_size_gb(data.get("AveDiskRead", ""))
        max_disk_write = format_size_gb(data.get("MaxDiskWrite", ""))
        ave_disk_write = format_size_gb(data.get("AveDiskWrite", ""))

        stats = [
            (
                "Elapsed Time",
                fmt(data.get("Elapsed")),
                f"{time_usage_pct:.1f}%" if time_usage_pct > 0 else None,
            ),
            ("Time Limit", fmt(data.get("Timelimit")), None),
            ("Allocated CPUs", fmt(data.get("AllocCPUS")), None),
            ("CPU Time Used (Max)", fmt(data.get("TotalCPU")), cpu_efficiency_max),
            ("CPU Time Available", fmt(data.get("CPUTime")), None),
            (
                "Max RSS Memory",
                max_rss_display,
                f"{mem_usage_pct:.1f}%" if mem_usage_pct > 0 else None,
            ),
            ("Avg RSS Memory", ave_rss_display, None),
            ("Memory Requested", fmt(data.get("ReqMem")), None),
            ("Max Disk Read", max_disk_read, None),
            ("Avg Disk Read", ave_disk_read, None),
            ("Max Disk Write", max_disk_write, None),
            ("Avg Disk Write", ave_disk_write, None),
        ]

        if gpu_count:
            gpu_count_int = (
                int(gpu_count) if gpu_count and str(gpu_count).isdigit() else 1
            )

            stats.append(("GPUs Allocated", gpu_count, None))
            if gpu_type:
                stats.append(("GPU Type", gpu_type, None))

            # GPU Memory (show max and avg)
            if gpu_mem_max_gb is not None:
                stats.append(("GPU Memory Max", f"{gpu_mem_max_gb:.1f}GB", None))
            if gpu_mem_ave_gb is not None:
                stats.append(("GPU Memory Avg", f"{gpu_mem_ave_gb:.1f}GB", None))

            # GPU Utilization (show max and avg, with per-GPU average for multi-GPU)
            if gpu_util_max_pct is not None:
                per_gpu_max = (
                    gpu_util_max_pct / gpu_count_int
                    if gpu_count_int > 1
                    else gpu_util_max_pct
                )
                util_display = f"{gpu_util_max_pct:.1f}%"
                util_note = f"{per_gpu_max:.1f}%/gpu" if gpu_count_int > 1 else None
                stats.append(("GPU SM Util Max", util_display, util_note))

            if gpu_util_ave_pct is not None:
                per_gpu_ave = (
                    gpu_util_ave_pct / gpu_count_int
                    if gpu_count_int > 1
                    else gpu_util_ave_pct
                )
                util_display = f"{gpu_util_ave_pct:.1f}%"
                util_note = f"{per_gpu_ave:.1f}%/gpu" if gpu_count_int > 1 else None
                stats.append(("GPU SM Util Avg", util_display, util_note))

        # Add state and exit code
        exit_code = fmt(data.get("ExitCode"))
        state_display = fmt(data.get("State"))
        if exit_code != "-" and exit_code != "0:0":
            state_display = f"{state_display} ({exit_code})"
        stats.append(("Job State", state_display, None))

        for label, value, pct in stats:
            pct_display = pct if pct else ""
            print(f"│ {label:<23} │ {value:<24} │ {pct_display:<12} │")

        print("└─────────────────────────┴──────────────────────────┴──────────────┘")

    except Exception as e:
        # Fail-soft but log the issue for debugging
        print(f"\n⚠  Warning: Could not retrieve complete Slurm job statistics: {e}")
        import traceback

        traceback.print_exc()


def check_jobs_complete(job_ids: list[str]) -> dict[str, bool | str]:
    """Check if SLURM jobs are complete.

    Args:
        job_ids: List of SLURM job IDs to check

    Returns:
        dict with keys:
            - "all_complete": bool, True if all jobs are in terminal state
            - "states": dict[str, str], job_id -> state mapping
            - "failed_jobs": list[str], job IDs that failed
            - "running": int, count of running jobs
            - "pending": int, count of pending jobs
            - "completed": int, count of successfully completed jobs
    """
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
    failed_states = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}

    if not job_ids:
        return {
            "all_complete": True,
            "states": {},
            "failed_jobs": [],
            "running": 0,
            "pending": 0,
            "completed": 0,
        }

    # Query sacct for job status
    cmd = ["sacct", "-j", ",".join(str(j) for j in job_ids), "-o", "JobID,State", "-n", "-P"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"sacct failed with code {result.returncode}")

        states = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            if "|" in line:
                job_id, state = line.split("|", 1)
                job_id = job_id.strip()
                state = state.strip()
                # Skip job steps (with "."), but keep array jobs (with "_")
                if "." not in job_id and job_id:
                    states[job_id] = state

        # If sacct didn't return statuses for some jobs (common for running jobs),
        # fill missing ones using squeue so we can detect RUNNING/PENDING accurately.
        # Use -r to expand array jobs into individual task lines.
        missing = [str(j) for j in job_ids if str(j) not in states or not states.get(str(j))]
        if missing:
            sq_cmd = ["squeue", "-r", "-j", ",".join(missing), "-h", "-o", "%i %T"]
            sq_res = subprocess.run(sq_cmd, capture_output=True, text=True, timeout=10)
            if sq_res.returncode == 0:
                for line in sq_res.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            states[parts[0]] = parts[1]

    except Exception:
        # If sacct fails, fall back to squeue (only shows running/pending jobs).
        # Use -r to expand array jobs into individual task lines.
        cmd = ["squeue", "-r", "-j", ",".join(str(j) for j in job_ids), "-h", "-o", "%i %T"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        states = {}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    states[parts[0]] = parts[1]

        # Jobs visible in squeue are running/pending; jobs NOT in squeue
        # may have completed or may not have appeared yet. Treat as PENDING
        # so the monitor keeps polling until sacct confirms the terminal state.
        for jid in job_ids:
            if str(jid) not in states:
                states[str(jid)] = "PENDING"

    # Jobs not found in either sacct or squeue are likely too new for the
    # accounting DB. Treat as PENDING so we keep polling; sacct will report
    # the real terminal state once it catches up.
    for jid in job_ids:
        if str(jid) not in states:
            states[str(jid)] = "PENDING"

    # Compute counts
    running = sum(1 for s in states.values() if s == "RUNNING")
    pending = sum(1 for s in states.values() if s == "PENDING")
    completed = sum(1 for s in states.values() if s == "COMPLETED")
    failed_jobs = [jid for jid, s in states.items() if s in failed_states]
    all_complete = all(states.get(str(jid)) in terminal_states for jid in job_ids)

    return {
        "all_complete": all_complete,
        "states": states,
        "failed_jobs": failed_jobs,
        "running": running,
        "pending": pending,
        "completed": completed,
    }


def monitor_jobs_with_retry(
    job_ids: list[str],
    job_specs: list[dict],
    experiment: str,
    slurm_params: dict,
    submit_fn,
    poll_interval: int = 10,
    max_retries: int = 1,
    verbose: bool = True,
    phase_name: str = "jobs",
) -> dict:
    """
    Monitor SLURM jobs until completion, retrying failed jobs up to max_retries times.

    This function monitors a list of jobs and automatically retries any that fail.
    If jobs still fail after all retries, raises RuntimeError to prevent proceeding
    with incomplete results.

    Parameters
    ----------
    job_ids : list[str]
        List of SLURM job IDs to monitor
    job_specs : list[dict]
        List of job specifications (same order as job_ids). Each spec should contain
        the information needed by submit_fn to resubmit the job (e.g., name, func, kwargs).
    experiment : str
        Experiment name for logging
    slurm_params : dict
        SLURM parameters for retry submissions
    submit_fn : callable
        Function to submit jobs. Should accept (jobs_to_submit, experiment, slurm_params, ...)
        and return dict with 'base_job_id' and 'jobs' keys.
    poll_interval : int
        Seconds between status checks (default: 10)
    max_retries : int
        Maximum retry attempts for failed jobs (default: 1)
    verbose : bool
        Print progress updates
    phase_name : str
        Name for this phase in log messages (default: "jobs")

    Returns
    -------
    dict
        - "final_job_ids": list[str] - Final job IDs (may differ from input if retried)
        - "all_succeeded": bool - True if all jobs completed successfully
        - "failed_jobs": list[str] - Job IDs that failed (empty if all succeeded)
        - "retry_count": int - Total number of retries performed

    Raises
    ------
    RuntimeError
        If any jobs fail after all retry attempts are exhausted
    """
    import sys
    import time
    from datetime import timedelta

    if not job_ids:
        return {
            "final_job_ids": [],
            "all_succeeded": True,
            "failed_jobs": [],
            "retry_count": 0,
        }

    # If running in DAG context, delegate to PhaseTracker for live progress.
    # Note: retry logic is not available through PhaseTracker — if jobs fail,
    # the step fails and the user can re-run via the DAG.
    try:
        from cyclops_utils.hpc.phase_tracker import _current_phase_tracker
        tracker = _current_phase_tracker.get(None)
        if tracker:
            result = tracker.wait_for_job_ids(
                job_ids, job_specs, label=phase_name,
            )
            if not result["all_succeeded"]:
                failed = result["failed_jobs"]
                raise RuntimeError(
                    f"{len(failed)}/{len(job_ids)} {phase_name} failed: "
                    + ", ".join(failed[:5])
                )
            return result
    except (ImportError, LookupError):
        pass

    # Current job tracking - maps current job_id to original index
    current_job_ids = list(job_ids)
    job_id_to_spec_idx = {jid: i for i, jid in enumerate(job_ids)}
    retry_counts = {jid: 0 for jid in job_ids}
    total_retries = 0

    submission_time = time.time()
    first_job_start_time = None

    if verbose:
        print(f"Monitoring {len(current_job_ids)} {phase_name} (polling every {poll_interval}s)...\n")

    while True:
        time.sleep(poll_interval)

        status = check_jobs_complete(current_job_ids)
        running = status["running"]
        completed = status["completed"]
        failed_job_ids = status["failed_jobs"]
        failed_count = len(failed_job_ids)

        # Detect first job start
        if first_job_start_time is None and running > 0:
            first_job_start_time = time.time()
            wait_time = int(first_job_start_time - submission_time)
            if verbose:
                print(f"⏱️  First job started running after {wait_time}s in queue\n")

        # Print progress
        if verbose:
            n_done = completed + failed_count
            total = len(current_job_ids)
            pct = (n_done / total) * 100 if total > 0 else 0

            if first_job_start_time is None:
                queue_time = int(time.time() - submission_time)
                timer_str = f"⏳ Progress: {n_done}/{total} ({pct:.0f}%) | Queued for {format_time(queue_time)}"
            else:
                elapsed = int(time.time() - first_job_start_time)
                timer_str = f"⏳ Progress: {n_done}/{total} ({pct:.0f}%) | Runtime: {format_time(elapsed)}"

            sys.stdout.write(f"\r{timer_str}" + " " * 20)
            sys.stdout.flush()

        # Check if all jobs are in terminal state
        if status["all_complete"]:
            # Clear progress line
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()

            if not failed_job_ids:
                # All succeeded
                if verbose:
                    elapsed = timedelta(seconds=int(time.time() - submission_time))
                    print(f"✓ All {len(current_job_ids)} {phase_name} completed successfully [{elapsed}]")
                return {
                    "final_job_ids": current_job_ids,
                    "all_succeeded": True,
                    "failed_jobs": [],
                    "retry_count": total_retries,
                }

            # Some jobs failed - check if we can retry
            if verbose:
                print(f"⚠️  {len(failed_job_ids)} {phase_name} failed:")
                for jid in failed_job_ids:
                    state = status['states'].get(jid, 'UNKNOWN')
                    attempts = retry_counts.get(jid, 0) + 1
                    print(f"    Job {jid}: {state} (attempt {attempts}/{max_retries + 1})")

            # Identify jobs that can be retried vs exhausted
            jobs_to_retry = []
            jobs_exhausted = []

            for jid in failed_job_ids:
                if retry_counts.get(jid, 0) >= max_retries:
                    jobs_exhausted.append(jid)
                else:
                    jobs_to_retry.append(jid)

            # If any jobs exhausted retries, fail
            if jobs_exhausted:
                exhausted_details = []
                for jid in jobs_exhausted:
                    spec_idx = job_id_to_spec_idx.get(jid)
                    spec_name = job_specs[spec_idx]["name"] if spec_idx is not None else "unknown"
                    state = status['states'].get(jid, 'UNKNOWN')
                    exhausted_details.append(f"Job {jid} ({spec_name}): {state}")

                raise RuntimeError(
                    f"{phase_name.capitalize()} failed after {max_retries + 1} attempt(s). "
                    f"The following jobs could not be completed:\n  " +
                    "\n  ".join(exhausted_details) +
                    f"\n\nCannot proceed - all {phase_name} must succeed."
                )

            # Retry failed jobs
            if jobs_to_retry:
                if verbose:
                    print(f"\n🔄 Retrying {len(jobs_to_retry)} failed job(s)...")

                # Collect specs for retry
                retry_specs = []
                retry_original_indices = []

                for jid in jobs_to_retry:
                    spec_idx = job_id_to_spec_idx.get(jid)
                    if spec_idx is not None:
                        retry_specs.append(job_specs[spec_idx])
                        retry_original_indices.append(spec_idx)
                        retry_counts[jid] = retry_counts.get(jid, 0) + 1
                        total_retries += 1
                        if verbose:
                            print(f"    Resubmitting: {job_specs[spec_idx]['name']}")

                # Submit retry jobs
                retry_result = submit_fn(
                    jobs_to_submit=retry_specs,
                    experiment=experiment,
                    slurm_params=slurm_params,
                    log_dir="slurm_logs/retry_%j",
                    manifest_prefix="retry",
                    dry_run=False,
                    wait_for_completion=False,
                    verbose=False,
                )

                if retry_result.get("base_job_id") and retry_result.get("jobs"):
                    base_id = retry_result["base_job_id"]
                    num_jobs = len(retry_result["jobs"])

                    if num_jobs == 1:
                        new_job_ids = [str(base_id)]
                    else:
                        new_job_ids = [f"{base_id}_{i}" for i in range(num_jobs)]

                    if verbose:
                        print(f"  → Submitted retry: {base_id} ({num_jobs} jobs)\n")

                    # Update tracking
                    for i, old_jid in enumerate(jobs_to_retry):
                        new_jid = new_job_ids[i]
                        spec_idx = job_id_to_spec_idx.pop(old_jid)
                        job_id_to_spec_idx[new_jid] = spec_idx

                        # Transfer retry count to new job ID
                        old_count = retry_counts.pop(old_jid, 0)
                        retry_counts[new_jid] = old_count

                        # Update current_job_ids
                        idx_in_current = current_job_ids.index(old_jid)
                        current_job_ids[idx_in_current] = new_jid

                    # Reset timing for retry monitoring
                    submission_time = time.time()
                    first_job_start_time = None

                    if verbose:
                        print(f"Monitoring retry jobs...\n")


def print_job_submission_table(
    job_id: str,
    experiment: str,
    step_name: str,
    slurm_params: dict,
    log_path: str = None,
) -> None:
    """
    Display a formatted table of Slurm job submission parameters.

    Args:
        job_id: The SLURM job ID
        experiment: Experiment name
        step_name: Name of the step being submitted
        slurm_params: Dictionary of SLURM parameters with keys like:
            - timeout_min: Timeout in minutes
            - mem_gb or mem: Memory allocation
            - cpus_per_task: Number of CPUs
            - gpus_per_node or gpus: Number of GPUs
            - partition or slurm_partition: SLURM partition
            - constraint or slurm_constraint: Node constraint
        log_path: Optional path to log file(s)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n--- Submitting step '{step_name}' to Slurm ---")
    print("┌─────────────┬──────────────────────────┐")
    print("│ Parameter   │ Value                    │")
    print("├─────────────┼──────────────────────────┤")
    print(f"│ Experiment  │ {experiment:<24} │")
    print(f"│ Step        │ {step_name:<24} │")
    print(f"│ Job ID      │ {str(job_id):<24} │")
    print(f"│ Submitted   │ {timestamp:<24} │")

    # Map various param names to display labels
    params_to_display = [
        (["timeout_min"], "Timeout", lambda v: f"{v} min"),
        (["mem_gb"], "Memory", lambda v: f"{v}G"),
        (["mem"], "Memory", lambda v: str(v)),
        (["cpus_per_task", "cpus"], "CPUs", lambda v: str(v)),
        (["gpus_per_node", "gpus"], "GPUs", lambda v: str(v)),
        (["partition", "slurm_partition"], "Partition", lambda v: str(v)),
        (["constraint", "slurm_constraint"], "Constraint", lambda v: str(v)),
    ]

    displayed_keys = set()
    for keys, label, formatter in params_to_display:
        for key in keys:
            if key in slurm_params and key not in displayed_keys:
                value = formatter(slurm_params[key])
                # Truncate long values
                if len(value) > 24:
                    value = value[:21] + "..."
                print(f"│ {label:<11} │ {value:<24} │")
                displayed_keys.add(key)
                break

    print("└─────────────┴──────────────────────────┘")

    if log_path:
        print(f"  Logs: {log_path}")
