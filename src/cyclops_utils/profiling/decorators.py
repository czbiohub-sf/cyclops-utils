import datetime
import itertools
import logging
import time
from functools import wraps
import threading
from contextlib import suppress
from datetime import datetime
import os
import shutil
import subprocess
import inspect  # allows dynamic args to be passed to slack notifier
from joblib import Parallel, delayed
import click
import yaml
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union

import numpy as np

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.resource_manager import _get_available_ram_gb

# NOTE: slack_notifier is imported lazily inside notify_step's wrapper, not here.
# This module is pulled in pipeline-wide for @versioned_function, and importing
# slack_notifier at module scope made every one of those consumers load slack_sdk,
# read the shared dotenv token file, and run the notifier's import-time
# auth_test() -- a live network call -- even when nothing used @notify_step.


ATTACHMENT_RESOLVERS: list[dict] = []  # populated at decoration time; each entry: {name, module, step_message, resolver, func}


def notify_step(step_message: str, success_message: str = None, attachments=None):
    """
    Decorator to send Slack notifications for a pipeline step.
    Messages can be formatted with arguments from the decorated function.
    Example: @notify_step("Running for {process}")

    `attachments` (optional) uploads files into the experiment thread after the
    success notification. Accepts:
      - a list/tuple of paths (str or Path), evaluated as-is, or
      - a callable `(result, bound_arguments) -> Iterable[path]` that derives
        paths from the function's return value and bound args. The callable
        may also yield (path, caption) tuples to label individual attachments.
    Missing paths are skipped silently; per-file upload failures are logged
    but never raise (attachments are best-effort, not pipeline-critical).
    """

    def decorator(func):
        if attachments is not None:
            ATTACHMENT_RESOLVERS.append({
                "name": getattr(func, "__name__", str(func)),
                "module": getattr(func, "__module__", ""),
                "step_message": step_message,
                "resolver": attachments,
                "func": func,
            })

        @wraps(func)
        def wrapper(*args, **kwargs):
            from cyclops_utils.profiling.slack_notifier import (
                get_active_notifier,
                is_orchestrator_notifying,
            )

            notifier = get_active_notifier()
            # When PipelineRunner is handling the step/success/error messages
            # itself, suppress the decorator's own messages — but still run
            # the function and still process attachments so artifacts land in
            # the thread regardless of who sends the surrounding messages.
            send_messages = notifier is not None and not is_orchestrator_notifying()

            # Bind args/kwargs to the function signature to access them by name
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Create a "safe" dictionary for formatting by converting all args to strings
            safe_args = {k: str(v) for k, v in bound_args.arguments.items()}

            # --- Smartly find the experiment name ---
            experiment_name = None
            if "experiment" in bound_args.arguments:
                experiment_name = bound_args.arguments["experiment"]
            elif args and hasattr(args[0], "dataset") and hasattr(args[0].dataset, "experiment"):
                experiment_name = args[0].dataset.experiment
            elif args and isinstance(args[0], str):
                experiment_name = args[0]
            # ---

            # Format the initial step message
            final_step_message = step_message.format(**safe_args)
            if experiment_name:
                final_step_message += f" for '{experiment_name}'"

            if send_messages:
                notifier.step(final_step_message)

            try:
                result = func(*args, **kwargs)

                # Format the success message
                final_success_message = (
                    success_message
                    if success_message
                    else f"Finished: {step_message.format(**safe_args)}"
                )
                if success_message:
                    final_success_message = success_message.format(**safe_args)
                if experiment_name:
                    final_success_message += f" for '{experiment_name}'"

                if send_messages:
                    notifier.success(final_success_message)
                # Attachments run whenever a notifier exists — orchestrator
                # mode included — so the artifacts land in the same thread
                # the orchestrator is posting step/success messages into.
                if notifier:
                    _attach_files(notifier, attachments, result, bound_args)

                return result

            except Exception as e:
                # Format a failure message
                failure_message = f"Failed: {step_message.format(**safe_args)}"
                if experiment_name:
                    failure_message += f" for '{experiment_name}'"

                if send_messages:
                    # Pass the exception to the notifier to include traceback
                    notifier.error(failure_message, e)

                # Re-raise the exception to not swallow it
                raise

        return wrapper

    return decorator


def _attach_files(notifier, attachments, result, bound_args):
    """Upload `attachments` into the notifier's thread. Best-effort: missing
    files are skipped, upload errors are logged but never raise.

    Resolver return entries may be:
      - a path (str/Path)              -> single upload
      - a (path, caption) tuple        -> single upload with caption
      - a dict {"paths": [...], "caption": "..."} -> one batched message
        containing every existing path, with a shared caption. Inner items
        in "paths" follow the same path/(path,title) shape.
    """
    if attachments is None:
        print("[notify_step.attach] no attachments resolver — skipping")
        return
    if notifier is None:
        print("[notify_step.attach] no active notifier — skipping")
        return
    try:
        items = attachments(result, bound_args) if callable(attachments) else attachments
    except Exception as e:
        print(f"Warning: notify_step attachments resolver raised: {e}")
        return
    if items is None:
        print("[notify_step.attach] resolver returned None — skipping")
        return
    items = list(items)
    print(f"[notify_step.attach] uploading {len(items)} item(s) to thread_ts={getattr(notifier, 'thread_ts', None)}")
    for entry in items:
        if isinstance(entry, dict) and "paths" in entry:
            batch = []
            for inner in entry["paths"]:
                if isinstance(inner, tuple):
                    p_in, t_in = inner[0], (inner[1] if len(inner) > 1 else None)
                else:
                    p_in, t_in = inner, None
                try:
                    pp = Path(p_in)
                except TypeError:
                    continue
                if pp.exists():
                    batch.append((str(pp), t_in))
            if not batch:
                continue
            try:
                notifier.attach_batch(batch, caption=entry.get("caption"))
            except Exception as e:
                print(f"Warning: notify_step failed to batch-attach {len(batch)} file(s): {e}")
            continue
        caption = None
        if isinstance(entry, tuple):
            path, caption = entry[0], (entry[1] if len(entry) > 1 else None)
        else:
            path = entry
        try:
            p = Path(path)
        except TypeError:
            continue
        if not p.exists():
            continue
        try:
            notifier.attach(str(p), caption=caption)
        except Exception as e:
            print(f"Warning: notify_step failed to attach {p}: {e}")


def versioned_function(version):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- Smartly find the experiment name ---
            experiment = kwargs.get("experiment")
            if not experiment and args:
                # Check if first arg is an object with a dataset attribute (like a class instance)
                if hasattr(args[0], "dataset") and hasattr(
                    args[0].dataset, "experiment"
                ):
                    experiment = args[0].dataset.experiment
                # Otherwise, assume it's a string
                else:
                    experiment = args[0]
            # ---

            if experiment is None or not isinstance(experiment, str):
                # If experiment cannot be determined or is not a string, run the original function without logging.
                return func(*args, **kwargs)

            # --- Create a unique key for the function call ---
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Align step key generation with PipelineRunner/report status: use bare function name
            func_name = func.__name__
            key_parts = [func_name]

            if "process" in bound_args.arguments:
                key_parts.append(str(bound_args.arguments["process"]))
            # Include method (e.g., 'mine' vs 'probabilistic') for finer-grained auditing
            if "method" in bound_args.arguments:
                key_parts.append(str(bound_args.arguments["method"]))
            if "well" in bound_args.arguments:
                key_parts.append(str(bound_args.arguments["well"]).replace("/", "_"))
            # Disambiguate parallel-dispatched subtasks. Parent steps that fan
            # out via submit_parallel_jobs hand each child the same
            # func/process/well but different position ranges, FOV ids, or
            # task ids; if we don't fold those into the log_key, all children
            # write to the same key and clobber each other's GPU/CPU metrics.
            # Range form (reconstruct_tilt_corrected, reconstruct, ...):
            for arg, label in (
                ("position_start", "ps"),
                ("position_end", "pe"),
                ("array_task_id", "at"),
                ("task_id", "t"),
            ):
                v = bound_args.arguments.get(arg)
                if v is not None:
                    key_parts.append(f"{label}{v}")
            # Single-FOV form (cell_seg.segment_single_position, ...):
            pos = bound_args.arguments.get("position")
            if isinstance(pos, str):
                key_parts.append("pos" + pos.replace("/", "_"))

            dataset = OpsDataset(experiment)
            log_key = "_".join(key_parts)
            log_file_path = dataset.logfile
            log_entry = {
                log_key: {
                    "version": version,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "git_commit": _get_git_commit_hash(),
                }
            }

            _load_and_write_log(log_key, log_file_path, log_entry)
            log_storage_dir = dataset.experiment_path / "logs"
            log_storage_dir.mkdir(exist_ok=True)

            # Take a timestamped snapshot of the aggregate yaml. Under heavy
            # concurrency (e.g. 150-task fan-out pyramid builds), the source
            # file can be atomic-renamed by another worker mid-copy, leaving
            # our open fd pointing at a now-deleted NFS inode → OSError 116
            # "Stale file handle". The snapshot is auxiliary (the actual entry
            # is already in function_call_log.yaml from _load_and_write_log
            # above), so swallow that failure instead of bringing down the
            # whole task. Retry once on stale-fd with a fresh open.
            retained_log_file_path =  log_storage_dir / f"function_call_log-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml"
            for _attempt in range(2):
                try:
                    shutil.copy(log_file_path, retained_log_file_path)
                    break
                except OSError as _e:
                    if _attempt == 0 and getattr(_e, "errno", None) == 116:
                        continue  # stale handle — retry once with a fresh open
                    print(
                        f"Warning: failed to snapshot {log_file_path} → "
                        f"{retained_log_file_path}: {_e}"
                    )
                    break


            # --- Lightweight peak memory sampler (CPU + GPU) ---
            sampler_running = True
            max_cpu_rss_bytes = 0
            max_cpu_mem_percent = 0.0
            max_gpu_used_bytes = 0
            max_gpu_mem_percent = 0.0
            max_proc_cpu_percent = 0.0
            max_num_threads = 0
            logical_cpu_count = None
            per_core_peak_pct = None  # track peak utilization per logical CPU core
            proc_obj = None  # persistent psutil.Process to get meaningful CPU%
            # Per-process tracker for the whole subprocess tree. cpu_percent()
            # returns "% CPU since last call on THIS object", so we have to
            # keep a primed Process instance per child PID — first call returns
            # 0, subsequent calls return real values. We refresh the children
            # set every iteration so newly spawned workers (e.g. ProcessPool
            # under parallel_mode='shard_stripes') get sampled correctly.
            primed_procs: dict = {}  # pid -> psutil.Process
            # Running averages: accumulate and divide at end. Separate counters
            # for CPU vs GPU because nvml init may fail or nvml may not be
            # available, so the GPU sample count may be lower than CPU's.
            n_cpu_samples = 0
            sum_cpu_rss_bytes = 0
            sum_cpu_mem_percent = 0.0
            sum_proc_cpu_percent = 0.0
            n_gpu_samples = 0
            sum_gpu_sm_util_pct = 0.0
            sum_gpu_mem_ctrl_util_pct = 0.0
            sum_gpu_mem_used_mib = 0.0
            sum_gpu_power_draw_watts = 0.0
            # NVML state and metrics (best-effort)
            nvml_inited = False
            nvml_gpu_count = 0
            nvml_gpu_models: list[str] | None = None
            nvml_gpu_total_mib: list[float] | None = None
            nvml_gpu_power_limit_watts: list[float | None] | None = None
            nvml_gpu_util_gpu_max_pct = 0.0  # device SM util (max across devices)
            nvml_gpu_util_mem_max_pct = 0.0  # device mem controller util
            nvml_gpu_power_draw_watts_max = 0.0
            nvml_gpu_mem_used_mib_max = 0.0
            # nvml_proc_sm_util_max_pct = 0.0  # per-process SM util if available
            # nvml_proc_mem_util_max_pct = 0.0  # per-process mem util if available

            def _sample_memory_loop():
                nonlocal max_cpu_rss_bytes, max_cpu_mem_percent, max_gpu_used_bytes, max_gpu_mem_percent, max_proc_cpu_percent, max_num_threads, per_core_peak_pct, proc_obj, sampler_running, primed_procs
                nonlocal n_cpu_samples, sum_cpu_rss_bytes, sum_cpu_mem_percent, sum_proc_cpu_percent
                nonlocal n_gpu_samples, sum_gpu_sm_util_pct, sum_gpu_mem_ctrl_util_pct, sum_gpu_mem_used_mib, sum_gpu_power_draw_watts
                nonlocal nvml_inited, nvml_gpu_count, nvml_gpu_models, nvml_gpu_total_mib, nvml_gpu_power_limit_watts
                nonlocal nvml_gpu_util_gpu_max_pct, nvml_gpu_util_mem_max_pct, nvml_gpu_power_draw_watts_max
                # nonlocal nvml_proc_sm_util_max_pct, nvml_proc_mem_util_max_pct
                while sampler_running:
                    # CPU RSS — sample across the whole subprocess tree so
                    # multiprocess workloads (parallel_mode='shard_stripes',
                    # ProcessPoolExecutor, etc.) report aggregate utilization
                    # rather than just the parent dispatcher process.
                    try:
                        import psutil  # type: ignore

                        if proc_obj is None:
                            proc_obj = psutil.Process()
                            _ = proc_obj.cpu_percent(interval=None)  # prime
                            primed_procs[proc_obj.pid] = proc_obj

                        # Refresh children set; prime each new child once so
                        # subsequent cpu_percent() calls return real values.
                        try:
                            for child in proc_obj.children(recursive=True):
                                if child.pid not in primed_procs:
                                    try:
                                        child.cpu_percent(interval=None)
                                        primed_procs[child.pid] = child
                                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                                        pass
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            pass

                        # Aggregate across parent + tracked children. Drop
                        # exited PIDs from the primed set.
                        agg_rss = 0
                        agg_mem_pct = 0.0
                        agg_cpu_pct = 0.0
                        agg_threads = 0
                        dead_pids: list[int] = []
                        for pid, p in primed_procs.items():
                            try:
                                with p.oneshot():
                                    agg_rss += p.memory_info().rss
                                    agg_mem_pct += p.memory_percent()
                                    agg_cpu_pct += p.cpu_percent(interval=None)
                                    agg_threads += p.num_threads()
                            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                                dead_pids.append(pid)
                        for pid in dead_pids:
                            primed_procs.pop(pid, None)

                        rss = agg_rss
                        pct = agg_mem_pct
                        cpu_pct = agg_cpu_pct
                        num_threads = agg_threads
                        if rss > max_cpu_rss_bytes:
                            max_cpu_rss_bytes = rss
                        if pct and pct > max_cpu_mem_percent:
                            max_cpu_mem_percent = float(pct)
                        if cpu_pct and cpu_pct > max_proc_cpu_percent:
                            max_proc_cpu_percent = float(cpu_pct)
                        if num_threads and num_threads > max_num_threads:
                            max_num_threads = int(num_threads)
                        # Running averages — increment after every successful sample.
                        n_cpu_samples += 1
                        sum_cpu_rss_bytes += int(rss)
                        sum_cpu_mem_percent += float(pct or 0.0)
                        sum_proc_cpu_percent += float(cpu_pct or 0.0)
                        # Update per-core peak utilization (system-wide instantaneous per-core)
                        try:
                            core_list = psutil.cpu_percent(interval=None, percpu=True)
                            if core_list:
                                if per_core_peak_pct is None:
                                    per_core_peak_pct = [0.0 for _ in core_list]
                                for i, v in enumerate(core_list):
                                    if (
                                        i < len(per_core_peak_pct)
                                        and v > per_core_peak_pct[i]
                                    ):
                                        per_core_peak_pct[i] = float(v)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # GPU per-process usage (prefer task-specific metrics)
                    # Torch: per-process allocated/reserved memory
                    torch_used_total = 0
                    torch_pct_max = 0.0
                    try:
                        import torch  # type: ignore

                        if torch.cuda.is_available():
                            num_dev_t = torch.cuda.device_count()
                            for i in range(num_dev_t):
                                try:
                                    alloc = torch.cuda.memory_allocated(i)
                                    reserved = torch.cuda.memory_reserved(i)
                                    used = max(alloc, reserved)
                                except Exception:
                                    used = 0
                                try:
                                    total = torch.cuda.get_device_properties(
                                        i
                                    ).total_memory
                                except Exception:
                                    total = 0
                                torch_used_total += used
                                if total:
                                    pct = (used / total) * 100.0
                                    if pct > torch_pct_max:
                                        torch_pct_max = float(pct)
                    except Exception:
                        pass

                    # CuPy: per-process memory pools (device and pinned)
                    cupy_used_total = 0
                    cupy_pct_max = 0.0
                    try:
                        import cupy  # type: ignore

                        try:
                            num_dev_c = int(cupy.cuda.runtime.getDeviceCount())
                        except Exception:
                            num_dev_c = 0
                        for dev_id in range(num_dev_c):
                            try:
                                dev = cupy.cuda.Device(dev_id)
                                with dev:
                                    pool_used = (
                                        cupy.get_default_memory_pool().used_bytes()
                                    )
                                    pinned_used = (
                                        cupy.get_default_pinned_memory_pool().used_bytes()
                                    )
                                    used = (pool_used or 0) + (pinned_used or 0)
                                    free_b, total_b = cupy.cuda.runtime.memGetInfo()
                                    cupy_used_total += used
                                    if total_b:
                                        pct = (used / total_b) * 100.0
                                        if pct > cupy_pct_max:
                                            cupy_pct_max = float(pct)
                            except Exception:
                                continue
                    except Exception:
                        pass

                    # Combine torch and cupy readings conservatively (avoid double counting): take max
                    used_candidate = max(torch_used_total, cupy_used_total)
                    pct_candidate = max(torch_pct_max, cupy_pct_max)
                    if used_candidate > max_gpu_used_bytes:
                        max_gpu_used_bytes = used_candidate
                    if pct_candidate > max_gpu_mem_percent:
                        max_gpu_mem_percent = pct_candidate
                    # NVML: device utilization and power (optional)
                    try:
                        import pynvml  # type: ignore

                        if not nvml_inited:
                            pynvml.nvmlInit()
                            nvml_inited = True
                            nvml_gpu_count = int(pynvml.nvmlDeviceGetCount())
                            nvml_gpu_models = []
                            nvml_gpu_total_mib = []
                            nvml_gpu_power_limit_watts = []
                            for i in range(nvml_gpu_count):
                                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                                try:
                                    name = pynvml.nvmlDeviceGetName(h)
                                    name = (
                                        name.decode("utf-8")
                                        if isinstance(name, bytes)
                                        else str(name)
                                    )
                                except Exception:
                                    name = f"GPU-{i}"
                                try:
                                    mem_total_mib = pynvml.nvmlDeviceGetMemoryInfo(
                                        h
                                    ).total / (1024 * 1024)
                                except Exception:
                                    mem_total_mib = 0
                                try:
                                    p_lim = (
                                        pynvml.nvmlDeviceGetEnforcedPowerLimit(h)
                                        / 1000.0
                                    )
                                except Exception:
                                    try:
                                        p_lim = (
                                            pynvml.nvmlDeviceGetPowerManagementLimit(h)
                                            / 1000.0
                                        )
                                    except Exception:
                                        p_lim = 0.0
                                nvml_gpu_models.append(name)
                                nvml_gpu_total_mib.append(
                                    round(float(mem_total_mib), 1)
                                )
                                nvml_gpu_power_limit_watts.append(
                                    round(float(p_lim), 1) if p_lim else None
                                )
                        if nvml_inited:
                            # Per-tick aggregates across all GPUs (max for util,
                            # sum for memory + power) so the running average is
                            # consistent with the max metrics already reported.
                            tick_sm_max = 0.0
                            tick_mem_ctrl_max = 0.0
                            tick_mem_used_mib_sum = 0.0
                            tick_power_watts_sum = 0.0
                            tick_gpu_seen = False
                            for i in range(int(pynvml.nvmlDeviceGetCount())):
                                try:
                                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                                    nvml_gpu_util_gpu_max_pct = max(
                                        nvml_gpu_util_gpu_max_pct, float(util.gpu)
                                    )
                                    nvml_gpu_util_mem_max_pct = max(
                                        nvml_gpu_util_mem_max_pct, float(util.memory)
                                    )
                                    tick_sm_max = max(tick_sm_max, float(util.gpu))
                                    tick_mem_ctrl_max = max(tick_mem_ctrl_max, float(util.memory))
                                    tick_gpu_seen = True
                                    try:
                                        p_watts = (
                                            pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                                        )
                                        if (
                                            p_watts
                                            and p_watts > nvml_gpu_power_draw_watts_max
                                        ):
                                            nvml_gpu_power_draw_watts_max = float(
                                                p_watts
                                            )
                                        if p_watts:
                                            tick_power_watts_sum += float(p_watts)
                                    except Exception:
                                        pass
                                    # Track device-level used memory (MiB)
                                    try:
                                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                                        used_mib = mem.used / (1024 * 1024)
                                        if used_mib > nvml_gpu_mem_used_mib_max:
                                            nvml_gpu_mem_used_mib_max = float(used_mib)
                                        tick_mem_used_mib_sum += float(used_mib)
                                    except Exception:
                                        pass
                                except Exception:
                                    continue
                            # Accumulate per-tick aggregates for running averages
                            if tick_gpu_seen:
                                n_gpu_samples += 1
                                sum_gpu_sm_util_pct += tick_sm_max
                                sum_gpu_mem_ctrl_util_pct += tick_mem_ctrl_max
                                sum_gpu_mem_used_mib += tick_mem_used_mib_sum
                                sum_gpu_power_draw_watts += tick_power_watts_sum
                    except Exception:
                        pass
                    time.sleep(1.0)

            # Prime CPU percent measurement and cache CPU count if available
            with suppress(Exception):
                import psutil  # type: ignore

                proc_obj = psutil.Process()
                _ = proc_obj.cpu_percent(interval=None)
                logical_cpu_count = psutil.cpu_count(logical=True)
                # Prime per-core readings and init peaks
                core_list = psutil.cpu_percent(interval=None, percpu=True)
                if core_list:
                    per_core_peak_pct = [0.0 for _ in core_list]

            mem_thread = threading.Thread(target=_sample_memory_loop, daemon=True)
            mem_thread.start()

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                sampler_running = False
                with suppress(Exception):
                    mem_thread.join(timeout=2.0)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Convert bytes to MiB and MB with one decimal
            def _to_mib(b: int | float | None):
                if not b:
                    return None
                return round(float(b) / (1024 * 1024), 1)

            def _to_mb(b: int | float | None):
                if not b:
                    return None
                return round(float(b) / 1_000_000, 1)

            def _round_pct(p: float | None):
                if p is None or p == 0:
                    return None
                return round(float(p), 1)

            def _round_pct_zero_ok(p: float | None):
                if p is None:
                    return None
                return round(float(p), 1)

            # Count how many cores exceeded a utilization threshold at any time
            active_cores_gt10pct = None
            try:
                if per_core_peak_pct is not None:
                    active_cores_gt10pct = int(
                        sum(1 for v in per_core_peak_pct if v >= 10.0)
                    )
            except Exception:
                active_cores_gt10pct = None

            # Final NVML refresh for device memory used to avoid nulls if sampling missed peaks
            with suppress(Exception):
                import pynvml  # type: ignore

                if nvml_inited:
                    for i in range(int(pynvml.nvmlDeviceGetCount())):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        used_mib = float(mem.used) / (1024 * 1024)
                        if used_mib > nvml_gpu_mem_used_mib_max:
                            nvml_gpu_mem_used_mib_max = used_mib

            # Base (CPU + timing) fields
            # Determine total CPU RAM available to this task (MB), preferring SLURM/cgroup-aware value
            cpu_total_mem_mb_value = None
            with suppress(Exception):
                ram_gb, _src = _get_available_ram_gb()
                if ram_gb:
                    cpu_total_mem_mb_value = round(float(ram_gb) * 1024.0, 1)
            if cpu_total_mem_mb_value is None:
                with suppress(Exception):
                    import psutil  # type: ignore

                    cpu_total_mem_mb_value = round(
                        psutil.virtual_memory().total / (1024.0 * 1024.0), 1
                    )

            # Compute normalized process CPU utilization relative to cores utilized
            cpu_util_norm_pct = None
            try:
                if max_proc_cpu_percent:
                    denom_cores = (
                        int(active_cores_gt10pct)
                        if active_cores_gt10pct is not None and active_cores_gt10pct > 0
                        else (int(logical_cpu_count) if logical_cpu_count else None)
                    )
                    if denom_cores and denom_cores > 0:
                        cpu_util_norm_pct = round(
                            float(max_proc_cpu_percent) / float(denom_cores), 1
                        )
            except Exception:
                cpu_util_norm_pct = None

            # SLURM job identification (None when not running under SLURM)
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_job_id and slurm_array_task_id:
                slurm_id = f"{slurm_job_id}_{slurm_array_task_id}"
            else:
                slurm_id = slurm_job_id  # None when not on SLURM

            # Hostname identifies the SLURM node — critical for correlating
            # slow steps with specific node hardware in Grafana dashboards.
            try:
                import socket as _socket
                hostname = _socket.gethostname()
            except Exception:
                hostname = None

            # Averaged CPU metrics from the running totals collected by the
            # sampling loop. Falls back to None if no samples landed (e.g.
            # very short steps that exit before the 1Hz sampler ticks).
            cpu_mem_avg_mb = (
                _to_mb(int(sum_cpu_rss_bytes / n_cpu_samples))
                if n_cpu_samples else None
            )
            cpu_mem_percent_avg = (
                _round_pct(sum_cpu_mem_percent / n_cpu_samples)
                if n_cpu_samples else None
            )
            cpu_util_process_avg_pct_raw = (
                _round_pct_zero_ok(sum_proc_cpu_percent / n_cpu_samples)
                if n_cpu_samples else None
            )

            entry_fields = {
                "Version": version,
                "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Ran in": f"{elapsed_time:.2f} seconds",
                "Git_commit": _get_git_commit_hash(),
                "slurm_job_id": slurm_id,
                "hostname": hostname,
                # CPU
                "cpu_mem_max_mb": _to_mb(max_cpu_rss_bytes),
                "cpu_mem_avg_mb": cpu_mem_avg_mb,
                "cpu_mem_total_mb": cpu_total_mem_mb_value,
                "cpu_mem_percent": _round_pct(max_cpu_mem_percent),
                "cpu_mem_percent_avg": cpu_mem_percent_avg,
                "cpu_util_process_max_pct": _round_pct_zero_ok(max_proc_cpu_percent),
                "cpu_util_process_avg_pct": cpu_util_process_avg_pct_raw,
                "cpu_util_process_average_percent": cpu_util_norm_pct,
                "cpu_num_threads_max": (
                    int(max_num_threads) if max_num_threads else None
                ),
                "cpu_cores_count": (
                    int(logical_cpu_count) if logical_cpu_count else None
                ),
                # `node_cores_hot_count` is the COUNT OF CORES on the host
                # whose individual peak utilisation crossed 10% during this
                # step's wall window. It's NOT this step's own efficiency:
                # on a shared host (e.g. a login node), other users'
                # activity counts. Always interpret alongside
                # `hostname` and `cpu_util_process_avg_pct` — if the
                # process itself shows ~0%, this number is noise from
                # neighbours, not work done by this step.
                "node_cores_hot_count": (
                    int(active_cores_gt10pct)
                    if active_cores_gt10pct is not None
                    else None
                ),
            }

            # Conditionally include GPU metrics only if GPU was actually used or detected
            gpu_seen_via_libs = bool(max_gpu_used_bytes and max_gpu_used_bytes > 0)
            gpu_seen_via_nvml = bool(
                (nvml_inited and nvml_gpu_count)
                and (
                    nvml_gpu_util_gpu_max_pct > 0
                    or nvml_gpu_util_mem_max_pct > 0
                    or nvml_gpu_power_draw_watts_max > 0
                )
            )
            if gpu_seen_via_libs or gpu_seen_via_nvml:
                # entry_fields.update(
                #     {
                #         "gpu_process_vram_mb": _to_mb(max_gpu_used_bytes),
                #         "gpu_process_vram_pct": _round_pct(max_gpu_mem_percent),
                #     }
                # )
                # NVML extras if available
                if nvml_inited and nvml_gpu_count:
                    gpu_device_mem_used_mib = round(float(nvml_gpu_mem_used_mib_max), 1)
                    watts_used = round(nvml_gpu_power_draw_watts_max, 1)
                    # Running averages over the run (per-tick max-across-GPUs
                    # for util, sum-across-GPUs for memory + power, then avg
                    # over time). Falls back to None if the sampler never
                    # observed a GPU (e.g. job too short or NVML unreachable).
                    gpu_sm_avg_pct = (
                        _round_pct_zero_ok(sum_gpu_sm_util_pct / n_gpu_samples)
                        if n_gpu_samples else None
                    )
                    gpu_mem_ctrl_avg_pct = (
                        _round_pct_zero_ok(sum_gpu_mem_ctrl_util_pct / n_gpu_samples)
                        if n_gpu_samples else None
                    )
                    gpu_mem_used_mib_avg = (
                        round(sum_gpu_mem_used_mib / n_gpu_samples, 1)
                        if n_gpu_samples else None
                    )
                    gpu_power_watts_avg = (
                        round(sum_gpu_power_draw_watts / n_gpu_samples, 1)
                        if n_gpu_samples else None
                    )

                    entry_fields.update(
                        {
                            "gpu_count": int(nvml_gpu_count),
                            "gpu_models": nvml_gpu_models,
                            "gpu_device_mem_used_mib": gpu_device_mem_used_mib,
                            "gpu_device_mem_used_mib_avg": gpu_mem_used_mib_avg,
                            "gpu_device_mem_total_mib": nvml_gpu_total_mib,
                            "gpu_power_watts_limit": nvml_gpu_power_limit_watts,
                            "gpu_power_watts_draw_max": watts_used,
                            "gpu_power_watts_draw_avg": gpu_power_watts_avg,
                            "gpu_device_sm_util_max_pct": _round_pct_zero_ok(
                                nvml_gpu_util_gpu_max_pct
                            ),
                            "gpu_device_sm_util_avg_pct": gpu_sm_avg_pct,
                            "gpu_device_mem_ctrl_util_max_pct": _round_pct_zero_ok(
                                nvml_gpu_util_mem_max_pct
                            ),
                            "gpu_device_mem_ctrl_util_avg_pct": gpu_mem_ctrl_avg_pct,
                        }
                    )
                    # If single-GPU, compute percentages relative to totals/limits
                    try:
                        if int(nvml_gpu_count) == 1:
                            # total mem percent
                            total_list = nvml_gpu_total_mib
                            total_mib = (
                                float(total_list[0])
                                if isinstance(total_list, list) and total_list
                                else None
                            )
                            if total_mib:
                                entry_fields["gpu_device_mem_used_percent"] = round(
                                    (gpu_device_mem_used_mib / total_mib) * 100.0, 1
                                )
                            # power percent
                            limits = nvml_gpu_power_limit_watts
                            limit_watts = (
                                float(limits[0])
                                if isinstance(limits, list) and limits and limits[0]
                                else None
                            )
                            if limit_watts:
                                entry_fields["gpu_power_watts_percent"] = round(
                                    (watts_used / limit_watts) * 100.0, 1
                                )
                    except Exception:
                        pass

            # Squash singleton lists to scalars for cleaner YAML (e.g., gpu_models: A40 instead of list)
            try:
                for _k, _v in list(entry_fields.items()):
                    if isinstance(_v, list) and len(_v) == 1:
                        entry_fields[_k] = _v[0]
            except Exception:
                pass

            log_entry = {log_key: entry_fields}

            _load_and_write_log(log_key, log_file_path, log_entry)

            # Phase 3: also mirror this entry into the central per-step
            # slurm_stats.yaml when running in operational mode. Keeps the
            # aggregate yaml above as a rollback fallback. Sidesteps the
            # OPS_INPUT_BASE_DIR overlay symlink case where the aggregate
            # path is read-only.
            try:
                from cyclops_utils.ops_mode import mirror_step_stats
                mirror_step_stats(experiment, func_name, log_key, entry_fields)
            except Exception as _e:
                # Never let logging failures crash the actual work.
                print(f"Warning: failed to mirror step stats to central log: {_e}")

            return result

        wrapper._version = version
        return wrapper

    return decorator


def _acquire_nfs_lock(lock_path, timeout=120):
    """NFS-safe exclusive lock via O_CREAT|O_EXCL.

    ``fcntl.flock`` requires rpc.lockd, which is often not configured on
    HPC NFS mounts (silent failure — the lock LOOKS acquired but doesn't
    actually block other processes). Observed on ops0171: 30 concurrent
    tilt-recon workers writing decorator metrics → only ~2 of 30 landed
    because most writers raced and clobbered each other's aggregate
    updates.

    O_CREAT|O_EXCL is atomic on all POSIX FSes including NFS: exactly
    one process's open() succeeds; the others get FileExistsError and
    retry. Timeout guards against stale locks from crashed writers.
    """
    import time
    import random
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout:
                # Assume stale (crashed writer never cleaned up). Force take.
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                continue
            # Randomized backoff — avoids herds of workers all retrying in lockstep.
            time.sleep(0.02 + random.random() * 0.08)


def _release_nfs_lock(lock_path):
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def _load_and_write_log(log_key, log_file_path, log_entry):
    """Atomic, race-tolerant write of one entry into the aggregate YAML.

    Hold an NFS-safe lock for the whole load/modify/save cycle, then
    commit via tmp-file + rename so any concurrent reader sees either
    the OLD whole file or the NEW whole file (never a partial mid-write).
    On transient read failures (stale NFS fd, torn read during a
    concurrent rename), retry the read a few times; if all retries fail,
    skip this one write rather than clobber the file. Logging failures
    must never abort actual work — but they also must not delete other
    workers' entries.
    """
    import tempfile

    log_file_path = str(log_file_path)
    lock_path = log_file_path + ".lock"
    try:
        _acquire_nfs_lock(lock_path)
        try:
            existing = {}
            read_err = None
            if os.path.exists(log_file_path):
                # Retry the read a few times before giving up. Under heavy
                # concurrency (150+ baseline workers on ops0171), safe_load
                # can transiently fail because another writer's atomic
                # rename leaves our open fd pointing at a stale NFS inode
                # (OSError 116) or returns a torn read. The file itself
                # ends up parseable moments later. The PREVIOUS behaviour
                # treated the first parse error as "file is corrupt", moved
                # it aside, and started fresh — dropping every prior
                # entry. Observed on ops0171: 110 false-positive quarantines
                # in a single pheno baseline run, shrinking the aggregate
                # from ~500 entries down to ~30.
                for _read_attempt in range(3):
                    try:
                        with open(log_file_path, "r") as f:
                            loaded = yaml.safe_load(f)
                        if loaded is None:
                            existing = {}
                        elif isinstance(loaded, dict):
                            existing = loaded
                        else:
                            raise yaml.YAMLError(
                                f"expected top-level dict, got "
                                f"{type(loaded).__name__}"
                            )
                        read_err = None
                        break
                    except Exception as _re:
                        read_err = _re
                        time.sleep(0.05 * (1 + _read_attempt))
                if read_err is not None:
                    # Retries exhausted. Do NOT clobber the file — abort
                    # this one write with a warning. Losing one log entry
                    # is far cheaper than losing every prior entry. The
                    # per-step retention snapshots in logs/ still preserve
                    # the full history if the aggregate ever really is
                    # corrupt.
                    print(
                        f"Warning: could not read function_call_log after "
                        f"3 attempts ({log_file_path}: {read_err}); "
                        f"skipping this write. FCL left untouched."
                    )
                    return

            existing[log_key] = log_entry[log_key]

            # Atomic write: write to tmp on the same directory, fsync,
            # then rename over the target. The rename is atomic on
            # POSIX same-FS, so any reader sees one or the other but
            # never a partial file.
            tmp_dir = os.path.dirname(log_file_path) or "."
            fd, tmp_path = tempfile.mkstemp(
                prefix=".fcl.", suffix=".tmp", dir=tmp_dir,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    yaml.dump(existing, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(tmp_path, log_file_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            _release_nfs_lock(lock_path)
    except Exception as e:
        # Never let logging failures crash the actual work
        print(f"Warning: failed to write function call log: {e}")

    return


def aggregate_subtask_metrics(log_file_path, parent_log_key: str) -> dict:
    """Read all subtask entries whose key begins with ``parent_log_key`` and
    return a per-step summary dict (averages, peaks, hostnames, GPU models).

    Parent steps that fan out via ``submit_parallel_jobs`` get one
    function_call_log entry per child (e.g. ``reconstruct_pheno_A_2_ps0_pe100``).
    This helper rolls those up into single per-step numbers — call it after
    the fan-out completes to keep step-level reporting honest.

    Returns an empty dict if no matching subtasks are found.
    """
    import fcntl

    if not os.path.exists(log_file_path):
        return {}

    prefix = parent_log_key + "_"
    lock_path = str(log_file_path) + ".lock"
    try:
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_SH)
            try:
                with open(log_file_path, "r") as f:
                    log_data = yaml.safe_load(f) or {}
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: aggregate_subtask_metrics failed to read log: {e}")
        return {}

    children = {
        k: v for k, v in log_data.items()
        if k.startswith(prefix) and k != parent_log_key and isinstance(v, dict)
    }
    if not children:
        return {}

    def _nums(field):
        return [v[field] for v in children.values()
                if isinstance(v.get(field), (int, float))]

    def _set(field):
        return sorted({str(v[field]) for v in children.values() if v.get(field)})

    sm_avgs = _nums("gpu_device_sm_util_avg_pct")
    sm_maxes = _nums("gpu_device_sm_util_max_pct")
    mem_used_mib = _nums("gpu_device_mem_used_mib")
    mem_used_pct = _nums("gpu_device_mem_used_percent")
    pow_avgs = _nums("gpu_power_watts_draw_avg")
    pow_maxes = _nums("gpu_power_watts_draw_max")
    cpu_util_avgs = _nums("cpu_util_process_average_percent")
    ran_in_seconds = []
    for v in children.values():
        r = v.get("Ran in") or v.get("ran_in")
        if isinstance(r, str) and r.endswith("seconds"):
            try:
                ran_in_seconds.append(float(r.split()[0]))
            except (ValueError, IndexError):
                pass
        elif isinstance(r, (int, float)):
            ran_in_seconds.append(float(r))

    summary = {"n_subtasks": len(children)}
    if sm_avgs:
        summary["children_gpu_sm_util_avg_pct_mean"] = round(sum(sm_avgs) / len(sm_avgs), 2)
        summary["children_gpu_sm_util_avg_pct_min"] = round(min(sm_avgs), 2)
        summary["children_gpu_sm_util_avg_pct_max"] = round(max(sm_avgs), 2)
    if sm_maxes:
        summary["children_gpu_sm_util_max_pct"] = round(max(sm_maxes), 2)
    if mem_used_mib:
        summary["children_gpu_mem_used_mib_max"] = round(max(mem_used_mib), 1)
    if mem_used_pct:
        summary["children_gpu_mem_used_percent_max"] = round(max(mem_used_pct), 2)
    if pow_avgs:
        summary["children_gpu_power_watts_avg"] = round(sum(pow_avgs) / len(pow_avgs), 1)
    if pow_maxes:
        summary["children_gpu_power_watts_max"] = round(max(pow_maxes), 1)
    if cpu_util_avgs:
        summary["children_cpu_util_pct_avg"] = round(sum(cpu_util_avgs) / len(cpu_util_avgs), 2)
    if ran_in_seconds:
        summary["children_ran_in_seconds_total"] = round(sum(ran_in_seconds), 2)
        summary["children_ran_in_seconds_max"] = round(max(ran_in_seconds), 2)

    hostnames = _set("hostname")
    if hostnames:
        summary["children_hostnames"] = hostnames
    gpu_models = _set("gpu_models")
    if gpu_models:
        summary["children_gpu_models"] = gpu_models

    return summary


def emit_subtask_summary(
    experiment: str,
    child_key_prefix: str,
    summary_key: str | None = None,
) -> dict:
    """High-level wrapper for parent steps. Resolves the experiment's
    function_call_log path, aggregates subtask metrics matching
    ``child_key_prefix``, and writes the summary under
    ``<summary_key or child_key_prefix>__summary``.

    Never raises — logging failures are warned and swallowed so the parent
    step's actual work isn't affected.
    """
    try:
        from cyclops_utils.data.experiment import OpsDataset
        ds = OpsDataset(experiment)
        return write_subtask_summary(ds.logfile, child_key_prefix, summary_key)
    except Exception as e:
        print(f"Warning: emit_subtask_summary skipped ({child_key_prefix}): {e}")
        return {}


def write_subtask_summary(
    log_file_path,
    child_key_prefix: str,
    summary_key: str | None = None,
) -> dict:
    """Aggregate subtasks whose key begins with ``child_key_prefix + "_"`` and
    write the summary back into the log.

    Many parent steps dispatch a different function than they're named after
    (e.g. ``reconstruct_tilt_corrected_pheno`` fans out via
    ``reconstruct(process="pheno", ...)``), so the search prefix and the
    summary key need to be specified separately. ``summary_key`` defaults
    to ``child_key_prefix`` if omitted; the summary is written under
    ``<summary_key>__summary``.

    Returns the summary dict (empty if no subtasks were found).
    """
    summary = aggregate_subtask_metrics(log_file_path, child_key_prefix)
    if not summary:
        return {}
    sk = (summary_key or child_key_prefix) + "__summary"
    _load_and_write_log(sk, log_file_path, {sk: summary})
    return summary


def _get_git_commit_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
