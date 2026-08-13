"""Process-level CPU, memory, and I/O monitoring via /proc.

Provides lightweight monitoring of a process and its children by reading
/proc/[pid]/stat and /proc/[pid]/io directly (no external dependencies).

Usage::

    from cyclops_utils.profiling.proc_monitor import start_monitor

    stop = start_monitor(interval=5.0, n_cores=32)
    # ... do work ...
    stop.set()  # signal the monitor thread to exit
"""

from __future__ import annotations

import os
import threading


def read_proc_stat(pid: int) -> tuple[int | None, int | None, float | None]:
    """Read /proc/[pid]/stat for CPU ticks, thread count, RSS.

    Returns:
        (cpu_ticks, num_threads, rss_mb) or (None, None, None) on error.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        # utime(13), stime(14), num_threads(19), rss(23)
        cpu_ticks = int(fields[13]) + int(fields[14])
        num_threads = int(fields[19])
        page_size = os.sysconf("SC_PAGE_SIZE")
        rss_mb = int(fields[23]) * page_size / 1e6
        return cpu_ticks, num_threads, rss_mb
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None, None, None


def read_proc_io(pid: int) -> tuple[int, int]:
    """Read /proc/[pid]/io for read/write bytes.

    Returns:
        (read_bytes, write_bytes). Returns (0, 0) on error.
    """
    try:
        with open(f"/proc/{pid}/io") as f:
            data = {}
            for line in f:
                k, v = line.split(":")
                data[k.strip()] = int(v.strip())
        return data.get("read_bytes", 0), data.get("write_bytes", 0)
    except (FileNotFoundError, PermissionError, ValueError):
        return 0, 0


def get_child_pids(parent_pid: int) -> list[int]:
    """Get direct child PIDs via /proc/[pid]/task/[tid]/children."""
    children = []
    try:
        task_dir = f"/proc/{parent_pid}/task"
        for tid in os.listdir(task_dir):
            child_file = f"{task_dir}/{tid}/children"
            try:
                with open(child_file) as f:
                    for pid_str in f.read().split():
                        children.append(int(pid_str))
            except (FileNotFoundError, PermissionError, ValueError):
                continue
    except (FileNotFoundError, PermissionError):
        pass
    return list(set(children))


def _monitor_loop(
    stop_event: threading.Event,
    interval: float,
    n_cores: int,
) -> None:
    """Background thread: log per-worker CPU%, thread count, IO rates."""
    clk_tck = os.sysconf("SC_CLK_TCK")
    pid = os.getpid()
    prev: dict[int, tuple[int, int, int]] = {}

    while not stop_event.wait(interval):
        children = get_child_pids(pid)
        all_pids = [pid] + children

        total_cpu_pct = 0.0
        total_threads = 0
        total_rss = 0.0
        total_read_rate = 0.0
        total_write_rate = 0.0
        worker_parts = []

        for p in all_pids:
            cpu_ticks, n_threads, rss_mb = read_proc_stat(p)
            if cpu_ticks is None:
                continue
            read_bytes, write_bytes = read_proc_io(p)

            prev_vals = prev.get(p)
            if prev_vals:
                dt_cpu = cpu_ticks - prev_vals[0]
                dt_read = read_bytes - prev_vals[1]
                dt_write = write_bytes - prev_vals[2]
                cpu_pct = 100.0 * dt_cpu / (clk_tck * interval)
                read_rate = dt_read / interval / 1e6  # MB/s
                write_rate = dt_write / interval / 1e6  # MB/s
            else:
                cpu_pct = read_rate = write_rate = 0.0

            prev[p] = (cpu_ticks, read_bytes, write_bytes)
            total_cpu_pct += cpu_pct
            total_threads += n_threads
            total_rss += rss_mb
            total_read_rate += read_rate
            total_write_rate += write_rate

            if p != pid:
                worker_parts.append(f"{cpu_pct:.0f}%/{n_threads}t")

        core_util = total_cpu_pct / n_cores if n_cores > 0 else 0
        workers_str = " ".join(worker_parts) if worker_parts else "none"
        print(
            f"[CPU Monitor] cores={core_util:.0f}%of{n_cores} "
            f"total_cpu={total_cpu_pct:.0f}% threads={total_threads} "
            f"rss={total_rss / 1e3:.1f}GB "
            f"read={total_read_rate:.0f}MB/s write={total_write_rate:.0f}MB/s "
            f"workers=[{workers_str}]"
        )

    print("[CPU Monitor] stopped")


def start_monitor(interval: float = 5.0, n_cores: int = 32) -> threading.Event:
    """Start background CPU/IO monitor.

    Args:
        interval: Seconds between samples.
        n_cores: Total CPU cores (for utilization percentage).

    Returns:
        A ``threading.Event`` — call ``.set()`` to stop the monitor.
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=_monitor_loop,
        args=(stop_event, interval, n_cores),
        daemon=True,
        name="cpu-monitor",
    )
    t.start()
    return stop_event
