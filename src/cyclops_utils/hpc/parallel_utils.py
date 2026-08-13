import os
from multiprocessing import get_context

from dask.distributed import Client, LocalCluster
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None


def _spawn_runner(q, func, args, kwargs):
    """Top-level runner used by spawned child process.

    Must be defined at module top-level to be picklable with the spawn method.
    """
    try:
        func(*args, **kwargs)
        q.put(None)
    except Exception as e:  # pragma: no cover - propagate full traceback
        import traceback

        q.put((str(e), traceback.format_exc()))


def call_in_spawned_process(func, *args, **kwargs) -> None:
    """Execute `func(*args, **kwargs)` in a fresh "spawn" process.

    This isolates thread/interop settings (e.g., torch.set_num_interop_threads)
    from prior parallel work in the parent process.
    """
    ctx = get_context("spawn")
    error_queue = ctx.Queue()

    process = ctx.Process(target=_spawn_runner, args=(error_queue, func, args, kwargs))
    process.start()
    process.join()

    if process.exitcode != 0:
        if not error_queue.empty():
            message, tb = error_queue.get()
            raise RuntimeError(f"Child process failed: {message}\n{tb}")
        raise RuntimeError(f"Child process exited with code {process.exitcode}")


def _cleanup_worker_memory(*variables) -> None:
    """Clean up worker memory and CUDA cache."""
    for var in variables:
        del var
    torch.cuda.empty_cache()


def _verify_gpu_env():
    """Return (n_devices, msg) for CUDA as seen by a Dask worker. n_devices=0 means
    the worker cannot use a GPU (broken/incompatible CUDA env) and would fall back to CPU."""
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    try:
        import torch
        n_devices = torch.cuda.device_count()
        if n_devices > 0:
            name = torch.cuda.get_device_properties(0).name
            return n_devices, f"CUDA_VISIBLE_DEVICES={cuda_vis}, devices={n_devices}, gpu0={name}"
        return 0, f"CUDA_VISIBLE_DEVICES={cuda_vis}, no CUDA devices"
    except Exception as e:
        return 0, f"CUDA_VISIBLE_DEVICES={cuda_vis}, error: {e}"


class MultiGPUCluster:
    """Creates one LocalCluster per GPU with CUDA_VISIBLE_DEVICES set before spawn.

    LocalCluster uses fork on Linux. Setting CUDA_VISIBLE_DEVICES via os.environ
    after fork is silently ignored by the CUDA driver. This class solves it by
    creating separate clusters, each with env={'CUDA_VISIBLE_DEVICES': gpu_id}
    set BEFORE workers spawn (via Nanny).

    Usage:
        with MultiGPUCluster([0, 1], workers_per_gpu=19) as mgc:
            futures = [mgc.submit(func, arg) for arg in args]
            for future in as_completed(futures):
                result = future.result()
    """

    def __init__(self, available_gpus, workers_per_gpu, threads_per_worker=1,
                 env=None, **kwargs):
        self.clusters = []
        self.clients = []
        self._submit_counter = 0

        # Save the dask global scheduler setting BEFORE creating any
        # Client. distributed.Client.__init__ calls
        # dask.config.set({"scheduler": "distributed"}) and Client.close()
        # does NOT undo it — so without restoring, every downstream
        # da.compute() in the same process fails with
        # "Requested dask.distributed scheduler but no Client active."
        # (We've seen this from get_metrics's iss_heatmaps / iss_stats.)
        import dask
        try:
            self._saved_dask_scheduler = dask.config.get("scheduler")
        except KeyError:
            self._saved_dask_scheduler = None

        # Build env: CUDA_VISIBLE_DEVICES per GPU + caller overrides.
        # Propagate OMP_NUM_THREADS from parent to prevent Dask from
        # clamping it to 1 (its default for LocalCluster).
        import os
        base_env = {}
        omp = os.environ.get("OMP_NUM_THREADS")
        if omp:
            base_env["OMP_NUM_THREADS"] = omp
        if env:
            base_env.update(env)

        # Clear CUDA_VISIBLE_DEVICES in the parent BEFORE creating clusters.
        # If CUDA was initialized in the parent (e.g., by CuPy import),
        # child processes inherit the context and ignore env overrides.
        # Clearing it here + setting per-cluster ensures workers only see
        # their assigned GPU.
        _parent_cuda = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        for gpu_id in available_gpus:
            worker_env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu_id))
            cluster = LocalCluster(
                n_workers=workers_per_gpu,
                threads_per_worker=threads_per_worker,
                env=worker_env,
                **kwargs,
            )
            client = Client(cluster, timeout="2m")
            self.clusters.append(cluster)
            self.clients.append(client)

        # Restore parent CUDA_VISIBLE_DEVICES (for stitching phase etc.)
        if _parent_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _parent_cuda

        # Verify GPU assignment
        total_workers = workers_per_gpu * len(available_gpus)
        print(f"  [MultiGPU] Created {len(available_gpus)} clusters, "
              f"{workers_per_gpu} workers/GPU, {total_workers} total workers")
        for i, client in enumerate(self.clients):
            n_devices, gpu_check = client.submit(_verify_gpu_env, pure=False).result()
            print(f"  [MultiGPU] Cluster {i} (GPU {available_gpus[i]}): "
                  f"workers see CUDA_VISIBLE_DEVICES={gpu_check}")
            # Fail fast: a worker with no usable CUDA device silently falls back to
            # CPU, which on large workloads just times out and leaves empty output.
            # Better to die loudly now so the job can requeue onto a working node.
            if n_devices == 0:
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"GPU verification failed on cluster {i} (GPU {available_gpus[i]}): "
                    f"workers report no usable CUDA device ({gpu_check}). Aborting instead "
                    f"of falling back to CPU. This node's CUDA env is broken/incompatible."
                )

    @property
    def dashboard_link(self):
        return self.clients[0].dashboard_link if self.clients else None

    def submit(self, func, *args, **kwargs):
        """Submit to next client (round-robin)."""
        client = self.clients[self._submit_counter % len(self.clients)]
        self._submit_counter += 1
        return client.submit(func, *args, **kwargs)

    def map(self, func, items):
        """Round-robin distribute items across GPU clusters."""
        futures = []
        for i, item in enumerate(items):
            client = self.clients[i % len(self.clients)]
            futures.append(client.submit(func, item))
        return futures

    def close(self):
        """Graceful shutdown of all clusters with suppressed Dask noise."""
        import logging
        import warnings

        # Suppress ALL distributed logging during shutdown (heartbeat errors, comm
        # closed, nanny close, etc. all fire asynchronously at ERROR level)
        logging.getLogger("distributed").setLevel(logging.CRITICAL)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for client in self.clients:
                try:
                    client.close()
                except Exception:
                    pass
            for cluster in self.clusters:
                try:
                    cluster.close()
                except Exception:
                    pass

        # Restore the dask global scheduler config we captured in __init__.
        # Without this, downstream da.compute() in the same process fails
        # with "Requested dask.distributed scheduler but no Client active."
        import dask
        if self._saved_dask_scheduler is None:
            try:
                # Clear the override (back to dask's default scheduler).
                dask.config.set({"scheduler": None})
            except Exception:
                pass
        else:
            try:
                dask.config.set({"scheduler": self._saved_dask_scheduler})
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class GPUPinnedSpecCluster:
    """Single Dask scheduler with N workers, each pinned to a specific GPU.

    Difference vs MultiGPUCluster:
    - MultiGPUCluster: N separate LocalClusters (one per GPU), each with
      its own scheduler. Jobs are statically round-robin distributed to
      a cluster at submit time. No cross-GPU work stealing — if one
      GPU's chunks finish early, it sits idle.
    - GPUPinnedSpecCluster: ONE Scheduler with N Nanny workers spawned
      via SpecCluster. Each Nanny gets a per-worker ``env`` setting
      ``CUDA_VISIBLE_DEVICES`` to its pinned GPU BEFORE spawn (so the
      CUDA driver respects it — same trick as MultiGPUCluster, just
      one-scheduler instead of per-cluster). Single scheduler means
      Dask's central task queue feeds workers dynamically and supports
      work stealing.

    Designed for workloads where chunks are independent but variable in
    cost: static round-robin leaves the GPU with the small chunks idle
    near the end of the run. With one scheduler the idle worker pulls
    the next available task from any other worker's queue.

    Caveat: needs forkserver multiprocessing pre-warmed BEFORE any CUDA
    import in the parent process — same constraint MultiGPUCluster has.
    Our iss_pro6000_runner.py handles that.
    """

    def __init__(self, available_gpus, workers_per_gpu=1,
                 threads_per_worker=1, env=None):
        import os
        from distributed import SpecCluster, Scheduler, Nanny, Client

        # Save the global scheduler config so we can restore on close
        # (same pattern as MultiGPUCluster).
        import dask
        try:
            self._saved_dask_scheduler = dask.config.get("scheduler")
        except KeyError:
            self._saved_dask_scheduler = None

        # Capture and clear the parent's CUDA_VISIBLE_DEVICES so workers
        # don't inherit a wider device list than the env={...} they're
        # supposed to get.
        self._parent_cvd = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        base_env = {}
        omp = os.environ.get("OMP_NUM_THREADS")
        if omp:
            base_env["OMP_NUM_THREADS"] = omp
        if env:
            base_env.update(env)

        # One Nanny spec per worker. Within a single SpecCluster you can
        # have heterogeneous worker options (different env per worker),
        # which is what we use to pin each worker to a specific GPU.
        worker_specs = {}
        for i in range(workers_per_gpu * len(available_gpus)):
            gpu_id = available_gpus[i % len(available_gpus)]
            worker_env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu_id))
            worker_specs[f"w{i}-gpu{gpu_id}"] = {
                "cls": Nanny,
                "options": {
                    "env": worker_env,
                    "nthreads": threads_per_worker,
                    "memory_limit": 0,
                },
            }

        scheduler_spec = {"cls": Scheduler, "options": {"dashboard_address": ":0"}}
        self._cluster = SpecCluster(
            workers=worker_specs, scheduler=scheduler_spec, asynchronous=False,
        )
        self._client = Client(self._cluster, timeout="2m")

        # Restore parent CUDA_VISIBLE_DEVICES (for any in-parent work
        # after this cluster shuts down).
        if self._parent_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._parent_cvd

        total = workers_per_gpu * len(available_gpus)
        print(f"  [GPUPinnedSpec] One scheduler, {total} workers "
              f"({workers_per_gpu}/GPU × {len(available_gpus)} GPUs); "
              f"dynamic dispatch with work stealing")

    @property
    def client(self):
        return self._client

    @property
    def dashboard_link(self):
        return self._client.dashboard_link

    def submit(self, func, *args, **kwargs):
        # Single client; the central scheduler picks an available worker.
        return self._client.submit(func, *args, **kwargs)

    def map(self, func, items):
        return self._client.map(func, items)

    def close(self):
        import logging
        import warnings
        logging.getLogger("distributed").setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._client.close()
            except Exception:
                pass
            try:
                self._cluster.close()
            except Exception:
                pass
        import dask
        if self._saved_dask_scheduler is None:
            try:
                dask.config.set({"scheduler": None})
            except Exception:
                pass
        else:
            try:
                dask.config.set({"scheduler": self._saved_dask_scheduler})
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run_jobs_inproc(jobs_to_submit, available_gpus, workers_per_gpu=1,
                    work_stealing=False):
    """In-process replacement for ``slurm_batch_utils.submit_parallel_jobs``.

    Why this exists: a few pipeline steps (reconstruct_tilt_corrected,
    virtual_staining_inference, register_iss_cycles) fan out to slurm
    array sub-jobs via submitit. That's fine when the parent only
    orchestrates and holds no GPU. But our reservation-pinned runs
    allocate the parent's GPUs at sbatch time and then sit them idle
    waiting on children that land on arbitrary other nodes. This helper
    runs the same jobs in-process on the parent's GPUs via
    MultiGPUCluster, returning the same dict shape so callers don't
    have to branch beyond a single ``if`` at the call site.

    Each job dict needs ``"name"``, ``"func"``, and ``"kwargs"`` (matching
    submit_parallel_jobs' job format).
    """
    if not jobs_to_submit:
        return {"success": True, "all_completed": True, "failed": [],
                "completed": []}

    failed = []
    completed = []

    if work_stealing:
        print(f"  [inproc/ws] Running {len(jobs_to_submit)} jobs across "
              f"{len(available_gpus)} GPU(s) × {workers_per_gpu} worker(s)/GPU "
              f"with dynamic work stealing")
        cluster_cls = GPUPinnedSpecCluster
    else:
        print(f"  [inproc] Running {len(jobs_to_submit)} jobs across "
              f"{len(available_gpus)} GPU(s) × {workers_per_gpu} worker(s)/GPU "
              f"(static round-robin)")
        cluster_cls = MultiGPUCluster

    with cluster_cls(available_gpus, workers_per_gpu=workers_per_gpu) as mgc:
        future_to_name = {}
        for job in jobs_to_submit:
            fut = mgc.submit(job["func"], **job.get("kwargs", {}))
            future_to_name[fut] = job["name"]

        for fut, name in list(future_to_name.items()):
            try:
                fut.result()
                completed.append(name)
                print(f"    ✓ {name}")
            except Exception as e:
                failed.append(name)
                print(f"    ✗ {name}: {type(e).__name__}: {e}")

    return {
        "success": True,
        "all_completed": len(failed) == 0,
        "failed": failed,
        "completed": completed,
    }


