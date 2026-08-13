import os
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None
from dask.distributed import WorkerPlugin


class GPUAssignmentPlugin(WorkerPlugin):
    """Dask worker plugin to assign GPUs to workers on startup.

    This plugin runs once when each worker starts up and assigns it to a specific GPU
    using the worker's name/address to determine GPU assignment in a deterministic way.
    """

    def __init__(self, available_gpus):
        self.available_gpus = available_gpus

    def setup(self, worker):
        """Called when the worker is created."""
        # Store original CUDA_VISIBLE_DEVICES for debugging
        cuda_before = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

        # Use the worker's address port number for deterministic GPU assignment
        # Worker addresses are like "tcp://127.0.0.1:40789"
        try:
            worker_addr = str(worker.address)
            # Extract port number from address
            port = int(worker_addr.split(':')[-1])
            # Use port number modulo number of GPUs for assignment
            worker_gpu_idx = port % len(self.available_gpus)
            gpu_to_use = self.available_gpus[worker_gpu_idx]
        except (ValueError, AttributeError, IndexError):
            # Fallback: use hash of worker name
            gpu_to_use = self.available_gpus[abs(hash(str(worker.name))) % len(self.available_gpus)]

        # Set CUDA_VISIBLE_DEVICES for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)

        cuda_after = os.environ.get("CUDA_VISIBLE_DEVICES")

        print(f"[GPU Plugin Setup] Worker {worker.address} | Before: {cuda_before} | Assigned GPU: {gpu_to_use} | After: {cuda_after} | Available: {self.available_gpus}")

        # Store assignment on worker for later reference
        worker.gpu_id = gpu_to_use


def _setup_gpu_environment() -> list:
    """Setup GPU environment for Dask workers.

    Returns the list of GPU indices available to this job.

    IMPORTANT: Does NOT call torch.cuda to avoid initializing CUDA in the
    parent process. If CUDA is initialized before MultiGPUCluster spawns
    workers, all child processes inherit the parent's CUDA context and
    ignore CUDA_VISIBLE_DEVICES — all workers end up on GPU 0.

    Instead, parses CUDA_VISIBLE_DEVICES or queries nvidia-smi.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print(f"[GPU Setup] CUDA_VISIBLE_DEVICES in main process: {cuda_visible or 'not set'}")

    if cuda_visible:
        # SLURM sets CUDA_VISIBLE_DEVICES to the assigned GPU indices
        num_gpus = len(cuda_visible.split(","))
    else:
        # No CUDA_VISIBLE_DEVICES — query nvidia-smi for GPU count
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        num_gpus = len(result.stdout.strip().split("\n"))

    available_gpus = list(range(num_gpus))
    print(f"[GPU Setup] Detected {num_gpus} GPU(s), available indices: {available_gpus}")

    return available_gpus
