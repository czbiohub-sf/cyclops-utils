"""
Async zarr writer for overlapping I/O with GPU compute.

This module provides an AsyncZarrWriter class that writes zarr arrays
in a background thread, allowing GPU compute to continue while I/O happens.
"""

import threading
import queue
from pathlib import Path
from typing import Optional, Any
import numpy as np
import zarr
import time


class AsyncZarrWriter:
    """
    Async writer that writes zarr arrays in a background thread.

    This allows GPU compute to continue while I/O happens in parallel,
    reducing GPU idle time from ~25% to ~3-4%.

    Usage:
        writer = AsyncZarrWriter(output_store_path, max_queue_size=4)
        writer.start()

        for batch in dataloader:
            output = model(batch)
            writer.write(f'position_{i}', output.cpu().numpy(), ...)

        writer.stop()  # Wait for all writes to complete

    The writer uses a bounded queue to prevent memory buildup if compute
    is faster than I/O. When the queue is full, write() will block until
    there's space.
    """

    def __init__(
        self,
        output_path: Path,
        max_queue_size: int = 4,
        mode: str = 'w',
    ):
        """
        Initialize async writer.

        Args:
            output_path: Path to zarr store
            max_queue_size: Maximum number of pending writes (bounds memory usage)
            mode: Zarr open mode ('w' for write, 'a' for append)
        """
        self.output_path = Path(output_path)
        self.max_queue_size = max_queue_size
        self.mode = mode

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._store: Optional[zarr.Group] = None
        self._error: Optional[Exception] = None

        # Stats for profiling
        self.total_write_time = 0.0
        self.total_writes = 0
        self.total_bytes_written = 0
        self.queue_full_waits = 0

    def start(self):
        """Start the background writer thread."""
        if self._thread is not None:
            raise RuntimeError("Writer already started")

        # Open zarr store
        self._store = zarr.open(str(self.output_path), mode=self.mode)

        # Start background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def write(
        self,
        name: str,
        data: np.ndarray,
        chunks: Optional[tuple] = None,
        dtype: Optional[np.dtype] = None,
        attrs: Optional[dict] = None,
    ):
        """
        Queue data for async writing.

        This method may block if the queue is full (backpressure).

        Args:
            name: Dataset name in zarr store
            data: Numpy array to write
            chunks: Chunk shape (defaults to data shape)
            dtype: Data type (defaults to data dtype)
            attrs: Optional attributes to attach to dataset
        """
        if self._thread is None:
            raise RuntimeError("Writer not started - call start() first")

        if self._error is not None:
            raise self._error

        # Check if queue is full (for stats)
        if self._queue.full():
            self.queue_full_waits += 1

        # Queue the write (blocks if full)
        self._queue.put({
            'name': name,
            'data': data,
            'chunks': chunks or data.shape,
            'dtype': dtype or data.dtype,
            'attrs': attrs,
        })

    def stop(self, timeout: float = 60.0) -> dict:
        """
        Stop the writer and wait for all pending writes to complete.

        Args:
            timeout: Maximum time to wait for pending writes

        Returns:
            dict with stats: total_write_time, total_writes, avg_write_time,
                            total_bytes_written, queue_full_waits
        """
        if self._thread is None:
            return self._get_stats()

        # Signal stop
        self._stop_event.set()

        # Wait for thread to finish
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            print(f"Warning: Writer thread did not stop within {timeout}s timeout")

        self._thread = None

        if self._error is not None:
            raise self._error

        return self._get_stats()

    def _writer_loop(self):
        """Background thread that processes write requests."""
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    # Get next write request (with timeout to check stop event)
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Perform the write
                start_time = time.perf_counter()

                try:
                    # zarr 3.x removed Group.create_dataset (v2 API); create_array is
                    # the v3 replacement (Fix 0l — zarr 3.2.1 from Fix 0b). NOTE: v3
                    # create_array forbids passing `data` AND `shape`/`dtype` together
                    # (it infers shape+dtype from data), so pass only `data`, cast to
                    # the requested dtype (no-op/no-copy when already matching).
                    arr = self._store.create_array(
                        item['name'],
                        data=item['data'].astype(item['dtype'], copy=False),
                        chunks=item['chunks'],
                        overwrite=True,
                    )

                    if item['attrs']:
                        arr.attrs.update(item['attrs'])

                    # Update stats
                    write_time = time.perf_counter() - start_time
                    self.total_write_time += write_time
                    self.total_writes += 1
                    self.total_bytes_written += item['data'].nbytes

                except Exception as e:
                    self._error = e
                    break

                finally:
                    self._queue.task_done()

        except Exception as e:
            self._error = e

    def _get_stats(self) -> dict:
        """Get writer statistics."""
        avg_write_time = (
            self.total_write_time / self.total_writes
            if self.total_writes > 0 else 0
        )
        return {
            'total_write_time': self.total_write_time,
            'total_writes': self.total_writes,
            'avg_write_time': avg_write_time,
            'total_bytes_written': self.total_bytes_written,
            'total_mb_written': self.total_bytes_written / (1024 * 1024),
            'queue_full_waits': self.queue_full_waits,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
