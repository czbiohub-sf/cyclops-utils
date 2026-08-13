"""Return type for launcher steps that submit inner SLURM job arrays.

The DAG runner checks isinstance(result, LauncherResult) and if so,
enters a polling loop to track the actual inner jobs instead of just
the lightweight wrapper.
"""

from dataclasses import dataclass, field


@dataclass
class JobArray:
    """A single SLURM job array submitted by a launcher step.

    Attributes:
        base_job_id: The SLURM base job ID for sacct queries.
        submitted_jobs: List of dicts, each with "job" (submitit.Job),
            "name" (str), and optionally "metadata" (dict).
        label: Optional label for display (e.g., "base", "seg", "image").
    """

    base_job_id: str
    submitted_jobs: list[dict]
    label: str = ""


@dataclass
class LauncherResult:
    """Returned by launcher steps when wait_for_completion=False.

    Supports both single and multi-batch submissions. For steps that submit
    multiple job arrays (e.g., convert_v3 submits base + seg per store,
    build_napari submits image + seg), all arrays are tracked together.

    Attributes:
        job_arrays: List of JobArray objects to track.
        total_jobs: Total number of inner jobs across all arrays.
    """

    job_arrays: list[JobArray] = field(default_factory=list)
    total_jobs: int = 0

    @property
    def base_job_id(self) -> str:
        """Primary job ID for display (first array)."""
        return self.job_arrays[0].base_job_id if self.job_arrays else ""

    @property
    def submitted_jobs(self) -> list[dict]:
        """Flattened list of all submitted jobs across all arrays."""
        return [job for arr in self.job_arrays for job in arr.submitted_jobs]

    @staticmethod
    def from_submit_result(result: dict, label: str = "") -> "LauncherResult":
        """Create a LauncherResult from a single submit_parallel_jobs result.

        Args:
            result: Dict returned by submit_parallel_jobs(wait_for_completion=False).
            label: Optional label for display.
        """
        arr = JobArray(
            base_job_id=result["base_job_id"],
            submitted_jobs=result["submitted_jobs"],
            label=label,
        )
        return LauncherResult(
            job_arrays=[arr],
            total_jobs=len(result["submitted_jobs"]),
        )

    def add_array(self, result: dict, label: str = ""):
        """Add another job array from a submit_parallel_jobs result.

        Args:
            result: Dict returned by submit_parallel_jobs(wait_for_completion=False).
            label: Optional label for display.
        """
        arr = JobArray(
            base_job_id=result["base_job_id"],
            submitted_jobs=result["submitted_jobs"],
            label=label,
        )
        self.job_arrays.append(arr)
        self.total_jobs += len(result["submitted_jobs"])
