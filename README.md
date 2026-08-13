# cyclops_utils

Shared utility library for the Optical Pooled Screening (OPS) pipelines at biohub San Francisco.

`cyclops_utils` holds the code that more than one OPS repo needs: experiment path
resolution, OME-Zarr I/O, SLURM submission and monitoring, resource sizing,
profiling/notification decorators, and analysis helpers (mAP scoring, PCA, UMAP,
normalization). It is a library, not a pipeline, and is not intended to be used
directly — it exists to centralize support functions shared by
[`cyclops_process`](https://github.com/czbiohub-sf/cyclops-process),
[`cyclops_model`](https://github.com/czbiohub-sf/cyclops-model), and
[`organelle_profiler`](https://github.com/czbiohub-sf/organelle-profiler).

## Preprint

This repository accompanies the following preprint, and should be preserved and kept public
indefinitely:

> [A multimodal perturbation atlas defines the phenotypic resolution of cellular morphology](https://www.biorxiv.org/content/10.64898/2026.06.01.728087v1.abstract) — bioRxiv, 2026. doi:10.64898/2026.06.01.728087

## Data availability

The processed image datasets that the OPS pipelines built on this library ingest are available for
download through the Biohub OPS Explorer portal:

> [OPS Explorer — perturbation atlas collection](https://biohub.ai/ops-explorer?collection=6a3f8b91-1c5e-4d3a-9b4c-f7e0a2d8b6f3)

## Installation

### Via the monorepo (recommended)

`cyclops_utils` is a submodule of
[czbiohub-sf/cyclops-monorepo](https://github.com/czbiohub-sf/cyclops-monorepo), which
installs every OPS package into one [uv](https://docs.astral.sh/uv/) workspace:

```bash
git clone --recurse-submodules git@github.com:czbiohub-sf/cyclops-monorepo.git
cd cyclops-monorepo
uv sync
```

### Optional dependency groups

The base install is deliberately light (numpy, pandas, zarr, iohub, scikit-image,
click, joblib, tqdm, psutil, pyyaml). Everything else is an extra:

| Extra | Pulls in | Needed for |
|---|---|---|
| `gpu` | `torch`, `cupy-cuda12x`, `pynvml`, `monai` | `hpc.gpu_utils`, `hpc.parallel_utils`, GPU dataloaders |
| `hpc` | `submitit`, `dask` | `hpc.parallel_utils` clusters |
| `slack` | `slack-sdk`, `python-dotenv`, `certifi` | `profiling.slack_notifier` |
| `all` | the above + `tensorstore`, `anndata`, `prettytable` | full functionality |

```bash
uv pip install -e ".[all]"
```

Declared dependencies and the optional extras above live in
[`pyproject.toml`](pyproject.toml); Python 3.12 is required. Exact resolved versions for the whole
workspace are pinned in the monorepo's
[`uv.lock`](https://github.com/czbiohub-sf/cyclops-monorepo/blob/main/uv.lock), which is the
authoritative environment specification.

## Environment configuration

The library ships no site-specific paths, so storage roots come from the
environment. **`OPS_BASE_PATH` is required** — importing any module that
resolves paths raises a `RuntimeError` if it is unset, rather than silently
reading or writing somebody else's storage:

```bash
# Required: root of the shared OPS data tree
export OPS_BASE_PATH=/path/to/ops_data
```

Optional overrides:

```bash
# Mounts holding the raw acquisitions, read only by the conversion steps
# (default: $OPS_BASE_PATH/raw/iss and $OPS_BASE_PATH/raw/dragonfly)
export OPS_INSTRUMENT_ROOT=/path/to/iss_tiles        # one dir per experiment
export OPS_DRAGONFLY_ROOT=/path/to/dragonfly         # one dir per OPS key

# Pipeline outputs, and a faster partition for the hot ones
export OPS_OUTPUT_BASE_DIR=/path/to/ops_data
export OPS_FAST_OUTPUT_BASE_DIR=/path/to/fast/ops_data

# Configs (codebook, channel maps, ...) and shared seed affines
export OPS_CONFIGS_DIR=/path/to/configs
export OPS_AFFINES_DIR=/path/to/configs/affines

# Shared root for operational-mode dual-written logs
# (default: $OPS_BASE_PATH/logs)
export OPS_LOG_ROOTDIR=/path/to/monitoring/logs
```

**Verify:**

```bash
uv run python -c "import cyclops_utils; from cyclops_utils.data.experiment import OpsDataset; print('OK')"
```

## Package layout

There is no re-export at the top level — import from the submodule directly
(`from cyclops_utils.data.experiment import OpsDataset`).

### `cyclops_utils.data` — experiments, paths, metadata

| Module | What it provides |
|---|---|
| `experiment` | `OpsDataset` — the canonical experiment object. Resolves every pipeline directory (`0-convert/`, `1-preprocess/`, `2-tracking/`, `3-assembly/`, …), loads the codebook and gene index, infers wells, and checks store existence/shape. |
| `filesystem` | Path and naming helpers: `resolve_experiment_name`, `extract_ops_key`, `parse_well`, `canonicalize_well_path`, `canonicalize_channel_name`, `get_experiment_wells`, `setup_experiment_directories`, `find_monorepo_root`, overwrite/resume/skip prompts. |
| `bad_experiments` | Single source of truth for experiment exclusion lists (`bad`, `iss_only`, `do_not_run`, `non_standard`, `positive_control`, `need_rescue`) loaded from `bad_experiment.yaml` under `OPS_CONFIGS_DIR` (empty — nothing excluded — when that file is absent): `is_excluded`, `get_category`, `get_reason`, `get_date_cutoff`. |
| `feature_metadata` | `FeatureMetadata` — maps `(experiment, channel)` to biological signal via `ops_channel_maps.yaml`, and builds informative feature names. |
| `feature_discovery` | Discovery of experiments with DINO / CellProfiler feature files, and grouping `(experiment, channel)` pairs by biological signal. |
| `cell_data_loader` | `CellDataLoader` / `FlexibleCellDataset` — extract single cells at natural boundaries or cropped to a fixed patch size. |
| `bbox_utils` | Bounding-box normalization and the `BaseDataset` crop reader (requires `iohub>=0.3.7`). |
| `disk_cache` | `df_cache` — parquet sidecar cache for expensive DataFrames, mtime-keyed to the source file. |
| `naming`, `image_utils`, `shifts`, `positive_controls` | Channel naming/typing, pure-NumPy image helpers, tile registration shift readers, CHAD positive-control clusters. |

### `cyclops_utils.io` — OME-Zarr

| Module | What it provides |
|---|---|
| `zarr_utils` | The large one: format detection, well listing, array creation, pyramid level management (`ensure_pyramid_levels`), resharding, metadata repair, parallel slice writes. |
| `zarr_precreate` | `create_hcs_store_fast` — build an empty HCS OME-Zarr store (all positions, correct scale/channel metadata) in one pass so parallel writers only fill arrays. |
| `zarr_labels` | Creating and managing label arrays inside zarr stores. |
| `async_zarr_writer` | `AsyncZarrWriter` — background-thread writes so GPU compute overlaps I/O. |
| `anndata_utils` | AnnData validation/inspection for feature-extraction output (reads provenance from `adata.var`). |
| `tiling` | Tile splitting for array partitioning. |

### `cyclops_utils.hpc` — SLURM and resources

| Module | What it provides |
|---|---|
| `slurm_batch_utils` | Batch submission front end: `submit_parallel_jobs`, `wait_for_multiple_job_arrays`, `monitor_slurm_arrays`, `detect_experiments_needing_processing`, `check_step_dependencies_satisfied`, plus `handle_batch_mode_cli` / `handle_single_experiment_cli`. |
| `slurm_utils` | Lower-level job accounting: completion checks, retry-aware monitoring, GPU/memory TRES parsing, job stat tables. |
| `resource_manager` | `get_optimal_workers` — worker count derived from CPU cores, RAM, GPU count and VRAM, respecting SLURM allocation limits; also `get_cpu_resources`, `get_gpu_resources`, `compute_gpu_workers`. |
| `parallel_utils` | `MultiGPUCluster`, `GPUPinnedSpecCluster`, `run_jobs_inproc`, `call_in_spawned_process`. |
| `gpu_utils` | `GPUAssignmentPlugin` for pinning workers to devices. |
| `phase_tracker` | Transparent inner-job tracking: when a launcher step runs under the DAG runner, array waits delegate to the runner's live progress table with no step-function changes. |
| `launcher_result` | `LauncherResult` — signals to the DAG runner that a step submitted inner job arrays. |

### `cyclops_utils.profiling` — instrumentation and notifications

| Module | What it provides |
|---|---|
| `decorators` | `notify_step` — wraps a pipeline step so start/success/failure post to the active notifier thread (with optional artifact attachments), plus `versioned_function` and subtask metric aggregation/summary writers. |
| `slack_notifier` | `ExperimentNotifier` context manager (`step`, `stage`, `success`, `error`, `attach`, `attach_batch`) plus module-level `notify` / `notify_file` / `notify_files`. Without credentials, or without the `slack` extra, every call prints the notification to the terminal instead. |
| `proc_monitor` | `start_monitor` — lightweight CPU/memory/IO sampling of a process tree straight from `/proc`. |

Slack credentials come from `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID`; set
`OPS_SLACK_ENV_FILE` to load them from a dotenv file. The client is built on the
first send, so importing the module touches neither the network nor the disk.
Experiment threads are tracked in `~/.cache/cyclops_utils/slack_threads.json` (honouring
`XDG_CACHE_HOME`, or `OPS_SLACK_THREAD_CACHE` to override) so an orchestrator and
its SLURM workers post into one thread per experiment; point several accounts at one
path to share threads between them.

### `cyclops_utils.analysis` — scoring and visualization

| Module | What it provides |
|---|---|
| `map_scores` | copairs mAP phenotypic scoring: activity (guide level), distinctiveness among active perturbations, CORUM consistency. |
| `map_umap` | `metric_umap` (interactive) and `plot_metric_umap` (batch, saves to file) for mAP metrics on UMAP embeddings. |
| `normalization` | `zscore_normalize`, `df_to_adata` (drops zero-variance features). |
| `pca`, `pca_sweep_plots` | PCA fitting with variance thresholds; per-channel sweep curves, overlays, peak bar charts. |
| `gene_supercategories` | Gene → cell-biology category resolver with four schemes (`chad`, `chad_boosted` ~98% coverage, `reactome_toplevel`, `reactome_cell_biology`). |
| `embedding_discovery` | Classifies experiments by embedding status — fully processed, embeddings only, no embeddings, or no config. Backs the `ops-embedding-status` console script. |
| `embedding_plots` | UMAP / PHATE overlay helpers. |

### `cyclops_utils.ops_mode`

Operational vs. research mode. Operational dual-writes logs to a central root and
writes to real data paths; research keeps logs local and redirects outputs to a
`/rerun/` subdirectory so test runs cannot overwrite production data. Selected by
`OPS_MODE` (inherited into SLURM children) or a `--mode` flag in the runner.

## Tests

```bash
uv run pytest tests/
```

## Ownership and maintenance

**This repository is the result of work done at [biohub San Francisco](https://github.com/czbiohub-sf).**

This repository is owned by the [Leonetti group](https://biohub.org/leonetti/) at [biohub San Francisco](https://github.com/czbiohub-sf).

Maintainers (see also [`.github/CODEOWNERS`](.github/CODEOWNERS)):

- Alexander Hillsley ([@ahillsley](https://github.com/ahillsley))
- Gav Sturm ([@gav-sturm](https://github.com/gav-sturm))

Please open an issue or pull request for questions, bugs, or contributions.

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
