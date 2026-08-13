from pathlib import Path
from iohub import open_ome_zarr
import pandas as pd
import numpy as np
import re
import yaml
import os
from cyclops_utils.paths import BASE_PATH, DRAGONFLY_ROOT, INSTRUMENT_ROOT


class OpsDataset:
    def __init__(self, experiment, config: dict | None = None, method: str = None, slurm_task_config_path: str | Path | None = None):
        self.store_props = {
            "chunk_size": (1, 1, 1, 2048, 2048),
            "5x_scale": [1.0, 1.0, 1.0, 1.3, 1.3],
            "20x_scale": [
                1.0,
                1.0,
                2.0,
                0.65,
                0.65,
            ],  # Updated to match actual 20x metadata
            "sc_crop_size": 64,  # a square in xy
            "tile_size": (2048, 2048),
        }
        self.experiment = experiment

        # Allow override of base directory via environment variable
        base_dir = os.environ.get('OPS_OUTPUT_BASE_DIR', f'{BASE_PATH}')
        self.experiment_path = Path(f"{base_dir}/{experiment}")
        # Allow override of fast_ops base directory (defaults to base_dir)
        fast_base_dir = os.environ.get('OPS_FAST_OUTPUT_BASE_DIR', base_dir)
        self.experiment_path_fast = Path(f"{fast_base_dir}/{experiment}")
        self.iss_tif_dir = Path(f"{INSTRUMENT_ROOT}/{experiment}")
        self.convert_live = self.experiment_path / Path("0-convert/live_imaging")
        self.convert_in_situ = self.experiment_path / Path(
            "0-convert/in_situ_sequencing"
        )
        self.preprocess_live = self.experiment_path / Path("1-preprocess/live_imaging")
        self.preprocess_live_fast = self.experiment_path_fast / Path("1-preprocess/live_imaging")
        self.preprocess_in_situ = self.experiment_path / Path(
            "1-preprocess/in_situ_sequencing"
        )
        self.preprocess_in_situ_fast = self.experiment_path_fast / Path("1-preprocess/in_situ_sequencing")
        self.tracking = self.experiment_path / Path("2-tracking")

        self.results = self.experiment_path / "3-assembly"
        self.results_fast = self.experiment_path_fast / "3-assembly"
        self.results_iss = self.results / "ISS"
        if method == "mine":
            self.results_iss = self.results_iss / "mine"
        elif method == "probabilistic":
            self.results_iss = self.results_iss / "prob"

        self.analysis_path = self.results / "feature_extraction"

        # self.iss_tif_dir =
        self.lc_dragonfly_dir = Path(
            f"{DRAGONFLY_ROOT}/{experiment.split('_')[0].upper()}"
        )
        # Allow override of configs directory via environment variable
        self.configs = Path(os.environ.get('OPS_CONFIGS_DIR', f'{BASE_PATH}/configs'))
        self.codebook = self.configs / "library" / "pool1_design.csv"
        self.gene_index = self.configs / "library" / "twist1k_pool_CERES.csv"
        self.codebook_column_map = None
        self.gene_index_column_map = None
        self.gene_name_output_column = None
        self.iss_secondary_gene_column = None
        self.codebook_round_offset: int = 0  # Slice sgRNA at offset when loading (e.g. 10 for rounds 11-20)
        self.channel_maps = self.configs / "ops_channel_maps.yaml"
        self.marker_seg_params = self.configs / "org_seg_params.yaml"
        self.failed_rounds = self.configs / "ops_failed_rounds.yaml"
        self.library_map = self.configs / "library" / "ops_library_map.yaml"
        self.plasmid_pool_ngs = (
            self.configs / "twist_Q390204_CZ_Biohub_Normalized_Read_Count.csv"
        )
        # Shared generic seed affines (fluor->Phase2D), derived from the median
        # of recent manual registrations. Used to seed/replace the manual
        # point-and-click registration. Override via OPS_AFFINES_DIR.
        self.shared_affines = Path(
            os.environ.get('OPS_AFFINES_DIR', f'{BASE_PATH}/configs/affines')
        )

        self.logfile = self.experiment_path / "function_call_log.yaml"

        self.store_paths = {
            "iss": self.convert_in_situ / "bc_symlink.zarr",
            "iss_drift_corrected": self.convert_in_situ / "bc_drift_corrected.zarr",
            "lc_5x": self.convert_live / "tracking_symlink.zarr",
            "lc_20x": self.convert_live / "phenotyping_transform.zarr",
            "lc_20x_beads": self.convert_live / "20x_beads.zarr",
            "lc_20x_beads_phase": self.convert_live / "20x_beads_phase.zarr",
            "lc_20x_beads_assembled": self.convert_live / "20x_beads_assembled.zarr",
            "iss_stitch": self.preprocess_in_situ / "stitch/bc_stitched.zarr",
            "iss_stitch_registered": self.preprocess_in_situ / "register/bc_stitched_registered.zarr",
            # iss_stitch_registered_v3 still has its own path because
            # register_iss_cycles writes v2; convert_v3 transforms it to v3.
            # When register_iss_cycles becomes v3-native this can collapse
            # to point at iss_stitch_registered.
            "iss_stitch_registered_v3": self.preprocess_in_situ_fast / "register/bc_stitched_registered_v3.zarr",
            "iss_segmentation": self.preprocess_in_situ
            / "segmentation/bc_segmentation.zarr",
            "lc_5x_bf_corrected": self.preprocess_live
            / "reconstruction/tracking_bf_corrected.zarr",
            "lc_5x_bf_corrected": self.preprocess_live
            / "reconstruction/tracking_bf_corrected.zarr",
            "lc_5x_phase": self.preprocess_live 
            / "reconstruction/tracking_phase.zarr",
            # Unified 2D + focus output store (C=2) for 5x tracking
            "lc_5x_phase_focus": self.preprocess_live
            / "reconstruction/tracking_phase_2d.zarr",
            "lc_5x_phase_2d": self.preprocess_live
            / "reconstruction/tracking_phase_2d.zarr",
            # Tilt-optimized reconstruction outputs (from reconstruct_tilt_corrected)
            "lc_5x_phase_3d_optimized": self.preprocess_live
            / "reconstruction/tracking_phase_optimized.zarr",
            "lc_5x_phase_2d_optimized": self.preprocess_live
            / "reconstruction/tracking_phase_2d_optimized.zarr",
            # Stitched tracking 2D recon (per well). The v2 path
            # (tracking_phase_2d_stitched.zarr) and the v3 path
            # (tracking_phase_2d_stitched_v3.zarr) are kept distinct so v2 and
            # v3 outputs never collide on disk. Callers writing v3-native
            # (zarr_version="0.5") should reference the "_v3" key.
            # DEFUNCT v2 — use "lc_5x_phase_2d_stitched_v3".
            # "lc_5x_phase_2d_stitched": self.preprocess_live
            # / "stitch/tracking_phase_2d_stitched.zarr",
            "lc_5x_phase_2d_stitched_v3": self.preprocess_live
            / "stitch/tracking_phase_2d_stitched_v3.zarr",
            "lc_5x_vs_intermediate": self.preprocess_live
            / "virtual_staining/tracking_vs",  # intermediate shards dir (inference output)
            "lc_5x_vs": self.preprocess_live
            / "virtual_staining/tracking_vs.zarr",  # NOTE: using 2D VS for 5x tracking
            "lc_5x_vs_max_proj": self.preprocess_live
            / "virtual_staining/tracking_max_proj.zarr",
            "lc_5x_segmentation": self.preprocess_live
            / "segmentation/tracking_segmentation_stitched.zarr",
            "lc_20x_phase": self.preprocess_live
            / "reconstruction/phenotyping_phase.zarr",
            # Fluorescence 3D reconstruction output (per-tile volumes)
            "lc_20x_fluor_3d": self.preprocess_live
            / "reconstruction/phenotyping_fluor_3d.zarr",
            # Unified 2D + focus output store (C=2). Keep old key for compatibility.
            "lc_20x_phase_focus": self.preprocess_live
            / "reconstruction/phenotyping_phase_2d.zarr",
            "lc_20x_phase_2d": self.preprocess_live
            / "reconstruction/phenotyping_phase_2d.zarr",
            # Tilt-optimized reconstruction outputs (from reconstruct_tilt_corrected)
            "lc_20x_phase_3d_optimized": self.preprocess_live
            / "reconstruction/phenotyping_phase_optimized.zarr",
            "lc_20x_phase_2d_optimized": self.preprocess_live
            / "reconstruction/phenotyping_phase_2d_optimized.zarr",
            # New: reconstructed fluorescence 2D tiles (separate from phase2d)
            "lc_20x_fluor_2d": self.preprocess_live
            / "reconstruction/phenotyping_fluor_2d.zarr",
            # Flatfield-corrected fluorescence 2D tiles
            "lc_20x_fluor_2d_flatfield": self.preprocess_live
            / "reconstruction/phenotyping_fluor_2d_flatfield_corrected.zarr",
            # Registered fluorescence 2D tiles (after applying manual bead-based affine)
            "lc_20x_fluor_2d_registered": self.preprocess_live
            / "reconstruction/phenotyping_fluor_2d_registered.zarr",
            # DEFUNCT v2 — use "pheno_assembled_v3".
            # "pheno_phase_stitched": self.preprocess_live / "stitch/pheno_phase_stitched.zarr",
            "pheno_fluor_stitched": self.preprocess_live
            / "stitch/pheno_fluor_stitched.zarr",
            "lc_20x_vs_intermediate": self.preprocess_live
            / "virtual_staining/phenotyping_vs",  # intermediate shards dir (inference output)
            "lc_20x_vs": self.preprocess_live
            / "virtual_staining/phenotyping_vs.zarr",  # NOTE: using 3D VS for 20x phenotyping
            # Stitched virtual staining (per well)
            "lc_20x_vs_stitched": self.preprocess_live
            / "stitch/phenotyping_vs_stitched.zarr",
            # Unified multi-channel tiles (Phase2D + registered Fluor2D + VS) prior to a single stitch
            "pheno_tiles_unified": self.preprocess_live
            / "stitch/phenotyping_tiles_unified.zarr",  # phenotyping_phase_vs_tiles_unified.zarr
            "lc_20x_vs_max_proj": self.preprocess_live
            / "virtual_staining/phenotyping_max_proj.zarr",
            # Retired: nuclei are segmented at native 20x into the v3
            # `nuclear_seg` label (submit_nuclei_segmentation_jobs), not a
            # standalone 5x store. See royerlab/ops_process#113.
            "lc_20x_segmentation_cells": self.preprocess_live
            / "segmentation/phenotyping_segmentation_cells.zarr",
            "tracking_geff": self.tracking / "tracks.geff",
           
            "pheno_assembled_v3": self.results / "phenotyping_v3.zarr",
            # Raw brightfield z-slices titration pipeline (run_bf_titration_pipeline):
            "bf_slices_assembled": self.results / "bf_slices_assembled.zarr",
            "bf_slices_assembled_v3": self.results / "bf_slices_assembled_v3.zarr",
        }

        self.model_paths = {
            "track_model": f"{BASE_PATH}/models/tracking/2026_03_17_CTC_HSC_01_50k_steps.pt",
            # v2: "track_model": f"{BASE_PATH}/models/tracking/2026_02_12_CTC_50k_steps.pt",
            # v1: "track_model": f"{BASE_PATH}/models/tracking/2025_10_24_09_38_38_job_24296485_whole_CTC.pt",
        }

        # Allow override of experiment config directory and file via environment variables
        exp_config_dir = os.environ.get('OPS_EXP_CONFIG_DIR', str(self.configs / "experiment_configs"))
        exp_config_file = os.environ.get('OPS_EXP_CONFIG_FILE',
                                         str(Path(exp_config_dir) / f"{self.experiment}_config.yaml"))

        self.config_paths = {
            "exp_config_dir": Path(exp_config_dir),
            "exp_config": Path(exp_config_file),
            "env_config": self.configs / "env_config.yaml",
            "iss_stitch": self.preprocess_in_situ / "stitch/stitch_settings.yml",
            "lc_5x_stitch": self.preprocess_live
            / "stitch/tracking_stitch_settings.yml",
            "lc_20x_stitch": self.preprocess_live
            / "stitch/phenotyping_stitch_settings.yml",
            "lc_5x_vs_norm": self.preprocess_live
            / "virtual_staining/tracking_vs_norm.yml",
            "lc_20x_vs_norm": self.preprocess_live
            / "virtual_staining/phenotyping_vs_norm.yml",
            "iss_seg_register": self.tracking / "register.yml",
            "lc_20x_seg_register": self.tracking / "pheno_register.yml",
            # Cell painting registration paths
            "lc_cell_painting1_register": self.tracking / "cell_painting1_register.yml",
            "lc_cell_painting2_register": self.tracking / "cell_painting2_register.yml",
            # Auto-registration paths (generated by auto_register.py)
            "auto_iss_register": self.tracking / "auto_register.yml",
            "auto_pheno_register": self.tracking / "auto_pheno_register.yml",
            "lc_20x_position_list": self.lc_dragonfly_dir
            / f"{experiment.split('_')[0].upper()}_1/pheno_position_list.json",
            "lc_5x_position_list": self.lc_dragonfly_dir
            / f"{experiment.split('_')[0].upper()}_1/tracking_position_list.json",
            "lc_GFP_register": self.results / "lc_GFP_register.yml",
            "lc_mCherry_register": self.results / "lc_mCherry_register.yml",
            "lc_Cy5_register": self.results / "lc_Cy5_register.yml",
            # Generic median seed affines (fluor->Phase2D), shared across experiments
            "lc_GFP_register_seed": self.shared_affines / "lc_GFP_seed_median.yml",
            "lc_mCherry_register_seed": self.shared_affines / "lc_mCherry_seed_median.yml",
            "lc_Cy5_register_seed": self.shared_affines / "lc_Cy5_seed_median.yml",
            "vs_helper": self.configs / "predict_slurm.sh",
            "vs_combine_script": self.configs / "combine_batch.sh",
            # Virtual staining logs and job metadata
            "vs_logs_dir": self.preprocess_live / "virtual_staining/logs",
            "vs_jobs_track": self.preprocess_live
            / "virtual_staining/logs/vs_jobs_track.yaml",
            "vs_jobs_pheno": self.preprocess_live
            / "virtual_staining/logs/vs_jobs_pheno.yaml",
            "lc_5x_phase_recon": self.configs / "phase_config_track.yaml",
            # 2D recon config for 5x tracking
            "lc_5x_phase_recon_2d": self.configs / "phase-3d-to-2d-5x.yml",
            "lc_20x_phase_recon": self.configs / "phase_config_pheno.yaml",
            "lc_20x_phase_recon_2d": self.configs / "phase-3d-to-2d-20x.yml",
            # fluorescence reconstruction config for 20x phenotyping
            "lc_20x_fluor_recon_gfp": self.configs / "recon_fluor_pheno_gfp.yaml",
            "lc_20x_fluor_recon_mCherry": self.configs
            / "recon_fluor_pheno_mCherry.yaml",
            "distortion_corr_params": self.configs / "coefficients.txt",
            "distoration_corr_params_iss": self.configs / "coefficients_iss.txt",
            "lc_5x_vs_config": self.configs / "predict_track.yml",
            "lc_5x_vs_config_2d": self.configs / "predict_track_2d.yml",
            "lc_20x_vs_config": self.configs / "predict_pheno.yml",
            "lc_20x_vs_config_2d": self.configs / "predict_pheno_2d.yml",
            "pheno_assembled_norm": self.results / "pheno_assembled_norm.yml",
            "tf_cache_shared": Path(f"{BASE_PATH}") / "cache" / "phase_reconstruction",
        }

        # Configure slurm_task_config with override support
        # Priority: 1) explicit parameter, 2) config dict, 3) environment variable, 4) default
        if slurm_task_config_path is not None:
            # Explicit parameter takes highest priority
            self.config_paths["slurm_task_config"] = Path(slurm_task_config_path)
        elif config and config.get("slurm_task_config"):
            # Config file override
            self.config_paths["slurm_task_config"] = Path(config["slurm_task_config"])
        elif os.environ.get("OPS_SLURM_TASK_CONFIG"):
            # Environment variable override
            self.config_paths["slurm_task_config"] = Path(os.environ["OPS_SLURM_TASK_CONFIG"])
        else:
            # Default path
            self.config_paths["slurm_task_config"] = self.configs / "slurm_task_config.yaml"

        # Channel map for this dataset (derived from config when provided)
        self.channel_map_data: dict[str, str] = {}
        # Fixed-cell panel channels (CP/4i) to exclude from live-cell phenotyping
        self.fixed_channels: list[str] = []
        # Auto-apply experiment config if provided, or load from disk if available
        try:
            if config is not None:
                self.apply_experiment_config(config)
            else:
                exp_cfg_path = self.config_paths.get("exp_config")
                if exp_cfg_path and Path(exp_cfg_path).exists():
                    with open(exp_cfg_path, "r") as f:
                        file_cfg = yaml.safe_load(f) or {}
                    if isinstance(file_cfg, dict):
                        self.apply_experiment_config(file_cfg)
        except Exception:
            # Fail-soft: leave channel_map_data empty if any issue arises
            self.channel_map_data = {}

        self.result_paths = {
            "spots": self.preprocess_in_situ / "base_calling/detected_points.npy",
            "reads": self.preprocess_in_situ / "base_calling/mine/reads.csv",
            "linked_results": self.results_fast / "linked_pheno_iss.csv",
            "link_metrics": self.results_fast / "link_metrics.csv",
            "all_time_points": self.results / "track_all_time_points.csv",
            "sc_dataset": self.results / "sc_crop.zarr",
            "cell_sizes": self.results / "cell_sizes" / "cell_sizes.csv",
        }

        self.metrics_paths = {
            "iss_cycle_drift": self.results_iss / "iss_cycle_drift.png",
            "stitch_confidence": self.results_iss / "stitch_confidence.png",
            "confluency": self.results_iss / "confluence_heatmap.png",
            "read_accuracy_heatmap": self.results_iss / "read_accuracy_heatmap.png",
            "read_acc_by_round": self.results_iss / "read_accuracy_by_round.png",
            "base_frac_by_round": self.results_iss / "base_fraction_by_round.png",
            "statistics": self.results_iss / "plate_stats.csv",
            "frequency_table": self.results_iss / "frequency_table.csv",
            "timing": self.results_iss / "pipeline_timing.png",
            "hamming_distance": self.results_iss / "hamming_distance.png",
            "percent_cells_with_reads_heatmap": self.results_iss
            / "percent_cells_with_reads_heatmap.png",
            "cell_count_vs_growth_effect": self.results_iss
            / "cell_count_vs_growth_effect.png",
            "statistics_bio_plot": self.results_iss
            / "cell_count_vs_growth_effect_stats.csv",
            "ntc_sgRNA_distrib": self.results_iss / "ntc_sgRNA_distrib.png",
            "statistics_bio_plot_ntc_distrib": self.results_iss
            / "ntc_sgRNA_distrib_stats.csv",
            "cells_per_gene_histogram": self.results_iss
            / "cells_per_gene_histogram.png",
            "top_genes_by_cell_count": self.results_iss / "top_genes_by_cell_count.png",
            "top_guides_by_cell_count": self.results_iss
            / "top_guides_by_cell_count.png",
            "guide_entropy_vs_cell_count": self.results_iss
            / "guide_entropy_vs_cell_count.png",
            # Link-level (post-tracking) histogram plots
            "link_cells_per_gene_histogram": self.results_iss
            / "tracks"
            / "link_cells_per_gene_histogram.png",
            "link_top_genes_by_cell_count": self.results_iss
            / "tracks"
            / "link_top_genes_by_cell_count.png",
            "link_top_guides_by_cell_count": self.results_iss
            / "tracks"
            / "link_top_guides_by_cell_count.png",
            "cell_count_vs_growth_effect_per_well": self.results_iss
            / "cell_count_vs_growth_effect_per_well.png",
            # Imaging quality metrics (method-independent, organized in SNR directory)
            "iss_signal_vs_cycle": self.results_iss / "SNR" / "signal_vs_cycle.png",
            "iss_median_top10pct_vs_cycle": self.results_iss
            / "SNR"
            / "median_top10pct_vs_cycle.png",
            "iss_background_noise_vs_cycle": self.results_iss
            / "SNR"
            / "background_noise_vs_cycle.png",
            "iss_background_mean_vs_cycle": self.results_iss
            / "SNR"
            / "background_mean_vs_cycle.png",
            "iss_snr_vs_cycle": self.results_iss / "SNR" / "snr_vs_cycle.png",
            "iss_sbr_vs_cycle": self.results_iss / "SNR" / "sbr_vs_cycle.png",
            "iss_lld_vs_cycle": self.results_iss / "SNR" / "lld_vs_cycle.png",
            "iss_zprime_vs_cycle": self.results_iss / "SNR" / "zprime_vs_cycle.png",
            "estimated_crosstalk_heatmap": self.results_iss
            / "SNR"
            / "crosstalk_heatmap.png",
            "estimated_crosstalk_matrix": self.results_iss
            / "SNR"
            / "crosstalk_matrix.csv",
            # Base-calling related metrics (keep in results_iss root)
            "confidence_distribution_pooled": self.results_iss
            / "confidence_distribution_pooled.png",
            "read_length_challenge": self.results_iss / "read_length_challenge.png",
            "over_time": self.results_iss / "over_time",
            "tile_focus_heatmap": self.results / "tile_focus_heatmap.png",
            "tile_zoffset_heatmap": self.results / "tile_zoffset_heatmap.png",
            "subtile_focus_heatmap": self.results / "subtile_focus_heatmap.png",
            "subtile_zoffset_heatmap": self.results / "subtile_zoffset_heatmap.png",
            # Debug directories for signal/background analysis
            "iss_metrics_debug": self.results_iss / "SNR" / "debug",
            # SNR heatmap data and visualizations (in SNR directory)
            "snr_per_tile_data": self.results_iss / "SNR" / "per_tile_data.csv",
            "snr_mean_per_round_per_channel": self.results_iss
            / "SNR"
            / "mean_per_round_per_channel.csv",
            "snr_heatmap_overall": self.results_iss / "SNR" / "heatmap_overall.png",
            "snr_heatmap_per_channel": self.results_iss
            / "SNR"
            / "heatmap_per_channel.png",
            "snr_heatmap_per_channel_per_round": self.results_iss
            / "SNR"
            / "heatmap_per_channel_per_round",
            # Cell segmentation size/shape metrics
            "cell_size_summary": self.results_iss / "cell_size_summary.csv",
            "cell_size_distribution": self.results_iss / "cell_size_distribution.png",
        }

    def apply_experiment_config(self, config: dict):
        """
        Apply config to dataset to derive and store dataset-scoped settings.
        Handles channel_map, codebook, gene_index, and column maps.
        """
        try:
            raw_map = config.get("channel_map", {}) or {}
            self.channel_map_data = {
                str(k): v for k, v in raw_map.items() if v is not None
            }
            # Fixed-cell panel channels (CP/4i) — kept for labeling but excluded
            # from live-cell link_phenotyping.
            self.fixed_channels = [str(c) for c in (config.get("fixed_channels") or [])]
        except Exception:
            self.channel_map_data = {}
            self.fixed_channels = []

        # Apply codebook/gene_index overrides from config
        if config.get("codebook"):
            self.codebook = self.configs / "library" / config["codebook"]
        if config.get("gene_index"):
            self.gene_index = self.configs / config["gene_index"]
        if config.get("codebook_column_map"):
            self.codebook_column_map = config["codebook_column_map"]
        if config.get("gene_index_column_map"):
            self.gene_index_column_map = config["gene_index_column_map"]
        if config.get("gene_name_output_column"):
            self.gene_name_output_column = config["gene_name_output_column"]
        if config.get("iss_secondary_gene_column"):
            self.iss_secondary_gene_column = config["iss_secondary_gene_column"]
        if config.get("iss_tif_dir"):
            self.iss_tif_dir = Path(config["iss_tif_dir"])
        if config.get("codebook_round_offset"):
            self.codebook_round_offset = int(config["codebook_round_offset"])
        # Apply tile_size override from stitch params (for non-standard ISS cameras).
        # Only updates tile_size — chunk_size is left at the default so non-ISS
        # stores (track/pheno 2048×2048 tiles) don't get tagged with ISS chunk dims.
        stitch_params = config.get("stack_symlinks_params", {})
        if stitch_params.get("tile_size"):
            ts = stitch_params["tile_size"]
            self.store_props["tile_size"] = tuple(ts)

    def load_codebook(self) -> pd.DataFrame:
        """Load codebook CSV and apply column renaming if configured.

        Returns a DataFrame with standard columns (sgRNA, gene_id) regardless
        of the underlying CSV format.
        """
        print(f"[CODEBOOK] Loading codebook from {self.codebook}")
        if self.codebook_column_map:
            print(f"[CODEBOOK]   column_map: {self.codebook_column_map}")
        df = pd.read_csv(self.codebook)
        if self.codebook_column_map:
            df = df.rename(columns=self.codebook_column_map)
            print(f"[CODEBOOK]   Columns after rename: {list(df.columns)}")
        if self.codebook_round_offset and "sgRNA" in df.columns:
            o = self.codebook_round_offset
            df["sgRNA"] = df["sgRNA"].str[o:o + 10]
            print(f"[CODEBOOK]   Applied round offset {o}: slicing sgRNA to chars [{o}:{o+10}]")
        return df

    def load_gene_index(self) -> pd.DataFrame:
        """Load gene_index CSV and apply column renaming if configured.

        Returns a DataFrame with standard columns (barcode, Gene name) regardless
        of the underlying CSV format.
        """
        print(f"[GENE_INDEX] Loading gene_index from {self.gene_index}")
        if self.gene_index_column_map:
            print(f"[GENE_INDEX]   column_map: {self.gene_index_column_map}")
        df = pd.read_csv(self.gene_index)
        # Strip phantom "Unnamed: N" columns from trailing commas in source CSVs
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        if self.gene_index_column_map:
            df = df.rename(columns=self.gene_index_column_map)
            print(f"[GENE_INDEX]   Columns after rename: {list(df.columns)}")
        return df

    def _resolve_fast_partition_path(self, relative_path: str | Path, quiet: bool = False) -> Path:
        """
        Resolve a path, preferring the fast partition if it exists.

        Args:
            relative_path: Path relative to experiment root (e.g., "3-assembly/phenotyping_v3.zarr")
            quiet: If True, suppress print statements

        Checks for the path in:
        1. $OPS_FAST_OUTPUT_BASE_DIR/{experiment}/{relative_path} (fast partition)
        2. $OPS_OUTPUT_BASE_DIR/{experiment}/{relative_path} (standard partition)

        Both default to $OPS_BASE_PATH when unset.

        Returns the fast partition path if it exists, otherwise returns the standard partition path.
        """
        relative_path = Path(relative_path)
        fast_path = self.experiment_path_fast / relative_path
        standard_path = self.experiment_path / relative_path

        # Skip print for dummy datasets used only for config path lookups
        is_real_experiment = self.experiment and self.experiment not in ("dummy", "")
        should_print = is_real_experiment and not quiet

        # Check if the fast partition path exists
        if fast_path.exists():
            if should_print:
                print(f"[fast-partition] Using: {fast_path}")
            return fast_path

        # Fall back to standard partition
        if should_print:
            print(f"[standard-partition] Using: {standard_path}")
        return standard_path

    def append_well(self, name: str, well: str):
        """
        Build a per-well file path by prefixing the target filename with the well token.
        Accepts flexible well formats, e.g. "A/1", "A1", "AA4", "AA/4", "A-1".
        The final prefix is ROWLETTERS + COLNUM followed by underscore, e.g. "AA4_".
        """
        import re

        # Normalize to ROW + COL without separators
        # Try common split formats first
        normalized = None
        if "/" in well:
            parts = [p for p in well.split("/") if p]
            if len(parts) >= 2:
                row, col = parts[0], parts[1]
                normalized = f"{row}{col}"
        elif "-" in well:
            parts = [p for p in well.split("-") if p]
            if len(parts) >= 2:
                row, col = parts[0], parts[1]
                normalized = f"{row}{col}"

        if normalized is None:
            # Fallback: parse with regex, letters then digits at the end
            m = re.match(r"^([A-Za-z]+)(\d+)$", well)
            if m:
                normalized = f"{m.group(1)}{m.group(2)}"
            else:
                # As a last resort, strip non-alnum and use as-is
                normalized = re.sub(r"[^A-Za-z0-9]", "", well)

        well_prefix = f"{normalized}_"

        if name in self.result_paths:
            path = self.result_paths[name]
        elif name in self.config_paths:
            path = self.config_paths[name]
        elif name in self.metrics_paths:
            path = self.metrics_paths[name]
        elif name in self.store_paths:
            path = self.store_paths[name]
        else:
            raise ValueError(f"{name} not in result_paths or config_paths")

        return path.parent / f"{well_prefix}{path.name}"

    def get_output_files_for_step(self, log_key, config):
        """
        Returns a list of expected output files for a given pipeline step key.
        This allows for file-based completion checking instead of relying on the log file.
        Returns None if no file check is defined for the step, which triggers a
        fallback to the log-based checking mechanism.
        """
        # Handle per-well steps first
        if log_key.startswith("track_wells"):
            if log_key == "track_wells":
                # track_wells processes all wells at once
                wells = config.get("wells_to_process", []) or self.infer_wells()
                return [self.append_well("tracking_geff", w) for w in wells]
            else:
                # Handle per-well tracking if called with specific well suffix
                well = log_key.replace("track_wells_", "").replace("_", "/")
                return [self.append_well("tracking_geff", well)]
        if log_key.startswith("create_dataset_"):
            well = log_key.replace("create_dataset_", "").replace("_", "/")
            return [self.append_well("sc_dataset", well)]

        # Special handling for build_pyramids - check for pyramid level 1
        if log_key == "build_pyramids":
            return self._get_pyramid_check_paths()

        wells = config.get("wells_to_process", []) or self.infer_wells()

        # The keys here MUST match the log_key generated by PipelineRunner.
        # It is func.__qualname__.replace('.', '_') with optional _<process> or _<well>

        # Dynamically determine outputs for ISS conversion based on config or inferred wells
        convert_iss_outputs = []
        wells_cfg = (
            config.get("wells_to_process") if isinstance(config, dict) else None
        ) or []

        def _to_well_token(w: str) -> str:
            try:
                import re as _re

                if "/" in str(w):
                    parts = [p for p in str(w).split("/") if p]
                    if len(parts) >= 2:
                        return f"{parts[0]}{parts[1]}"
                m = _re.match(r"^([A-Za-z]+)\s*[-_/]?\s*(\d+)$", str(w))
                if m:
                    return f"{m.group(1)}{m.group(2)}"
                return _re.sub(r"[^A-Za-z0-9]", "", str(w))
            except Exception:
                return str(w)

        if wells_cfg:
            well_tokens = [_to_well_token(w) for w in wells_cfg]
        else:
            inferred = self.infer_wells()  # like 'A/1'
            well_tokens = [_to_well_token(w) for w in inferred]

        for well_token in well_tokens:
            matches = sorted(self.convert_in_situ.glob(f"{well_token}_*.zarr"))
            if matches:
                convert_iss_outputs.append(Path(matches[0]))
            else:
                convert_iss_outputs.append(
                    self.convert_in_situ / f"{well_token}_MISSING.zarr"
                )

        def _calibration_model_paths(phase_store: Path, process: str) -> list:
            from cyclops_utils.data.filesystem import get_experiment_wells
            tilt_base = phase_store.parent / "tilt_calibration" / process
            wells = get_experiment_wells(self.experiment, prefix_only=True)
            if not wells:
                return [tilt_base]
            return [
                tilt_base / w.replace("/", "_") / "model.yaml"
                for w in wells
            ]

        step_outputs = {
            # --- ISS Processing ---
            "iss_snr_bimodal": [self.results_iss / "SNR" / "snr_heatmap_overall_bimodal.png"],
            "convert_iss": convert_iss_outputs,
            # The stack_symlinks step creates the assembled ISS zarr store at bc_symlink.zarr
            # so we can use that as the existence check.
            "stack_symlinks": [self.store_paths["iss"]],
            "correct_cycle_drift": [self.store_paths["iss_drift_corrected"]],
            # ISS distortion correction outputs a new corrected store
            # "correct_distortion_iss": [self.store_paths["iss_distortion_corrected"]],
            "estimate_stitch_parameters_iss": [self.config_paths["iss_stitch"]],
            "segment_and_stitch_iss": [self.store_paths["iss_segmentation"]],
            "estimate_and_stitch_iss": [self.store_paths["iss_stitch"]],
            # register_iss_cycles runs with skip_apply_transforms=True under the
            # merge pipeline, so it no longer writes bc_stitched_registered.zarr.
            # Check the per-well anchor YAML written by step 2 instead.
            "register_iss_cycles": [
                self.preprocess_in_situ / "register" / "transforms" / _to_well_token(w) / "nucleus_to_round0.yml"
                for w in wells if w
            ],
            # merge_spots_base_calling fuses warp + detect_spots + base_calling
            # per well; the final-stage output is the per-well reads.csv.
            "merge_spots_base_calling": [
                OpsDataset(self.experiment, method="mine").append_well("reads", w)
                for w in wells if w
            ],
            # convert_iss_to_v3 runs right after merge: converts the ISS
            # registered store to v3 and async-deletes the v2 source. The v3
            # store is the completion signal.
            "convert_iss_to_v3": [self.store_paths["iss_stitch_registered_v3"]],
            # optimize_failed_rounds (before get_metrics): per-well failed-round
            # optimization. Writes a decision report under ISS/<method>/failed_rounds/
            # (the completion signal) and updates the experiment config +
            # ops_failed_rounds.yaml that get_metrics then reads.
            "optimize_failed_rounds": [
                OpsDataset(self.experiment, method="mine").results_iss
                / "failed_rounds" / "optimization_report.txt"
            ],
            "get_metrics": [OpsDataset(self.experiment, method="mine").metrics_paths["statistics"]],
            # recompute_metrics re-runs get_metrics with force=True; same outputs
            "recompute_metrics": [OpsDataset(self.experiment, method="mine").metrics_paths["statistics"]],
            # --- Raw conversion (feeds tracking + phenotyping chains) ---
            "convert_raw": [
                self.experiment_path_fast / "0-convert" / "live_imaging" / "raw_convert"
            ],
            # --- Live-cell Processing ---
            "link_phenotyping": [self.store_paths["lc_20x"]],
            "link_tracking": [self.store_paths["lc_5x"]],
            "correct_distortion": [self.store_paths["lc_5x_bf_corrected"]],
            "reconstruct_track": [self.store_paths["lc_5x_phase"]],
            # Tilt-optimized reconstruction (replaces reconstruct_track-2d)
            "calibrate_tilt_track": _calibration_model_paths(
                self.store_paths["lc_5x_phase"], "track"
            ),
            "reconstruct_tilt_corrected_track": [
                self.store_paths["lc_5x_phase_3d_optimized"],
                self.store_paths["lc_5x_phase_2d_optimized"],
            ],
            "estimate_stitch_parameters_track": [self.config_paths["lc_5x_stitch"]],
            "segment_and_stitch_track": [self.store_paths["lc_5x_segmentation"]],
            "estimate_and_stitch_track-2d": [
                self.store_paths["lc_5x_phase_2d_stitched_v3"]
            ],
            "reconstruct_pheno": [self.store_paths["lc_20x_phase"]],
            # Tilt-optimized reconstruction (replaces reconstruct_pheno-2d)
            "calibrate_tilt_pheno": _calibration_model_paths(
                self.store_paths["lc_20x_phase"], "pheno"
            ),
            "reconstruct_tilt_corrected_pheno": [
                self.store_paths["lc_20x_phase_3d_optimized"],
                self.store_paths["lc_20x_phase_2d_optimized"],
            ],
            "estimate_and_stitch_track-2d": [
                self.store_paths["lc_5x_phase_2d_stitched_v3"]
            ],
            # --- Segmentation ---
            "create_max_projection_lc_20x": [self.store_paths["lc_20x_vs_max_proj"]],
            # "estimate_stitch_parameters_track": [self.config_paths["lc_5x_stitch"]],
            "estimate_stitch_parameters_pheno": [self.config_paths["lc_20x_stitch"]],
            # New: fluorescence Z projection (sum) into fluor 2D tiles
            "create_max_projection_lc_20x_fluor": [self.store_paths["lc_20x_fluor_2d"]],
            # Flatfield correction for fluorescence tiles
            "correct_flatfield_fluor": [self.store_paths["lc_20x_fluor_2d_flatfield"]],
            # Unified tiles preparation before final stitch (legacy)
            # --- Virtual Staining (granular steps) ---
            "virtual_staining_preprocess_track": [self.config_paths["lc_5x_vs_norm"]],
            "virtual_staining_inference_track": [self.store_paths["lc_5x_vs_intermediate"]],
            "virtual_staining_combine_only_track": [self.store_paths["lc_5x_vs"]],
            "virtual_staining_preprocess_pheno": [self.config_paths["lc_20x_vs_norm"]],
            "virtual_staining_inference_pheno": [self.store_paths["lc_20x_vs_intermediate"]],
            "virtual_staining_combine_only_pheno": [self.store_paths["lc_20x_vs"]],
            "segment_and_stitch_track": [self.store_paths["lc_5x_segmentation"]],
            "submit_nuclei_segmentation_jobs": self._get_seg_label_check_paths("nuclear_seg", wells),
            "segment_and_stitch_pheno_cells": [
                self.store_paths["lc_20x_segmentation_cells"]
            ],
            # Channel registration is complete when every fluor channel has its
            # lc_<ch>_register.yml affine (auto-written, or pre-existing manual).
            "submit_channel_registration_jobs": self._fluor_register_ymls(config),
            # "register_stitched_fluor_to_phase": [self.store_paths["pheno_assembled"]],
            "prepare_unified_pheno_tiles": [self.store_paths["pheno_tiles_unified"]],
            # Final unified stitch should materialize the assembled phenotyping store
            "estimate_and_stitch_pheno-2d": [self.store_paths["pheno_assembled_v3"]],
            # Optional ViSCy normalization config
            "viscy_normalize": [self.config_paths["pheno_assembled_norm"]],
            "build_pyramids": self._get_pyramid_check_paths(),
            # build_pyramids is handled dynamically in get_output_files_for_step()
            # --- Cell Segmentation ---
            # Cell seg writes cell_seg labels to the v3 store (per-position)
            # Check for actual label group existence, not just the zarr store
            "submit_cell_segmentation_jobs": self._get_seg_label_check_paths("cell_seg", wells),
            
            # --- Registration ---
            "submit_registration_jobs": (
                [self.append_well("auto_iss_register", w) for w in wells if w]
                + (
                    []
                    if config.get("auto_register_params", {}).get("skip_track", False)
                    else [self.append_well("auto_pheno_register", w) for w in wells if w]
                )
            ),
            # --- Tracking ---
            "submit_tracking_jobs": [self.append_well("tracking_geff", w) for w in wells if w],
            "link_calls_tracks": [
                self.append_well("linked_results", w) for w in wells if w
            ],
            # --- Fix v3 stores (audit + fix all missing components) ---
            # No fixed output path — falls back to function_call_log.yaml
            "fix_v3_stores": None,

        }

        if log_key in ("create_max_projection_lc_20x_fluor", "correct_flatfield_fluor"):
            native = step_outputs.get(log_key) or []
            if not all(Path(p).exists() for p in native):
                fluor_reg = self.store_paths.get("lc_20x_fluor_2d_registered")
                if fluor_reg and Path(fluor_reg).exists():
                    return [fluor_reg]
            return native

        return step_outputs.get(log_key)

    def get_all_step_keys(self) -> list[str]:
        """
        Return canonical ordered step keys understood by get_output_files_for_step.

        Centralizing the list here avoids duplication in the runner or orchestrator.
        """
        return [
            # --- ISS Processing ---
            "convert_iss",
            "stack_symlinks",
            "iss_snr_bimodal",
            "correct_cycle_drift",
            # "correct_distortion_iss",
            "estimate_stitch_parameters_iss",
            "segment_and_stitch_iss",
            "estimate_and_stitch_iss",
            "register_iss_cycles",
            "merge_spots_base_calling",
            "convert_iss_to_v3",
            "optimize_failed_rounds",
            "get_metrics",
            # --- Raw conversion (feeds tracking + phenotyping chains) ---
            "convert_raw",
            # --- Live-cell Processing ---
            "link_phenotyping",
            "create_max_projection_lc_20x_fluor",
            "correct_flatfield_fluor",
            "link_tracking",
            "correct_distortion",
            "reconstruct_track",
            "calibrate_tilt_track",
            "reconstruct_tilt_corrected_track",
            "virtual_staining_preprocess_track",
            "virtual_staining_inference_track",
            "virtual_staining_combine_only_track",
            "estimate_stitch_parameters_track",
            "segment_and_stitch_track",
            "estimate_and_stitch_track-2d",
            "reconstruct_pheno",
            "calibrate_tilt_pheno",
            "reconstruct_tilt_corrected_pheno",
            # --- Virtual Staining (Pheno / 20x 3D) ---
            "virtual_staining_preprocess_pheno",
            "virtual_staining_inference_pheno",
            "virtual_staining_combine_only_pheno",
            # VS max projection for pheno tiles
            "create_max_projection_lc_20x",
            # --- Stitching & Segmentation ---
            "estimate_stitch_parameters_pheno",
            # "segment_and_stitch_pheno_cells",
            # Automatic fluor->Phase2D channel registration (before review checkpoint)
            "submit_channel_registration_jobs",
            # Prepare unified tiles (phase+VS) then separate stitch + registration
            "prepare_unified_pheno_tiles",
            "estimate_and_stitch_pheno-2d",
            # assemble
            # Optional normalization
            "viscy_normalize",
            "build_pyramids",
            # --- Cell Segmentation ---
            "submit_cell_segmentation_jobs",
            # --- Nuclei Segmentation (native 20x) ---
            "submit_nuclei_segmentation_jobs",
            # --- Registration ---
            "submit_registration_jobs",
            # --- Tracking ---
            # "track_wells",
            "submit_tracking_jobs",

            # --- Link Calls to Tracks ---
            "link_calls_tracks",
            # --- Fix v3 stores ---
            "fix_v3_stores",
            # --- Final QC Metrics ---
            "recompute_metrics",

            # "create_dataset",
        ]

    def _get_pyramid_check_paths(self):
        """
        Return paths to check for pyramid completion.

        For each store, check for pyramid level 1 at: store/A/1/0/1
        build_pyramids runs with use_v3_stores=True, so it builds the
        v3 stores: pheno/track stitch v3-native and ISS is converted to v3 by
        convert_iss_to_v3 right after merge. The v2 stores no longer exist.
        """
        check_paths = []
        # Check the v3 stores that build_pyramids builds in this step
        store_keys = [
            "pheno_assembled_v3",
            "lc_5x_phase_2d_stitched_v3",
            "iss_stitch_registered_v3",  # Use registered ISS store, not raw stitched
        ]

        for key in store_keys:
            store = self.store_paths.get(key)
            if not store:
                continue

            # Always include the pyramid path — if the store is missing,
            # the path won't exist and the step correctly shows incomplete.
            pyramid_level_1 = store / "A" / "1" / "0" / "1"
            check_paths.append(pyramid_level_1)

        return check_paths if check_paths else None

    def _get_first_position_labels_path(self) -> Path | None:
        """Get path to labels/ in first position of pheno_assembled_v3."""
        store = self.store_paths.get("pheno_assembled_v3")
        if not store or not store.exists():
            return None
        # Find first position: row/col/fov
        for row in sorted(store.iterdir()):
            if row.is_dir() and row.name.isalpha():
                for col in sorted(row.iterdir()):
                    if col.is_dir() and col.name.isdigit():
                        for fov in sorted(col.iterdir()):
                            if fov.is_dir() and fov.name.isdigit():
                                return fov / "labels"
        return None

    def _get_seg_label_check_paths(self, label_name: str, wells) -> list[Path] | None:
        """Per-FOV `labels/<label_name>` paths across all wells — complete only
        when every position has the label (store root is created upstream)."""
        store = self.store_paths.get("pheno_assembled_v3")
        if store is None:
            return None

        paths: list[Path] = []
        for w in wells:
            parts = [p for p in str(w).split("/") if p]
            if len(parts) < 2:
                continue
            well_dir = store / parts[0] / parts[1]
            fovs = sorted(
                d.name for d in well_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ) if well_dir.is_dir() else []
            for fov in (fovs or ["0"]):
                paths.append(well_dir / fov / "labels" / label_name)

        return paths or None

    def _fluor_register_ymls(self, config) -> list[Path] | None:
        """lc_<ch>_register.yml affine path for each fluor channel in the config —
        the completion signal for channel registration (auto or manual). The
        fluorophore (GFP/mCherry/Cy5) may be the channel_map key (e.g.
        {'GFP': 'cis-Golgi...'}) or the value, so match either."""
        label_to_stem = {"gfp": "lc_GFP_register", "mcherry": "lc_mCherry_register",
                         "cy5": "lc_Cy5_register"}
        ymls: list[Path] = []
        for k, v in (config.get("channel_map") or {}).items():
            for cand in (k, v):
                stem = label_to_stem.get(str(cand).strip().lower()) if cand is not None else None
                if stem:
                    if self.config_paths[stem] not in ymls:
                        ymls.append(self.config_paths[stem])
                    break
        return ymls or None


    def _get_organelle_seg_check_paths(self) -> list[Path] | None:
        """Check for organelle seg labels (phase2d_tubular_seg) in first position."""
        labels_path = self._get_first_position_labels_path()
        return [labels_path / "phase2d_tubular_seg"] if labels_path else None

    def _get_iss_overlay_check_paths(self) -> list[Path] | None:
        """Check for ISS overlay labels (iss_gene_image, iss_guide_image) in first position."""
        labels_path = self._get_first_position_labels_path()
        if labels_path:
            return [labels_path / "iss_gene_image", labels_path / "iss_guide_image"]
        return None

    def infer_wells(self):
        """
        Infer wells present for this experiment from existing filesystem outputs.
        Returns a list of wells in the canonical form used by the pipeline (e.g., "A/1").
        """
        well_tokens = set()

        # 1) Per-round ISS zarrs created during convert_iss (A1_*.zarr, etc.)
        for zarr_path in self.convert_in_situ.glob("*_*.zarr"):
            token = zarr_path.name.split("_", 1)[0]
            if re.match(r"^[A-Za-z]+\d+$", token):
                well_tokens.add(token)

        # 2) ISS base_calling artifacts
        base_calling_dir = self.preprocess_in_situ / "base_calling"
        for pattern in ["*_reads.csv", "*_detected_points.npy"]:
            for f in base_calling_dir.glob(pattern):
                token = f.name.split("_", 1)[0]
                if re.match(r"^[A-Za-z]+\d+$", token):
                    well_tokens.add(token)

        # 3) Tracking outputs
        for f in self.tracking.glob("*_tracks.csv"):
            token = f.name.split("_", 1)[0]
            if re.match(r"^[A-Za-z]+\d+$", token):
                well_tokens.add(token)

        # 4) Linked results
        for f in self.results.glob("*_linked_pheno_iss.csv"):
            token = f.name.split("_", 1)[0]
            if re.match(r"^[A-Za-z]+\d+$", token):
                well_tokens.add(token)

        # Convert tokens like "A1" to canonical form "A/1"
        # Convert tokens like "AA4" to canonical form "AA/4"
        wells = []
        for t in sorted(well_tokens):
            m = re.match(r"^([A-Za-z]+)(\d+)$", t)
            if m:
                wells.append(f"{m.group(1)}/{m.group(2)}")
            else:
                wells.append(t)
        return wells

    def check_exists(self, paths):
        existence = [path.exists() for path in paths.values()]
        existence_df = pd.DataFrame(
            existence, index=self.paths.keys(), columns=["exists"]
        )
        return existence_df

    @staticmethod
    def get_shape(ds):
        first_pos = next(ds.positions())[0]
        shape = ds[first_pos].data.shape
        return shape

    def get_shapes(self):
        for path in self.store_paths.values():
            ds = open_ome_zarr(path)
            shape = self.get_shape(ds)
            print(shape)

    def get_dtype(self):
        dtypes = []
        for path in self.store_paths.values():
            if path.exists():
                ds = open_ome_zarr(path)
                first_pos = next(ds.positions())[0]
                dtype = ds[first_pos].data.dtype
                dtypes.append(dtype)
            else:
                dtypes.append("N/A")
        return pd.DataFrame(dtypes, index=self.paths.keys(), columns=["dtype"])

    def contains_data(self):
        conatins_data_list = []
        for key, path in self.store_paths.items():
            try:
                ds = open_ome_zarr(path)
                first_pos = next(ds.positions())[0]
                shape = np.asarray(ds[first_pos].data.shape).astype(np.uint16)
                xy_middle = shape[-2:] // 2
                data_sample = ds[first_pos].data[
                    :,
                    :,
                    :,
                    xy_middle[0] - 10 : xy_middle[0] + 10,
                    xy_middle[1] - 10 : xy_middle[1] + 10,
                ]

                if np.all(data_sample) == 0:
                    print(f"{key} contains all 0s")
                    conatins_data_list.append(False)
                else:
                    print(f"{key} contains data")
                    conatins_data_list.append(True)
            except:
                conatins_data_list.append(False)

        return pd.DataFrame(
            conatins_data_list, index=self.paths.keys(), columns=["contains_data"]
        )

    def check_stores(self):
        exists = self.check_exists(self.store_paths)
        dtypes = self.get_dtype()
        contains_data = self.contains_data()
        return exists.join(dtypes).join(contains_data)
    
    @property
    def morphology_path_v3(self) -> Path:
        """
        Path to the v3 phenotyping zarr store containing morphology data.
        This is used for representative cell visualization and other morphology-based analyses.
        """
        return self.store_paths["pheno_assembled_v3"]
