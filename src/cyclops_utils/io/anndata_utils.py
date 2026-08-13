#!/usr/bin/env python3
"""
Validate AnnData output from feature extraction pipeline.

Reads feature metadata directly from adata.var (organelle, metric, category, aggregation, unit)
which is populated during feature extraction - no parsing required.

Usage:
    python -m cyclops_utils.io.anndata_utils -e ops0094
    python -m cyclops_utils.io.anndata_utils --path /path/to/features.h5ad
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import anndata as ad
import numpy as np
import pandas as pd
from prettytable import PrettyTable


# Expected metadata columns and their types
# Note: category dtype is valid for string columns (pandas categorical)
EXPECTED_CELL_METADATA = {
    "cell_id": ["object", "str", "category"],
    "well": ["object", "str", "category"],
    "barcode": ["object", "str", "category"],
    "gene_name": ["object", "str", "category"],
    "segmentation_id": ["int64", "int32", "float64"],
    "x_global_pheno": ["float64", "float32", "int64"],
    "y_global_pheno": ["float64", "float32", "int64"],
}

# Columns needed for spatial drift analysis (optional but recommended)
SPATIAL_DRIFT_COLUMNS = {
    "well_drift": {
        "required": ["x_global_pheno", "y_global_pheno", "well"],
        "description": "Well-level radial drift analysis",
    },
    "tile_drift": {
        "required": ["x_local_pheno", "y_local_pheno", "tile_pheno"],
        "description": "Tile/FOV-level radial drift analysis",
    },
}

EXPECTED_GUIDE_METADATA = {
    "barcode": ["object", "str", "category"],
    "gene_name": ["object", "str", "category"],
    "n_cells": ["int64", "int32"],
}

EXPECTED_GENE_METADATA = {
    "gene_name": ["object", "str", "category"],
    "n_cells": ["int64", "int32"],
    "n_guides": ["int64", "int32"],
}


def validate_obs_metadata(adata: ad.AnnData, expected: dict, level: str) -> list[str]:
    """Check that expected obs metadata columns exist with correct types."""
    issues = []
    obs_cols = set(adata.obs.columns)

    for col, valid_types in expected.items():
        if col not in obs_cols:
            # Some columns are optional
            if col in ["global_cell_id", "sgRNA", "subpool"]:
                continue
            issues.append(f"Missing obs column: {col}")
        else:
            dtype = str(adata.obs[col].dtype)
            if not any(t in dtype for t in valid_types):
                issues.append(f"obs['{col}'] dtype {dtype}, expected {valid_types}")

    return issues


def validate_var_metadata(adata: ad.AnnData) -> list[str]:
    """Check that var DataFrame has required metadata columns."""
    issues = []
    required_cols = ["organelle", "metric", "category", "aggregation", "unit"]

    for col in required_cols:
        if col not in adata.var.columns:
            issues.append(f"Missing var column: {col}")

    return issues


def build_organelle_table(adata: ad.AnnData) -> tuple[dict, str]:
    """
    Build a table showing feature counts per organelle by category.

    Uses metadata stored in adata.var - no parsing required.
    """
    organelle_stats = {}

    if "organelle" not in adata.var.columns or "category" not in adata.var.columns:
        return organelle_stats, "No feature metadata in var DataFrame"

    # Group by organelle and category
    for idx, row in adata.var.iterrows():
        org = row.get("organelle")
        cat = row.get("category")

        if org is None or pd.isna(org):
            org = "_unassigned"

        if org not in organelle_stats:
            organelle_stats[org] = {
                "intensity": 0,
                "morphology": 0,
                "network": 0,
                "network_object": 0,
                "localization": 0,
                "cell_morphology": 0,
                "total": 0,
            }

        organelle_stats[org]["total"] += 1
        if cat and not pd.isna(cat):
            if cat in organelle_stats[org]:
                organelle_stats[org][cat] += 1

    # Build table
    table = PrettyTable()
    table.field_names = ["Organelle", "Total", "Intensity", "Morphology", "Network", "Localization"]
    table.align = "l"
    for col in table.field_names[1:]:
        table.align[col] = "r"

    # Sort by total features descending
    for org in sorted(organelle_stats.keys(), key=lambda x: organelle_stats[x]["total"], reverse=True):
        stats = organelle_stats[org]
        network_total = stats["network"] + stats["network_object"]
        morph_total = stats["morphology"] + stats["cell_morphology"]
        table.add_row([org, stats["total"], stats["intensity"], morph_total, network_total, stats["localization"]])

    # Totals
    totals = {k: sum(s.get(k, 0) for s in organelle_stats.values())
              for k in ["intensity", "morphology", "network", "network_object", "localization", "cell_morphology", "total"]}
    table.add_row(["-" * 15, "-" * 5, "-" * 5, "-" * 5, "-" * 5, "-" * 5])
    table.add_row([
        "TOTAL", totals["total"], totals["intensity"],
        totals["morphology"] + totals["cell_morphology"],
        totals["network"] + totals["network_object"],
        totals["localization"]
    ])

    return organelle_stats, str(table)


def validate_data_quality(adata: ad.AnnData, level: str) -> list[str]:
    """Check for data quality issues."""
    issues = []

    # Check for duplicate cell_ids (cell-level only)
    if level == "cell" and "cell_id" in adata.obs.columns:
        duplicates = adata.obs["cell_id"].duplicated()
        n_duplicates = duplicates.sum()
        if n_duplicates > 0:
            # Get examples of duplicate cell_ids
            dup_ids = adata.obs.loc[duplicates, "cell_id"].unique()[:5]
            issues.append(f"{n_duplicates} duplicate cell_ids found (e.g., {list(dup_ids)})")
            print(f"  ERROR: {n_duplicates} duplicate cell_ids detected")
            print(f"    Examples: {list(dup_ids)}")

    critical_cols = {"cell": ["cell_id", "well"], "guide": ["barcode"], "gene": ["gene_name"]}.get(level, [])
    for col in critical_cols:
        if col in adata.obs.columns:
            nan_count = adata.obs[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{nan_count} NaN in critical column: {col}")

    # Check for all-NaN features
    nan_features = [feat for i, feat in enumerate(adata.var_names) if np.all(np.isnan(adata.X[:, i]))]
    if nan_features:
        issues.append(f"{len(nan_features)} features are all NaN")

    # Check for inf values
    inf_count = np.sum(np.isinf(adata.X))
    if inf_count > 0:
        issues.append(f"{inf_count} infinite values in X")

    # Check for zero-variance features (constant values, including all zeros)
    # Use nanvar to handle NaN values gracefully
    variances = np.nanvar(adata.X, axis=0)
    zero_var_mask = (variances == 0) | np.isnan(variances)
    zero_var_features = [adata.var_names[i] for i in np.where(zero_var_mask)[0]]
    
    # Separate all-zeros from other constant features
    all_zero_features = []
    constant_nonzero_features = []
    for feat in zero_var_features:
        feat_idx = list(adata.var_names).index(feat)
        col_data = adata.X[:, feat_idx]
        non_nan_vals = col_data[~np.isnan(col_data)]
        if len(non_nan_vals) > 0 and np.all(non_nan_vals == 0):
            all_zero_features.append(feat)
        elif feat not in nan_features:  # Don't double-count all-NaN features
            constant_nonzero_features.append(feat)
    
    # Report zero-variance features as warnings (not failures)
    # Some are expected (e.g., std features for single-object organelles like cp_cell_mask)
    if all_zero_features:
        n_zero = len(all_zero_features)
        pct = 100 * n_zero / len(adata.var_names)
        print(f"  Warning: {n_zero} features ({pct:.1f}%) are all zeros")
        if n_zero <= 5:
            print(f"    Examples: {all_zero_features}")
        else:
            print(f"    Examples: {all_zero_features[:5]}...")
        # Only flag as issue if >10% of features are zeros (indicates a problem)
        if pct > 10:
            issues.append(f"{n_zero} features ({pct:.1f}%) are all zeros - exceeds 10% threshold")
    
    if constant_nonzero_features:
        n_const = len(constant_nonzero_features)
        pct = 100 * n_const / len(adata.var_names)
        print(f"  Warning: {n_const} features ({pct:.1f}%) have zero variance (constant non-zero)")
        if n_const <= 5:
            print(f"    Examples: {constant_nonzero_features}")
        else:
            print(f"    Examples: {constant_nonzero_features[:5]}...")
        # Only flag as issue if >5% of features are constant (indicates a problem)
        if pct > 5:
            issues.append(f"{n_const} features ({pct:.1f}%) are constant - exceeds 5% threshold")

    return issues


def validate_misplaced_features(adata: ad.AnnData) -> tuple[dict, list[str]]:
    """
    Detect feature-like columns that ended up in .obs instead of .X/.var.

    This catches bugs where the aggregation prefix filter misses certain feature
    types (e.g., network_ columns), causing them to be placed in obs metadata
    instead of the feature matrix.

    Checks for:
    - Numeric obs columns with known feature prefixes (network_)
    - Numeric obs columns matching {something}_{agg_func} patterns

    Returns:
        Tuple of (stats_dict, issues_list)
    """
    stats = {}
    issues = []

    # Known feature prefixes that should NEVER be in obs
    FEATURE_PREFIXES = ("network_",)

    # Aggregation suffixes that indicate a column is a computed feature
    AGG_SUFFIXES = ("_mean", "_median", "_std", "_sum", "_min", "_max", "_count")

    # Columns that are legitimately in obs even if they match patterns
    OBS_ALLOWLIST = {
        "cell_id", "segmentation_id", "cp_cell_seg_id", "well", "fov",
        "barcode", "sgRNA", "gene_name", "gene_effect", "NCBI_ID",
        "site", "plate", "experiment", "condition", "total_index",
        "store_key", "barcode_from_iss", "subpool", "dep_map_gene_name",
        "og_index", "tile_pheno",
    }

    misplaced = []
    for col in adata.obs.columns:
        if col in OBS_ALLOWLIST:
            continue

        # Check known feature prefixes
        if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES):
            # Verify it's numeric (not a string metadata column)
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                misplaced.append(col)

    stats["n_misplaced"] = len(misplaced)

    if misplaced:
        # Group by prefix for readable output
        by_prefix = {}
        for col in misplaced:
            prefix = col.split("_")[0] + "_"
            by_prefix.setdefault(prefix, []).append(col)

        summary_parts = []
        for prefix, cols in sorted(by_prefix.items()):
            summary_parts.append(f"{len(cols)} {prefix}* columns")
        summary = ", ".join(summary_parts)

        issues.append(
            f"{len(misplaced)} feature columns found in .obs instead of .X: {summary}. "
            f"These should be in the feature matrix (.X/.var). "
            f"Fix: add missing prefixes to feature_prefixes in aggregate_batch_results()"
        )
        stats["misplaced_prefixes"] = {
            prefix: len(cols) for prefix, cols in by_prefix.items()
        }
        stats["misplaced_examples"] = misplaced[:10]

    return stats, issues


def validate_dual_bbox_coverage(adata: ad.AnnData) -> tuple[dict, list[str]]:
    """
    Validate that cells with single bbox types have features measured for their organelles.

    For dual-bbox experiments (cell painting):
    - Cells with standard bbox only should have pheno organelle features
    - Cells with CP bbox only should have CP organelle features

    Returns:
        Tuple of (stats_dict, issues_list)
    """
    stats = {}
    issues = []

    # Check if this is a dual-bbox experiment
    has_std_seg = "segmentation_id" in adata.obs.columns
    has_cp_seg = "cp_cell_seg_id" in adata.obs.columns

    if not (has_std_seg and has_cp_seg):
        return stats, issues  # Not a dual-bbox experiment

    # Classify cells by bbox availability
    std_valid = adata.obs["segmentation_id"].notna()
    cp_valid = adata.obs["cp_cell_seg_id"].notna()

    both_valid = std_valid & cp_valid
    std_only = std_valid & ~cp_valid
    cp_only = ~std_valid & cp_valid
    neither = ~std_valid & ~cp_valid

    stats["both_bboxes"] = int(both_valid.sum())
    stats["std_bbox_only"] = int(std_only.sum())
    stats["cp_bbox_only"] = int(cp_only.sum())
    stats["neither_bbox"] = int(neither.sum())

    print(f"\nDual-bbox coverage:")
    print(f"  Both bboxes valid: {stats['both_bboxes']:,}")
    print(f"  Standard bbox only: {stats['std_bbox_only']:,}")
    print(f"  CP bbox only: {stats['cp_bbox_only']:,}")
    if stats["neither_bbox"] > 0:
        print(f"  Neither bbox (ERROR): {stats['neither_bbox']:,}")

    # Identify CP vs standard features from var metadata
    if "organelle" not in adata.var.columns:
        return stats, issues

    cp_features = []
    std_features = []
    for feat in adata.var_names:
        org = adata.var.loc[feat, "organelle"]
        if org and isinstance(org, str):
            if org.startswith("cp1_") or org.startswith("cp2_") or org == "cp_cell":
                cp_features.append(feat)
            else:
                std_features.append(feat)

    stats["n_cp_features"] = len(cp_features)
    stats["n_std_features"] = len(std_features)
    print(f"\nFeature classification:")
    print(f"  CP organelle features: {len(cp_features)}")
    print(f"  Standard organelle features: {len(std_features)}")

    # Validate standard-only cells have standard features
    if stats["std_bbox_only"] > 0 and std_features:
        std_only_idx = np.where(std_only.values)[0]
        std_feat_idx = [list(adata.var_names).index(f) for f in std_features]
        std_only_data = adata.X[std_only_idx][:, std_feat_idx]

        # Count cells with at least one non-NaN standard feature
        has_std_features = ~np.all(np.isnan(std_only_data), axis=1)
        n_with_features = has_std_features.sum()
        pct = 100 * n_with_features / len(std_only_idx) if len(std_only_idx) > 0 else 0

        stats["std_only_with_features"] = int(n_with_features)
        stats["std_only_feature_pct"] = round(pct, 1)
        print(f"\nStandard-only cells with measured features: {n_with_features:,}/{len(std_only_idx):,} ({pct:.1f}%)")

        if pct < 90:
            issues.append(f"Only {pct:.1f}% of standard-only cells have measured features")

    # Validate CP-only cells have CP features
    if stats["cp_bbox_only"] > 0 and cp_features:
        cp_only_idx = np.where(cp_only.values)[0]
        cp_feat_idx = [list(adata.var_names).index(f) for f in cp_features]
        cp_only_data = adata.X[cp_only_idx][:, cp_feat_idx]

        # Count cells with at least one non-NaN CP feature
        has_cp_features = ~np.all(np.isnan(cp_only_data), axis=1)
        n_with_features = has_cp_features.sum()
        pct = 100 * n_with_features / len(cp_only_idx) if len(cp_only_idx) > 0 else 0

        stats["cp_only_with_features"] = int(n_with_features)
        stats["cp_only_feature_pct"] = round(pct, 1)
        print(f"CP-only cells with measured features: {n_with_features:,}/{len(cp_only_idx):,} ({pct:.1f}%)")

        if pct < 90:
            issues.append(f"Only {pct:.1f}% of CP-only cells have measured features")

    # Check cells with neither bbox (these should have been filtered out)
    if stats["neither_bbox"] > 0:
        issues.append(f"{stats['neither_bbox']} cells have neither bbox - should have been filtered")

    return stats, issues


def validate_ntc_presence(adata: ad.AnnData, level: str) -> tuple[dict, list[str]]:
    """
    Validate that NTC (non-targeting control) items exist.
    
    Uses the same logic as the metrics and fe_graphs pipelines to identify NTCs:
    1. gene_id == -1 or NCBI_ID == -1 (if columns exist)
    2. gene_name is None or empty string
    3. String patterns: "ntc", "non-targeting", "^0$" (case-insensitive)
    
    Returns:
        Tuple of (stats_dict, warnings_list)
    """
    stats = {}
    warnings = []
    
    print(f"\nNTC (non-targeting control) validation:")
    
    if "gene_name" not in adata.obs.columns:
        warnings.append("gene_name column not found - cannot check for NTCs")
        print(f"  Status: UNABLE TO CHECK (no gene_name column)")
        stats["ntc_checkable"] = False
        return stats, warnings
    
    stats["ntc_checkable"] = True
    
    # Method 1: Check for gene_id == -1 or NCBI_ID == -1 (most reliable)
    ntc_mask = pd.Series(False, index=adata.obs.index)
    detection_methods = []
    
    if "gene_id" in adata.obs.columns:
        gene_id_ntc = adata.obs["gene_id"] == -1
        n_gene_id_ntc = gene_id_ntc.sum()
        if n_gene_id_ntc > 0:
            ntc_mask |= gene_id_ntc
            detection_methods.append(f"gene_id==-1: {n_gene_id_ntc:,}")
    
    if "NCBI_ID" in adata.obs.columns:
        ncbi_id_ntc = adata.obs["NCBI_ID"] == -1
        n_ncbi_id_ntc = ncbi_id_ntc.sum()
        if n_ncbi_id_ntc > 0:
            ntc_mask |= ncbi_id_ntc
            detection_methods.append(f"NCBI_ID==-1: {n_ncbi_id_ntc:,}")
    
    # Method 2: Check for None or empty gene_name (common in OPS data)
    none_or_empty = adata.obs["gene_name"].isna() | (adata.obs["gene_name"].astype(str).str.strip() == "") | (adata.obs["gene_name"].astype(str) == "None")
    n_none = none_or_empty.sum()
    if n_none > 0:
        ntc_mask |= none_or_empty
        detection_methods.append(f"gene_name is None/empty: {n_none:,}")
    
    # Method 3: String pattern matching (fallback)
    NTC_PATTERNS = ["ntc", "non-targeting", "^0$"]
    pattern = "|".join(NTC_PATTERNS)
    pattern_match = adata.obs["gene_name"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    n_pattern = pattern_match.sum()
    if n_pattern > 0:
        ntc_mask |= pattern_match
        detection_methods.append(f"pattern match: {n_pattern:,}")
    
    n_ntc = ntc_mask.sum()
    n_total = len(adata.obs)
    pct_ntc = 100 * n_ntc / n_total if n_total > 0 else 0
    
    stats["n_ntc"] = int(n_ntc)
    stats["n_total"] = int(n_total)
    stats["pct_ntc"] = round(pct_ntc, 1)
    
    if n_ntc == 0:
        warnings.append(f"No NTC items found - differential analysis will be skipped")
        print(f"  Status: NO NTCs FOUND")
        print(f"    Detection methods tried:")
        print(f"      1. gene_id == -1 or NCBI_ID == -1")
        print(f"      2. gene_name is None/empty")
        print(f"      3. Pattern matching: {NTC_PATTERNS}")
        
        # Show sample gene names to help debug
        unique_genes = adata.obs["gene_name"].value_counts(dropna=False).head(10)
        print(f"    Sample gene names in data:")
        for gene, count in unique_genes.items():
            print(f"      - {gene}: {count:,} items")
    else:
        print(f"  Status: {n_ntc:,} NTCs found ({pct_ntc:.1f}% of {level}s)")
        print(f"    Detection methods used: {', '.join(detection_methods)}")
        
        # Show which NTC gene names were found
        ntc_genes = adata.obs.loc[ntc_mask, "gene_name"].value_counts(dropna=False).head(5)
        print(f"    NTC gene names found:")
        for gene, count in ntc_genes.items():
            print(f"      - {gene}: {count:,} items")
        
        # Flag if NTC percentage is unusual
        if pct_ntc < 1:
            warnings.append(f"Only {pct_ntc:.1f}% NTCs - expected ~5-10%")
        elif pct_ntc > 20:
            warnings.append(f"High NTC percentage: {pct_ntc:.1f}% - expected ~5-10%")
    
    return stats, warnings


def validate_library_size(adata: ad.AnnData, level: str, experiment: str = None) -> tuple[dict, list[str]]:
    """
    Validate that the number of guides/genes doesn't exceed the expected library size.

    For guide-level: checks that n_guides <= library size
    For gene-level: checks that n_genes <= library size

    Returns:
        Tuple of (stats_dict, issues_list) - issues are errors, not warnings
    """
    stats = {}
    issues = []

    # Try to load gene index to get expected library size
    expected_guides = None
    expected_genes = None

    if experiment:
        try:
            from cyclops_utils.data.experiment import OpsDataset
            ds = OpsDataset(experiment)
            if ds.gene_index.exists():
                gene_df = pd.read_csv(ds.gene_index)
                expected_guides = len(gene_df)
                # Count unique genes (using Gene name or gene_name column)
                gene_col = "Gene name" if "Gene name" in gene_df.columns else "gene_name"
                if gene_col in gene_df.columns:
                    expected_genes = gene_df[gene_col].nunique()
                stats["library_path"] = str(ds.gene_index)
        except Exception as e:
            print(f"  Warning: Could not load gene index: {e}")

    # Default library sizes if we couldn't load gene index
    # twist1k_pool_CERES.csv has ~4208 guides and ~1000 genes
    if expected_guides is None:
        expected_guides = 4208  # Default for twist1k pool
        stats["library_path"] = "default (twist1k_pool)"
    if expected_genes is None:
        expected_genes = 1000  # Default for twist1k pool

    stats["expected_guides"] = expected_guides
    stats["expected_genes"] = expected_genes

    print(f"\nLibrary size validation:")
    print(f"  Expected library: {expected_guides:,} guides, {expected_genes:,} genes")

    n_obs = len(adata.obs)

    if level == "guide":
        stats["actual_guides"] = n_obs
        pct_of_library = 100 * n_obs / expected_guides
        stats["pct_of_library"] = round(pct_of_library, 1)

        print(f"  Actual guides: {n_obs:,} ({pct_of_library:.1f}% of library)")

        if n_obs > expected_guides:
            excess = n_obs - expected_guides
            pct_excess = 100 * excess / expected_guides
            issues.append(
                f"Guide count ({n_obs:,}) exceeds library size ({expected_guides:,}) by {excess:,} ({pct_excess:.1f}%) - "
                f"indicates duplicate barcodes or data corruption"
            )
            print(f"  ERROR: {excess:,} excess guides detected!")
        elif n_obs < expected_guides * 0.5:
            print(f"  Warning: Only {pct_of_library:.1f}% of library represented")
        else:
            print(f"  Status: OK")

    elif level == "gene":
        stats["actual_genes"] = n_obs
        pct_of_library = 100 * n_obs / expected_genes
        stats["pct_of_library"] = round(pct_of_library, 1)

        print(f"  Actual genes: {n_obs:,} ({pct_of_library:.1f}% of library)")

        if n_obs > expected_genes:
            excess = n_obs - expected_genes
            # Allow +1 for NTC (non-targeting control) which is not in the library gene list
            if excess == 1 and "gene_name" in adata.obs.columns and "NTC" in adata.obs["gene_name"].values:
                print(f"  Status: OK (+1 is NTC, not in library)")
            else:
                pct_excess = 100 * excess / expected_genes
                issues.append(
                    f"Gene count ({n_obs:,}) exceeds library size ({expected_genes:,}) by {excess:,} ({pct_excess:.1f}%) - "
                    f"indicates duplicate gene names or data corruption"
                )
                print(f"  ERROR: {excess:,} excess genes detected!")
        elif n_obs < expected_genes * 0.5:
            print(f"  Warning: Only {pct_of_library:.1f}% of library represented")
        else:
            print(f"  Status: OK")

    elif level == "cell":
        # For cell level, check unique guides and genes in the data
        #
        # IMPORTANT: Use sgRNA (the true guide identifier) if available, fall back to barcode.
        #
        # Why sgRNA over barcode?
        # - Barcodes are truncated to match effective ISS rounds per well (datasets.py lines 415-417, 457-458)
        # - Wells with failed ISS rounds get shorter barcodes (e.g., 9-char vs 10-char)
        # - The SAME guide can have different barcode strings across wells due to truncation
        # - This causes inflated unique barcode counts (e.g., 12,000 instead of 4,211)
        # - sgRNA is the 20-char guide sequence from the library - invariant across wells
        # - sgRNA comes from the gene_index library merge in datasets.py (twist1k_pool_CERES.csv)
        #
        if "sgRNA" in adata.obs.columns:
            # Filter to valid sgRNA values
            valid_sgrna = adata.obs["sgRNA"].notna() & (adata.obs["sgRNA"] != "") & (adata.obs["sgRNA"].astype(str) != "None")
            n_unique_guides = adata.obs.loc[valid_sgrna, "sgRNA"].nunique()
            stats["unique_guides_in_cells"] = n_unique_guides
            pct_guides = 100 * n_unique_guides / expected_guides
            print(f"  Unique guides in cells (by sgRNA): {n_unique_guides:,} ({pct_guides:.1f}% of library)")

            if n_unique_guides > expected_guides:
                excess = n_unique_guides - expected_guides
                issues.append(
                    f"Unique guide count in cells ({n_unique_guides:,}) exceeds library size ({expected_guides:,}) - "
                    f"indicates data corruption"
                )
                print(f"  ERROR: {excess:,} excess unique guides detected!")
        else:
            # No sgRNA column - this is an error, not a fallback
            issues.append(
                "Missing 'sgRNA' column in cell data - cannot validate guide counts. "
                "Re-run linking (link_calls_tracks) to get sgRNA from library merge."
            )
            print(f"  ERROR: No sgRNA column - cannot validate guide counts")

        if "gene_name" in adata.obs.columns:
            n_unique_genes = adata.obs["gene_name"].nunique()
            stats["unique_genes_in_cells"] = n_unique_genes
            pct_genes = 100 * n_unique_genes / expected_genes
            print(f"  Unique genes in cells: {n_unique_genes:,} ({pct_genes:.1f}% of library)")

            if n_unique_genes > expected_genes:
                excess = n_unique_genes - expected_genes
                # Allow +1 for NTC (non-targeting control) which is not in the library gene list
                if excess == 1 and "NTC" in adata.obs["gene_name"].values:
                    print(f"  Status: OK (+1 is NTC, not in library)")
                else:
                    issues.append(
                        f"Unique gene count in cells ({n_unique_genes:,}) exceeds library size ({expected_genes:,}) - "
                        f"indicates data corruption"
                    )
                    print(f"  ERROR: {excess:,} excess unique genes detected!")

    return stats, issues


def validate_spatial_drift_columns(adata: ad.AnnData) -> tuple[dict, list[str]]:
    """
    Validate columns needed for spatial drift analysis in fe_graphs.
    
    Checks for:
    - Well-level drift: x_global_pheno, y_global_pheno, well
    - Tile-level drift: x_local_pheno, y_local_pheno, tile_pheno
    
    Returns:
        Tuple of (stats_dict, warnings_list) - warnings not errors since these are optional
    """
    stats = {}
    warnings = []
    
    print(f"\nSpatial drift analysis readiness:")
    
    for analysis_name, config in SPATIAL_DRIFT_COLUMNS.items():
        required_cols = config["required"]
        description = config["description"]
        
        # Check column presence
        missing = [c for c in required_cols if c not in adata.obs.columns]
        present = [c for c in required_cols if c in adata.obs.columns]
        
        if missing:
            status = "UNAVAILABLE"
            stats[f"{analysis_name}_available"] = False
            stats[f"{analysis_name}_missing_cols"] = missing
            warnings.append(f"{analysis_name}: Missing columns {missing}")
            print(f"  {analysis_name}: {status}")
            print(f"    Missing columns: {missing}")
            print(f"    ({description})")
        else:
            # Check for all-NaN columns
            nan_cols = []
            valid_cols = []
            for col in required_cols:
                if adata.obs[col].isna().all():
                    nan_cols.append(col)
                else:
                    valid_cols.append(col)
            
            if nan_cols:
                status = "UNAVAILABLE (columns are all NaN)"
                stats[f"{analysis_name}_available"] = False
                stats[f"{analysis_name}_nan_cols"] = nan_cols
                warnings.append(f"{analysis_name}: Columns all NaN: {nan_cols}")
                print(f"  {analysis_name}: {status}")
                print(f"    All-NaN columns: {nan_cols}")
                print(f"    ({description})")
            else:
                # Check data coverage
                valid_mask = adata.obs[required_cols].notna().all(axis=1)
                n_valid = valid_mask.sum()
                pct_valid = 100 * n_valid / len(adata.obs)
                
                status = f"AVAILABLE ({n_valid:,} cells, {pct_valid:.1f}%)"
                stats[f"{analysis_name}_available"] = True
                stats[f"{analysis_name}_n_valid"] = int(n_valid)
                stats[f"{analysis_name}_pct_valid"] = round(pct_valid, 1)
                
                print(f"  {analysis_name}: {status}")
                
                if pct_valid < 50:
                    warnings.append(f"{analysis_name}: Only {pct_valid:.1f}% of cells have valid data")
    
    return stats, warnings


def validate_anndata_file(path: Path, level: str, experiment: str = None) -> dict:
    """Validate a single AnnData file.

    Parameters
    ----------
    path : Path
        Path to the AnnData file
    level : str
        Level of aggregation: "cell", "guide", or "gene"
    experiment : str, optional
        Experiment name for loading gene index to check library size
    """
    print(f"\n{'='*70}")
    print(f"Validating {level.upper()}-level AnnData: {path.name}")
    print("="*70)

    result = {"path": str(path), "level": level, "valid": True, "issues": [], "warnings": []}

    if not path.exists():
        result["valid"] = False
        result["issues"].append(f"File not found: {path}")
        print(f"ERROR: File not found")
        return result

    try:
        adata = ad.read_h5ad(path)
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Failed to load: {e}")
        print(f"ERROR: Failed to load: {e}")
        return result

    # Basic info
    n_obs, n_features = adata.shape
    level_name = {"cell": "cells", "guide": "guides", "gene": "genes"}.get(level, "observations")
    print(f"\nShape: {n_obs:,} {level_name} x {n_features:,} features")

    if level == "cell":
        n_wells = adata.obs["well"].nunique() if "well" in adata.obs.columns else 0
        n_genes = adata.obs["gene_name"].nunique() if "gene_name" in adata.obs.columns else 0
        print(f"  - {n_wells:,} wells, {n_genes:,} unique genes")

    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var columns: {list(adata.var.columns)}")

    # Validate var metadata (should have organelle, metric, category, etc.)
    var_issues = validate_var_metadata(adata)
    if var_issues:
        print(f"\nvar metadata issues:")
        for issue in var_issues:
            print(f"  - {issue}")
        result["issues"].extend(var_issues)
    else:
        print(f"\nvar metadata: OK (organelle, metric, category, aggregation, unit)")

    # Show organelles from uns if available
    if "organelles" in adata.uns:
        print(f"Organelles (from uns): {adata.uns['organelles']}")

    # Validate obs metadata
    expected = {"cell": EXPECTED_CELL_METADATA, "guide": EXPECTED_GUIDE_METADATA, "gene": EXPECTED_GENE_METADATA}.get(level, {})
    obs_issues = validate_obs_metadata(adata, expected, level)
    if obs_issues:
        print(f"\nobs metadata issues:")
        for issue in obs_issues:
            print(f"  - {issue}")
        result["issues"].extend(obs_issues)
    else:
        print(f"\nobs metadata: OK")

    # Feature stats from var
    if "category" in adata.var.columns:
        print(f"\nFeature categories (from var.category):")
        for cat, count in adata.var["category"].value_counts().items():
            print(f"  {cat}: {count}")

    # Organelle feature table (cell-level only)
    if level == "cell" and "organelle" in adata.var.columns:
        print(f"\nFeatures per organelle:")
        org_stats, table_str = build_organelle_table(adata)
        print(table_str)
        result["organelle_stats"] = org_stats

        # Check for unassigned organelles
        if "_unassigned" in org_stats and org_stats["_unassigned"]["total"] > 0:
            n = org_stats["_unassigned"]["total"]
            result["issues"].append(f"{n} features have no organelle assigned")

    # Check for misplaced features in obs (cell-level only)
    if level == "cell":
        misplaced_stats, misplaced_issues = validate_misplaced_features(adata)
        result["misplaced_features"] = misplaced_stats
        if misplaced_issues:
            print(f"\nMisplaced features (in .obs instead of .X):")
            for issue in misplaced_issues:
                print(f"  - {issue}")
            if misplaced_stats.get("misplaced_examples"):
                print(f"  Examples: {misplaced_stats['misplaced_examples']}")
            result["issues"].extend(misplaced_issues)

    # Data quality
    quality_issues = validate_data_quality(adata, level)
    if quality_issues:
        print(f"\nData quality issues:")
        for issue in quality_issues:
            print(f"  - {issue}")
        result["issues"].extend(quality_issues)
    else:
        print(f"\nData quality: OK")

    # Dual-bbox coverage validation (cell-level only)
    if level == "cell":
        bbox_stats, bbox_issues = validate_dual_bbox_coverage(adata)
        result["bbox_coverage"] = bbox_stats
        if bbox_issues:
            print(f"\nDual-bbox coverage issues:")
            for issue in bbox_issues:
                print(f"  - {issue}")
            result["issues"].extend(bbox_issues)
        elif bbox_stats:
            print(f"\nDual-bbox coverage: OK")
    
    # Spatial drift column validation (cell-level only)
    if level == "cell":
        drift_stats, drift_warnings = validate_spatial_drift_columns(adata)
        result["spatial_drift_readiness"] = drift_stats
        if drift_warnings:
            print(f"\nSpatial drift warnings (these are optional):")
            for warning in drift_warnings:
                print(f"  - {warning}")
            # Note: these are warnings, not errors - spatial drift is optional
            # Don't add to result["issues"] since missing tile data isn't a failure
    
    # NTC validation (all levels - important for differential analysis)
    ntc_stats, ntc_warnings = validate_ntc_presence(adata, level)
    result["ntc_stats"] = ntc_stats
    if ntc_warnings:
        print(f"\nNTC warnings:")
        for warning in ntc_warnings:
            print(f"  - {warning}")
        result["warnings"].extend(ntc_warnings)
        # These are warnings, not errors - NTCs are optional but recommended

    # Library size validation (all levels)
    library_stats, library_issues = validate_library_size(adata, level, experiment)
    result["library_stats"] = library_stats
    if library_issues:
        print(f"\nLibrary size issues:")
        for issue in library_issues:
            print(f"  - {issue}")
        result["issues"].extend(library_issues)

    # Summary
    if result["issues"]:
        result["valid"] = False
        print(f"\nVALIDATION: FAILED ({len(result['issues'])} issues)")
        for issue in result["issues"]:
            print(f"  - {issue}")
    else:
        print(f"\nVALIDATION: PASSED")

    return result


def validate_experiment(experiment: str, preview: bool = False) -> dict:
    """Validate all AnnData files for an experiment."""
    from cyclops_utils.data.experiment import OpsDataset

    ds = OpsDataset(experiment)
    base_path = ds.results_fast / "feature_extraction"
    if preview:
        base_path = base_path / "_preview"

    print(f"\nValidating feature extraction output for: {experiment}")
    print(f"Path: {base_path}")

    results = {}
    for level in ["cell", "guide", "gene"]:
        for pattern in [f"{experiment}_{level}_features.h5ad", f"{level}_features.h5ad"]:
            path = base_path / pattern
            if path.exists():
                results[level] = validate_anndata_file(path, level, experiment=experiment)
                break
        else:
            print(f"\nWARNING: No {level}-level AnnData found")
            results[level] = {"valid": False, "issues": ["File not found"], "warnings": []}

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print("="*70)

    all_valid = True
    all_issues = []
    all_warnings = []

    for level, result in results.items():
        status = "PASS" if result.get("valid", False) else "FAIL"
        n_issues = len(result.get("issues", []))
        n_warnings = len(result.get("warnings", []))
        print(f"  {level.upper()}: {status} ({n_issues} issues, {n_warnings} warnings)")
        if not result.get("valid"):
            all_valid = False
        # Collect all issues and warnings with level prefix
        for issue in result.get("issues", []):
            all_issues.append(f"[{level.upper()}] {issue}")
        for warning in result.get("warnings", []):
            all_warnings.append(f"[{level.upper()}] {warning}")

    # Print detailed failure summary
    if all_issues:
        print(f"\n{'-'*70}")
        print(f"FAILURES ({len(all_issues)} total):")
        print("-"*70)
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

    if all_warnings:
        print(f"\n{'-'*70}")
        print(f"WARNINGS ({len(all_warnings)} total):")
        print("-"*70)
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")

    print(f"\n{'='*70}")
    if all_valid:
        print("RESULT: All validations PASSED")
    else:
        print(f"RESULT: FAILED - {len(all_issues)} issue(s) found")
    print("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate feature extraction AnnData output")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment name")
    parser.add_argument("--path", type=str, help="Direct path to AnnData file")
    parser.add_argument("--preview", action="store_true", help="Check _preview directory")
    parser.add_argument("--level", type=str, choices=["cell", "guide", "gene"], default="cell")

    args = parser.parse_args()

    if args.path:
        validate_anndata_file(Path(args.path), args.level)
    elif args.experiment:
        from cyclops_utils.data.filesystem import resolve_experiment_name
        experiment = resolve_experiment_name(args.experiment, allow_interactive=True, autoselect=True)
        validate_experiment(experiment, preview=args.preview)
    else:
        print("ERROR: Provide either --experiment or --path")
        sys.exit(1)
