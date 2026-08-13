"""
Feature provenance and biological metadata management.

This module provides tools for tracking the biological meaning of features
across experiments using the existing ops_channel_maps.yaml file.

Usage:
    from cyclops_utils.data.feature_metadata import FeatureMetadata

    # Initialize (automatically loads ops_channel_maps.yaml)
    meta = FeatureMetadata()

    # Get biological information
    bio_signal = meta.get_biological_signal("ops0089", "GFP")
    # Returns: "early endosome, EEA1"

    # Create informative feature names
    feat_name = meta.create_feature_name("ops0089", "GFP", "dinov3", 0)
    # Returns: "ops0089_EEA1_dinov3_0"

    # Add metadata to AnnData
    meta.add_to_anndata(adata, "ops0089", "GFP", "dinov3")
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import yaml

from cyclops_utils.data.filesystem import extract_ops_key
from cyclops_utils.paths import BASE_PATH

logger = logging.getLogger(__name__)


class FeatureMetadata:
    """
    Manage feature provenance metadata across experiments.

    Loads and parses the existing ops_channel_maps.yaml file to provide
    biological context for imaging channels across OPS experiments.

    Attributes:
        metadata_path: Path to ops_channel_maps.yaml
        metadata: Dictionary of experiment -> channel mappings
    """

    def __init__(self, metadata_path: str = None):
        """
        Initialize metadata manager.

        Args:
            metadata_path: Path to ops_channel_maps.yaml
                          (default: $OPS_BASE_PATH/configs/ops_channel_maps.yaml)
        """
        if metadata_path is None:
            # Default location of existing OPS channel maps
            metadata_path = Path(
                f"{BASE_PATH}/configs/ops_channel_maps.yaml"
            )

        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata from YAML file."""
        if not self.metadata_path.exists():
            print(f"Warning: Metadata file not found at {self.metadata_path}")
            return {}

        with open(self.metadata_path, "r") as f:
            return yaml.safe_load(f)

    def _parse_label(self, label: str) -> Dict[str, Optional[str]]:
        """
        Parse label string into components.

        Labels can be:
        - "organelle, marker_protein" (e.g., "mitochondria, TOMM70A")
        - "Phase" (for brightfield)
        - "no label" (for unlabeled channels)

        Args:
            label: Label string from ops_channel_maps.yaml

        Returns:
            Dict with 'organelle' and 'marker' keys
        """
        if label == "Phase":
            return {"organelle": "Phase", "marker": None}
        elif label == "no label" or label is None:
            return {"organelle": "unlabeled", "marker": None}
        elif "," in label:
            # Split on comma: "mitochondria, TOMM70A" -> ["mitochondria", "TOMM70A"]
            parts = [p.strip() for p in label.split(",", 1)]
            return {
                "organelle": parts[0],
                "marker": parts[1] if len(parts) > 1 else None,
            }
        else:
            # Single-word label (e.g., "5xUPRE")
            return {"organelle": label, "marker": None}

    def normalize_channel_name(self, data_channel: str) -> str:
        """
        Map data channel names to YAML channel names.

        The extraction uses names like "Phase2D", but YAML uses "BF".
        This method normalizes variants to match YAML keys.

        Args:
            data_channel: Channel name from feature columns (e.g., "Phase2D", "GFP")

        Returns:
            Normalized channel name for YAML lookup (e.g., "BF", "GFP")

        Examples:
            >>> meta = FeatureMetadata()
            >>> meta.normalize_channel_name("Phase2D")
            'BF'
            >>> meta.normalize_channel_name("Phase3D")
            'BF'
            >>> meta.normalize_channel_name("GFP")
            'GFP'
        """
        if data_channel.startswith("Phase"):
            return "BF"
        return data_channel

    def get_channel_info(self, experiment: str, channel: str) -> Dict:
        """
        Get biological metadata for a specific channel.

        Args:
            experiment: Experiment name (e.g., "ops0089" or "ops0089_20251119")
            channel: Channel name (e.g., "GFP", "BF", "mCherry", "Phase2D")

        Returns:
            Dictionary with organelle, marker, label, channel_name

        Example:
            >>> meta = FeatureMetadata()
            >>> info = meta.get_channel_info("ops0089", "GFP")
            >>> print(info)
            {'channel_name': 'GFP',
             'label': 'early endosome, EEA1',
             'organelle': 'early endosome',
             'marker': 'EEA1'}
        """
        # Remove date suffix if present (ops0089_20251119 -> ops0089)
        exp_short = extract_ops_key(experiment) or experiment.split("_")[0]

        if exp_short not in self.metadata:
            print(f"Warning: No metadata for experiment {exp_short}")
            return {
                "channel_name": channel,
                "label": "unknown",
                "organelle": "unknown",
                "marker": None,
            }

        # Normalize channel name (e.g., Phase2D -> BF)
        normalized_channel = self.normalize_channel_name(channel)

        # Find channel in list (skip entries without channel_name, e.g. cell_painting config)
        channels = self.metadata[exp_short]
        for ch in channels:
            if not isinstance(ch, dict) or "channel_name" not in ch:
                continue
            if ch["channel_name"] == normalized_channel:
                parsed = self._parse_label(ch["label"])
                return {
                    "channel_name": channel,  # Return original channel name
                    "label": ch["label"],
                    "organelle": parsed["organelle"],
                    "marker": parsed["marker"],
                }

        # Cell painting channels: derive metadata from name (e.g. CP1_mitochondria_TOMM20)
        if channel.startswith(("CP1_", "CP2_")):
            parts = channel.split("_", 2)  # ["CP1", "mitochondria", "TOMM20"]
            if len(parts) == 3:
                return {
                    "channel_name": channel,
                    "label": f"{parts[1]}, {parts[2]}",
                    "organelle": parts[1],
                    "marker": parts[2],
                }

        # 4i channels: derive metadata from name when the YAML uses a slightly
        # different channel_name (e.g. store "4i_R1_nuclei_DAPI" vs yaml
        # "4i_R1_nuclei"). Strip the "4i_R{N}_" round prefix; the remainder is
        # "{organelle}_{marker}" (e.g. nuclei_DAPI) or a bare marker (e.g. p53).
        if channel.startswith(("4i_R", "4i_r")):
            parts = channel.split("_", 2)  # ["4i", "R1", "nuclei_DAPI" | "p53"]
            if len(parts) == 3:
                rest = parts[2]
                if "_" in rest:  # organelle + marker (e.g. nuclei_DAPI)
                    organelle, marker = rest.split("_", 1)
                    label = f"{organelle}, {marker}"
                else:            # bare marker (e.g. p53, b-catenin)
                    organelle = marker = rest
                    label = rest
                return {
                    "channel_name": channel,
                    "label": label,
                    "organelle": organelle,
                    "marker": marker,
                }

        logger.debug(
            "Channel '%s' (normalized: '%s') not found in metadata YAML for %s; "
            "file loading uses separate fallback logic. Available channels: %s",
            channel, normalized_channel, exp_short,
            [ch.get('channel_name') for ch in channels if isinstance(ch, dict) and 'channel_name' in ch],
        )
        return {
            "channel_name": channel,
            "label": "unknown",
            "organelle": "unknown",
            "marker": None,
        }

    @staticmethod
    def sanitize_label(label: str) -> str:
        """
        Sanitize and simplify label for use as HDF5 key or dictionary key.

        Processing steps:
        1. Replace problematic path separators (/ and \\) with underscore
        2. If comma present: extract the marker (second part after comma)
        3. Replace all spaces with underscores

        This extracts the marker protein name from "organelle, marker" format
        and creates clean, HDF5-safe keys.

        Args:
            label: Label string to sanitize

        Returns:
            Sanitized label safe for HDF5 keys

        Examples:
            >>> FeatureMetadata.sanitize_label("ER/Golgi, COPE")
            'COPE'
            >>> FeatureMetadata.sanitize_label("early endosome, EEA1")
            'EEA1'
            >>> FeatureMetadata.sanitize_label("mitochondria, TOMM70A")
            'TOMM70A'
            >>> FeatureMetadata.sanitize_label("Phase")
            'Phase'
            >>> FeatureMetadata.sanitize_label("5xUPRE")
            '5xUPRE'
            >>> FeatureMetadata.sanitize_label("no label")
            'no_label'
            >>> FeatureMetadata.sanitize_label("early endosome")
            'early_endosome'
        """
        # Step 1: Replace path separators with underscore
        sanitized = label.replace("/", "_").replace("\\", "_")

        # Step 2: If comma present, extract the marker (second part)
        if "," in sanitized:
            parts = sanitized.split(",", 1)  # Split on first comma only
            sanitized = parts[1].strip()  # Take second part and strip whitespace

        # Step 3: Replace all spaces with underscores
        sanitized = sanitized.replace(" ", "_")

        return sanitized

    def get_biological_signal(self, experiment: str, channel: str) -> str:
        """
        Get human-readable biological signal name.

        Args:
            experiment: Experiment name
            channel: Channel name

        Returns:
            Full label string (e.g., "early endosome, EEA1")
            Sanitized for safe use as dictionary keys and HDF5 storage.

        Example:
            >>> meta = FeatureMetadata()
            >>> meta.get_biological_signal("ops0089", "GFP")
            'early endosome, EEA1'
            >>> meta.get_biological_signal("ops0062", "GFP")
            'ER_Golgi, COPE'  # Note: / replaced with _
        """
        info = self.get_channel_info(experiment, channel)
        label = info.get("label", "unknown")
        return self.sanitize_label(label)

    def get_short_label(self, experiment: str, channel: str) -> str:
        """
        Get short label for feature naming.

        Returns marker if available, otherwise organelle.
        For unlabeled channels, includes channel name (e.g., "unlabeled_GFP").

        Args:
            experiment: Experiment name
            channel: Channel name

        Returns:
            Short label suitable for variable names

        Examples:
            >>> meta = FeatureMetadata()
            >>> meta.get_short_label("ops0089", "GFP")
            'EEA1'  # marker from "early endosome, EEA1"
            >>> meta.get_short_label("ops0089", "BF")
            'Phase'  # no marker, returns organelle
            >>> meta.get_short_label("ops0031", "GFP")  # if no label
            'unlabeled_GFP'
        """
        info = self.get_channel_info(experiment, channel)
        marker = info.get("marker")
        if marker:
            return marker

        organelle = info.get("organelle", channel)
        # If unlabeled, include channel name to distinguish between channels
        if organelle == "unlabeled":
            return f"unlabeled_{channel}"

        return organelle

    def create_feature_name(
        self,
        experiment: str,
        channel: str,
        feature_type: str,
        feature_idx: int,
    ) -> str:
        """
        Create informative feature name with biological context.

        Args:
            experiment: Experiment name (e.g., "ops0089")
            channel: Channel name (e.g., "GFP")
            feature_type: Feature extraction method (dinov3, cellprofiler)
            feature_idx: Feature index (0-based)

        Returns:
            Feature name like "ops0089_EEA1_dinov3_0"

        Example:
            >>> meta = FeatureMetadata()
            >>> meta.create_feature_name("ops0089", "GFP", "dinov3", 0)
            'ops0089_EEA1_dinov3_0'
            >>> meta.create_feature_name("ops0108", "GFP", "dinov3", 512)
            'ops0108_TOMM70A_dinov3_512'
        """
        short_label = self.get_short_label(experiment, channel)
        exp_short = extract_ops_key(experiment) or experiment.split("_")[0]
        return f"{exp_short}_{short_label}_{feature_type}_{feature_idx}"

    def add_to_anndata(
        self,
        adata,
        experiment: str,
        channel: str,
        feature_type: str,
    ):
        """
        Add metadata to AnnData .uns field.

        Stores complete channel and biological information in the AnnData
        object for later reference and provenance tracking.

        Args:
            adata: AnnData object to annotate
            experiment: Experiment name
            channel: Channel name
            feature_type: Feature type (dinov3, cellprofiler, etc.)

        Example:
            >>> import anndata as ad
            >>> meta = FeatureMetadata()
            >>> adata = ad.read_h5ad("data.h5ad")
            >>> meta.add_to_anndata(adata, "ops0089", "GFP", "dinov3")
            >>> print(adata.uns["feature_metadata"])
            {'ops0089_GFP_dinov3': {
                'experiment': 'ops0089',
                'channel': 'GFP',
                'feature_type': 'dinov3',
                'channel_info': {
                    'channel_name': 'GFP',
                    'label': 'early endosome, EEA1',
                    'organelle': 'early endosome',
                    'marker': 'EEA1'
                }
            }}
        """
        if "feature_metadata" not in adata.uns:
            adata.uns["feature_metadata"] = {}

        exp_short = extract_ops_key(experiment) or experiment.split("_")[0]
        key = f"{exp_short}_{channel}_{feature_type}"
        adata.uns["feature_metadata"][key] = {
            "experiment": exp_short,
            "channel": channel,
            "feature_type": feature_type,
            "channel_info": self.get_channel_info(experiment, channel),
        }

    def list_experiments(self) -> List[str]:
        """
        Get list of all experiments with metadata.

        Returns:
            Sorted list of experiment names

        Example:
            >>> meta = FeatureMetadata()
            >>> exps = meta.list_experiments()
            >>> print(exps[:5])
            ['ops0003', 'ops0006', 'ops0007', 'ops0008', 'ops0009']
        """
        return sorted(self.metadata.keys())

    def get_all_channels(self, experiment: str) -> List[str]:
        """
        Get list of all channels for an experiment.

        Args:
            experiment: Experiment name (with or without date suffix)

        Returns:
            List of channel names

        Example:
            >>> meta = FeatureMetadata()
            >>> channels = meta.get_all_channels("ops0089")
            >>> print(channels)
            ['BF', 'GFP']
        """
        exp_short = extract_ops_key(experiment) or experiment.split("_")[0]
        if exp_short not in self.metadata:
            return []
        return [ch["channel_name"] for ch in self.metadata[exp_short]]

    def get_experiments_by_marker(self, marker: str) -> List[tuple]:
        """
        Find all experiments using a specific marker protein.

        Args:
            marker: Marker protein name (e.g., "TOMM70A")

        Returns:
            List of (experiment, channel) tuples

        Example:
            >>> meta = FeatureMetadata()
            >>> exps = meta.get_experiments_by_marker("TOMM70A")
            >>> print(exps)
            [('ops0052', 'GFP'), ('ops0108', 'GFP')]
        """
        results = []
        for exp, channels in self.metadata.items():
            for ch in channels:
                parsed = self._parse_label(ch["label"])
                if parsed["marker"] == marker:
                    results.append((exp, ch["channel_name"]))
        return results

    def get_experiments_by_organelle(self, organelle: str) -> List[tuple]:
        """
        Find all experiments targeting a specific organelle.

        Args:
            organelle: Organelle name (e.g., "mitochondria", "ER")

        Returns:
            List of (experiment, channel, marker) tuples

        Example:
            >>> meta = FeatureMetadata()
            >>> exps = meta.get_experiments_by_organelle("mitochondria")
            >>> for exp, ch, marker in exps[:3]:
            ...     print(f"{exp}/{ch}: {marker}")
            ops0046/GFP: TOMM20
            ops0052/GFP: TOMM70A
            ops0108/GFP: TOMM70A
        """
        results = []
        organelle_lower = organelle.lower()
        for exp, channels in self.metadata.items():
            for ch in channels:
                parsed = self._parse_label(ch["label"])
                if parsed["organelle"].lower() == organelle_lower:
                    results.append((exp, ch["channel_name"], parsed["marker"]))
        return results

    def validate_experiment(self, experiment: str, verbose: bool = True) -> bool:
        """
        Check if experiment has complete metadata.

        Args:
            experiment: Experiment name
            verbose: Print validation details

        Returns:
            True if valid, False otherwise

        Example:
            >>> meta = FeatureMetadata()
            >>> meta.validate_experiment("ops0089")
            Experiment ops0089:
              BF: Phase
                ✓ Brightfield
              GFP: early endosome, EEA1
                ✓ Organelle: early endosome, Marker: EEA1
            True
        """
        exp_short = extract_ops_key(experiment) or experiment.split("_")[0]

        if exp_short not in self.metadata:
            if verbose:
                print(f"✗ Experiment {exp_short} not found in metadata")
            return False

        channels = self.metadata[exp_short]
        if verbose:
            print(f"Experiment {exp_short}:")

        all_valid = True
        for ch in channels:
            if not isinstance(ch, dict) or "channel_name" not in ch:
                continue  # skip non-channel entries (e.g. cell_painting config)
            channel_name = ch["channel_name"]
            label = ch["label"]

            if verbose:
                print(f"  {channel_name}: {label}")

            if label == "no label":
                if verbose:
                    print(f"    ⚠ Warning: Unlabeled channel")
            elif label == "Phase":
                if verbose:
                    print(f"    ✓ Brightfield")
            elif "," in label:
                parts = label.split(",", 1)
                if verbose:
                    print(
                        f"    ✓ Organelle: {parts[0].strip()}, Marker: {parts[1].strip()}"
                    )
            else:
                if verbose:
                    print(f"    ⚠ Non-standard label format: {label}")

        return all_valid

    def replace_channel_in_feature_name(
        self, feature_name: str, experiment: str
    ) -> str:
        """
        Replace channel names with reporter labels in feature column names.

        Uses pattern matching against known channels from metadata to correctly
        handle channel names with spaces (which become underscores in CSV columns).
        Reporter names are sanitized using sanitize_label() for consistency with
        get_biological_signal() (spaces → underscores, organelle prefix removed).

        Args:
            feature_name: Original feature name from CSV
            experiment: Experiment identifier (e.g., "ops0031" or "ops0031_20250424")

        Returns:
            Feature name with reporter labels instead of channel names (spaces removed)

        Examples:
            >>> meta = FeatureMetadata()
            >>> # Single-object features
            >>> meta.replace_channel_in_feature_name("single_object_Phase2D_cell_Area", "ops0031")
            'single_object_Phase_cell_Area'
            >>> meta.replace_channel_in_feature_name("single_object_mCherry_nucleus_MeanIntensity", "ops0031")
            'single_object_SEC61B_nucleus_MeanIntensity'
            >>> # Channel names with spaces handled correctly
            >>> meta.replace_channel_in_feature_name("single_object_ChromaLive_561_emission_cell_Area", "ops0033")
            'single_object_ChromaLive_561_emission_cell_Area'  # Spaces → underscores (consistent with sanitize_label)
            >>> # Colocalization features
            >>> meta.replace_channel_in_feature_name("coloc_mCherry_GFP_cell_Correlation_Pearson", "ops0031")
            'coloc_SEC61B_5xUPRE_cell_Correlation_Pearson'
            >>> # Non-feature columns pass through unchanged
            >>> meta.replace_channel_in_feature_name("label_str", "ops0031")
            'label_str'
        """
        # Get all channels for this experiment from metadata
        exp_short = extract_ops_key(experiment) or experiment.split("_")[0]
        if exp_short not in self.metadata:
            return feature_name  # Can't process without metadata

        # Skip non-channel map entries (e.g. the `cell_painting: {...}` block in
        # Cell-Painting experiments) which carry no `channel_name`.
        channels = [
            c for c in self.metadata[exp_short]
            if isinstance(c, dict) and "channel_name" in c
        ]

        # Build list of channel aliases (for BF channel, also check Phase2D, Phase3D, Phase)
        def get_channel_aliases(channel_name):
            """Get list of possible CSV names for a metadata channel."""
            aliases = [channel_name.replace(" ", "_")]  # Base name with underscores
            if channel_name == "BF":
                # BF in metadata can appear as Phase2D, Phase3D, or Phase in CSV
                aliases.extend(["Phase2D", "Phase3D", "Phase"])
            return aliases

        # For single_object features: single_object_{CHANNEL}_{MASK}_{FEATURE}
        if feature_name.startswith("single_object_"):
            # Try to match against each known channel
            for ch_info in channels:
                channel_name = ch_info["channel_name"]  # e.g., "GFP", "BF"

                # Check all possible CSV representations of this channel
                for channel_csv in get_channel_aliases(channel_name):
                    pattern = f"single_object_{channel_csv}_"
                    if pattern in feature_name:
                        # Get reporter and sanitize consistently with get_biological_signal
                        reporter = self.get_short_label(experiment, channel_name)
                        reporter_clean = self.sanitize_label(
                            reporter
                        )  # "ChromaLive 561 emission" → "ChromaLive_561_emission"

                        # Replace the pattern
                        replacement = f"single_object_{reporter_clean}_"
                        return feature_name.replace(pattern, replacement)

            # No match found - return unchanged
            return feature_name

        # For coloc features: coloc_{CHANNEL1}_{CHANNEL2}_{MASK}_{FEATURE}
        elif feature_name.startswith("coloc_"):
            # Need to match both channels - try all combinations
            for ch1_info in channels:
                for ch2_info in channels:
                    ch1_name = ch1_info["channel_name"]  # e.g., "GFP", "BF"
                    ch2_name = ch2_info["channel_name"]

                    # Check all possible CSV representations of both channels
                    for ch1_csv in get_channel_aliases(ch1_name):
                        for ch2_csv in get_channel_aliases(ch2_name):
                            pattern = f"coloc_{ch1_csv}_{ch2_csv}_"
                            if pattern in feature_name:
                                # Get reporters and sanitize consistently with get_biological_signal
                                reporter1 = self.get_short_label(experiment, ch1_name)
                                reporter2 = self.get_short_label(experiment, ch2_name)
                                reporter1_clean = self.sanitize_label(reporter1)
                                reporter2_clean = self.sanitize_label(reporter2)

                                # Replace the pattern
                                replacement = (
                                    f"coloc_{reporter1_clean}_{reporter2_clean}_"
                                )
                                return feature_name.replace(pattern, replacement)

            # No match found - return unchanged
            return feature_name

        # Not a feature column - return unchanged
        return feature_name

    def __repr__(self) -> str:
        """String representation."""
        n_experiments = len(self.metadata)
        return (
            f"FeatureMetadata(experiments={n_experiments}, path='{self.metadata_path}')"
        )
