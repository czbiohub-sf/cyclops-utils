"""
Tests for feature_metadata.py module.

Tests the channel-to-reporter name mapping functionality.
"""

import pytest
from cyclops_utils.data.feature_metadata import FeatureMetadata


class TestFeatureMetadata:
    """Test FeatureMetadata class methods."""

    @pytest.fixture
    def metadata(self):
        """Create FeatureMetadata instance."""
        return FeatureMetadata()

    def test_normalize_channel_name(self, metadata):
        """Test channel name normalization."""
        # Phase variants should map to BF
        assert metadata.normalize_channel_name("Phase2D") == "BF"
        assert metadata.normalize_channel_name("Phase3D") == "BF"
        assert metadata.normalize_channel_name("Phase") == "BF"

        # Other channels should pass through
        assert metadata.normalize_channel_name("GFP") == "GFP"
        assert metadata.normalize_channel_name("mCherry") == "mCherry"
        assert metadata.normalize_channel_name("Cy5") == "Cy5"

    def test_replace_channel_single_object(self, metadata):
        """Test channel replacement in single-object feature names."""
        # Test with ops0031: mCherry=SEC61B, GFP=5xUPRE
        result = metadata.replace_channel_in_feature_name(
            "single_object_Phase2D_cell_Area", "ops0031"
        )
        assert result == "single_object_Phase_cell_Area"

        result = metadata.replace_channel_in_feature_name(
            "single_object_mCherry_nucleus_MeanIntensity", "ops0031"
        )
        assert result == "single_object_SEC61B_nucleus_MeanIntensity"

        result = metadata.replace_channel_in_feature_name(
            "single_object_GFP_cytoplasm_Texture_Contrast", "ops0031"
        )
        assert result == "single_object_5xUPRE_cytoplasm_Texture_Contrast"

        # Test with ops0089: GFP=EEA1
        result = metadata.replace_channel_in_feature_name(
            "single_object_GFP_cell_Area", "ops0089"
        )
        assert result == "single_object_EEA1_cell_Area"

        # Test with ops0108: GFP=TOMM70A
        result = metadata.replace_channel_in_feature_name(
            "single_object_GFP_nucleus_Intensity_Mean", "ops0108"
        )
        assert result == "single_object_TOMM70A_nucleus_Intensity_Mean"

    def test_replace_channel_colocalization(self, metadata):
        """Test channel replacement in colocalization feature names."""
        # Test with ops0031: mCherry=SEC61B, GFP=5xUPRE
        result = metadata.replace_channel_in_feature_name(
            "coloc_mCherry_GFP_cell_Correlation_Pearson", "ops0031"
        )
        assert result == "coloc_SEC61B_5xUPRE_cell_Correlation_Pearson"

        result = metadata.replace_channel_in_feature_name(
            "coloc_Phase2D_mCherry_nucleus_Correlation_Costes_1", "ops0031"
        )
        assert result == "coloc_Phase_SEC61B_nucleus_Correlation_Costes_1"

        result = metadata.replace_channel_in_feature_name(
            "coloc_Phase2D_GFP_cytoplasm_Correlation_Manders_2", "ops0031"
        )
        assert result == "coloc_Phase_5xUPRE_cytoplasm_Correlation_Manders_2"

    def test_replace_channel_non_feature_columns(self, metadata):
        """Test that non-feature columns pass through unchanged."""
        # Metadata columns should not be modified
        assert (
            metadata.replace_channel_in_feature_name("label_str", "ops0031")
            == "label_str"
        )
        assert (
            metadata.replace_channel_in_feature_name("label_int", "ops0031")
            == "label_int"
        )
        assert metadata.replace_channel_in_feature_name("sgRNA", "ops0031") == "sgRNA"
        assert (
            metadata.replace_channel_in_feature_name("experiment", "ops0031")
            == "experiment"
        )
        assert metadata.replace_channel_in_feature_name("well", "ops0031") == "well"
        assert (
            metadata.replace_channel_in_feature_name("x_position", "ops0031")
            == "x_position"
        )
        assert (
            metadata.replace_channel_in_feature_name("y_position", "ops0031")
            == "y_position"
        )

    def test_replace_channel_with_date_suffix(self, metadata):
        """Test that experiment names with date suffixes work correctly."""
        # ops0031_20250424 should map to ops0031 in YAML
        result = metadata.replace_channel_in_feature_name(
            "single_object_mCherry_cell_Area", "ops0031_20250424"
        )
        assert result == "single_object_SEC61B_cell_Area"

    def test_replace_channel_multiple_experiments(self, metadata):
        """Test that different experiments produce different reporter names."""
        # ops0046: mCherry=H2B
        result_046 = metadata.replace_channel_in_feature_name(
            "single_object_mCherry_cell_Area", "ops0046"
        )
        assert result_046 == "single_object_H2B_cell_Area"

        # ops0031: mCherry=SEC61B
        result_031 = metadata.replace_channel_in_feature_name(
            "single_object_mCherry_cell_Area", "ops0031"
        )
        assert result_031 == "single_object_SEC61B_cell_Area"

        # Verify they're different
        assert result_046 != result_031

    def test_replace_channel_unknown_experiment(self, metadata):
        """Test handling of unknown experiments."""
        # For unknown experiments, the method should still work but may return generic labels
        result = metadata.replace_channel_in_feature_name(
            "single_object_GFP_cell_Area", "ops9999"
        )
        # Should return something (may be "unknown" or the original channel name)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sanitize_label(self):
        """Test label sanitization for HDF5-safe keys."""
        # Test with comma: extract marker after comma, replace slashes, remove spaces
        assert FeatureMetadata.sanitize_label("ER/Golgi, COPE") == "COPE"
        assert FeatureMetadata.sanitize_label("early endosome, EEA1") == "EEA1"
        assert FeatureMetadata.sanitize_label("mitochondria, TOMM70A") == "TOMM70A"
        assert FeatureMetadata.sanitize_label("ER\\Golgi, COPE") == "COPE"

        # Test with spaces in marker name
        assert (
            FeatureMetadata.sanitize_label("organelle, MARKER PROTEIN")
            == "MARKER_PROTEIN"
        )

        # Test without comma: replace slashes and spaces with underscores
        assert FeatureMetadata.sanitize_label("ER/golgi") == "ER_golgi"
        assert FeatureMetadata.sanitize_label("A/B/C") == "A_B_C"
        assert FeatureMetadata.sanitize_label("A\\B\\C") == "A_B_C"
        assert FeatureMetadata.sanitize_label("A/B\\C") == "A_B_C"

        # Test labels with spaces but no comma
        assert FeatureMetadata.sanitize_label("no label") == "no_label"
        assert FeatureMetadata.sanitize_label("early endosome") == "early_endosome"

        # Test labels without problematic characters or spaces
        assert FeatureMetadata.sanitize_label("Phase") == "Phase"
        assert FeatureMetadata.sanitize_label("5xUPRE") == "5xUPRE"

        # Test edge cases
        assert FeatureMetadata.sanitize_label("") == ""
        assert FeatureMetadata.sanitize_label("_") == "_"
        assert FeatureMetadata.sanitize_label("/") == "_"
        assert FeatureMetadata.sanitize_label("\\") == "_"
        assert FeatureMetadata.sanitize_label(" ") == "_"
        assert FeatureMetadata.sanitize_label(", MARKER") == "MARKER"

        # Test multiple spaces
        assert (
            FeatureMetadata.sanitize_label("multiple   spaces") == "multiple___spaces"
        )

    def test_get_biological_signal_sanitized(self, metadata):
        """Test that get_biological_signal returns sanitized labels."""
        # Test with an experiment that has slashes in the label (ops0062: ER/Golgi, COPE)
        result = metadata.get_biological_signal("ops0062", "GFP")
        # Should extract marker and have no slashes or spaces
        assert "/" not in result
        assert "\\" not in result
        assert " " not in result
        assert result == "COPE"

        # Test with normal label (extracts marker)
        result = metadata.get_biological_signal("ops0089", "GFP")
        assert result == "EEA1"

        # Test Phase label (no comma, no spaces)
        result = metadata.get_biological_signal("ops0089", "Phase2D")
        assert result == "Phase"

    def test_replace_channel_with_spaces_in_reporter(self, metadata):
        """Test that reporter names with spaces are handled correctly.

        This test ensures that when reporter names (markers) contain spaces,
        they are completely removed to create clean identifiers suitable for:
        1. Valid Python identifiers
        2. Parsing in evaluate_cp.py's channel_mapping extraction
        3. Clean file names (e.g., features_processed_ChromaLive561emission.h5ad)

        Uses ops0033 where mCherry channel has label "mitochondria, ChromaLive 561 emission"
        The marker "ChromaLive 561 emission" should become "ChromaLive561emission" (no spaces).
        """
        # Test single-object feature with mCherry channel
        # ops0033: mCherry -> "mitochondria, ChromaLive 561 emission" -> marker is "ChromaLive 561 emission"
        # After replacement, spaces should be removed: "ChromaLive561emission"
        result = metadata.replace_channel_in_feature_name(
            "single_object_mCherry_cell_Area", "ops0033"
        )

        # The result should have NO spaces in the reporter name
        assert " " not in result, f"Result contains spaces: {result}"

        # The result should be "single_object_ChromaLive561emission_cell_Area"
        # Spaces are removed, not replaced with underscores
        assert (
            result == "single_object_ChromaLive561emission_cell_Area"
        ), f"Expected 'single_object_ChromaLive561emission_cell_Area', got '{result}'"

        # Verify there are NO underscores within the reporter name portion
        # The reporter should be "ChromaLive561emission", not "ChromaLive_561_emission"
        parts = result.split("_")
        assert (
            parts[2] == "ChromaLive561emission"
        ), f"Reporter should be 'ChromaLive561emission', got '{parts[2]}'"

        # Test colocalization feature
        result = metadata.replace_channel_in_feature_name(
            "coloc_Phase2D_mCherry_cell_Correlation_Pearson", "ops0033"
        )

        assert " " not in result, f"Result contains spaces: {result}"
        assert (
            result == "coloc_Phase_ChromaLive561emission_cell_Correlation_Pearson"
        ), f"Expected 'coloc_Phase_ChromaLive561emission_cell_Correlation_Pearson', got '{result}'"
