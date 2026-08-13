"""Channel naming and type utilities shared between cyclops_process and organelle_profiler."""

# Label-free channel names (used by get_channel_type)
LABELFREE_CHANNELS = {"Phase2D", "Focus3D", "Phase3D", "Raw"}


def get_channel_type(channel_name: str) -> str:
    """Determine if a channel is label-free or fluorescent."""
    return "labelfree" if channel_name in LABELFREE_CHANNELS else "fluorescent"


def determine_marker_type(marker: str) -> str:
    """
    Determine the type of marker based on naming conventions.

    Args:
        marker: Marker name (e.g., "TOMM20", "LysoTracker live-cell dye")

    Returns:
        One of: "live_cell_dye", "virtual_stain", "endogenous_tag", or None
    """
    if not marker:
        return None

    marker_lower = marker.lower()

    dye_keywords = ["dye", "tracker", "live", "spy", "bodipy", "phrodo",
                    "cellrox", "cellevent", "chromalive", "emission"]
    if any(kw in marker_lower for kw in dye_keywords):
        return "live_cell_dye"

    if marker_lower in ["vs", "2d", "3d", "virtual stain"]:
        return "virtual_stain"

    return "endogenous_tag"


def parse_channel_label(label: str) -> tuple:
    """
    Parse organelle and marker from channel map label.

    The format is "{organelle}, {marker}" (e.g., "mitochondria, TOMM20").

    Returns:
        Tuple of (organelle, marker). If no marker present, marker is None.
    """
    if ", " in label:
        parts = label.split(", ", 1)
        return parts[0].strip(), parts[1].strip()
    return label.strip(), None


def build_channel_metadata(
    channel_name: str,
    channel_index: int,
    channel_label: str = None,
    channel_type: str = None,
) -> dict:
    """
    Build comprehensive metadata for a single channel.

    Args:
        channel_name: Channel name (e.g., "GFP", "mCherry", "Phase2D")
        channel_index: Index of channel in the zarr store
        channel_label: Label from ops_channel_maps.yaml (e.g., "mitochondria, TOMM20")
        channel_type: Channel type (e.g., "fluorescent", "labelfree", "virtual_stain")

    Returns:
        Dictionary with comprehensive channel metadata
    """
    organelle, marker = parse_channel_label(channel_label) if channel_label else (None, None)
    marker_type = determine_marker_type(marker)

    metadata = {
        "name": channel_name,
        "index": channel_index,
    }

    if channel_type:
        metadata["channel_type"] = channel_type

    if channel_label and channel_label.lower() not in ["no label", "phase"]:
        metadata["biological_annotation"] = {
            "organelle": organelle,
            "marker": marker,
            "marker_type": marker_type,
            "full_label": channel_label,
        }

        if organelle and marker:
            if channel_type == "fluorescent":
                metadata["description"] = f"Max projected {organelle} visualized via {marker}"
            else:
                metadata["description"] = f"{organelle.capitalize()} visualized via {marker}"
        elif organelle:
            if channel_type == "fluorescent":
                metadata["description"] = f"Max projected {organelle}"
            else:
                metadata["description"] = organelle.capitalize()

    return metadata
