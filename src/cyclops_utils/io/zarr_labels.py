"""Zarr label I/O utilities for creating and managing label arrays in zarr stores."""

import numpy as np
from pathlib import Path
import zarr

try:
    from iohub.ngff_meta import TransformationMeta
except ImportError:
    TransformationMeta = None


def get_position_shape(zarr_path: Path, pos_path: str) -> tuple:
    """
    Get the shape of the image array at a position in a v3 zarr store.

    Args:
        zarr_path: Path to zarr v3 store
        pos_path: Position path (e.g., "A/1/0")

    Returns:
        tuple: Shape of the image array (T, C, Z, Y, X)
    """
    store = zarr.open(str(zarr_path), mode="r")
    return store[pos_path]["0"].shape


def _init_organelle_label_array(
    zarr_path: Path,
    pos_path: str,
    organelle_name: str,
    shape: tuple,
    dtype=np.int32,
    chunks: tuple = (1, 1, 1, 512, 512),
    shards_ratio: tuple = (1, 1, 1, 32, 32),
):
    """
    Create output array under labels/{organelle_name}/0 for organelle segmentation.

    Uses same chunking and sharding as convert_v3.py for consistency:
    - chunks=(1, 1, 1, 512, 512)
    - shards_ratio=(1, 1, 1, 32, 32) for single-channel labels (~1GB shard files)

    Args:
        zarr_path: Path to the zarr v3 store
        pos_path: Position path like "A/1/0"
        organelle_name: Name of the organelle label group
        shape: Shape of the output array (should be 5D: T, C, Z, Y, X)
        dtype: Data type for the array
        chunks: Chunk size for the array
        shards_ratio: Sharding ratio
    """
    store = zarr.open(str(zarr_path), mode="r+")
    pos_group = store[pos_path]
    # Auto-create labels/ group if missing. Used to be created by convert_v3
    # during v2->v3 conversion; with v3-native stitch the position is created
    # without labels/ so we make it on first use.
    if "labels" not in pos_group:
        labels_group = pos_group.create_group("labels")
        labels_group.attrs.update({
            "labels": [],
            "ome": {"version": "0.5", "labels": []},
        })
    else:
        labels_group = pos_group["labels"]

    if organelle_name not in labels_group:
        subgroup = labels_group.create_group(organelle_name)
        label_shape = (shape[0], 1, shape[2], shape[3], shape[4])

        shards = tuple(c * r for c, r in zip(chunks, shards_ratio))

        subgroup.create_array(
            "0",
            shape=label_shape,
            dtype=dtype,
            chunks=chunks,
            shards=shards,
            fill_value=0
        )
        print(f"  Created labels/{organelle_name}/0 with shape {label_shape}, chunks={chunks}, shards={shards}")
    else:
        print(f"  labels/{organelle_name} already exists, skipping creation")


def _update_labels_metadata(
    zarr_path: Path,
    pos_path: str,
    new_label_name: str,
    metadata: dict = None,
):
    """
    Update the labels group .zattrs to include the new organelle label.

    Args:
        zarr_path: Path to the zarr v3 store
        pos_path: Position path like "A/1/0"
        new_label_name: Name of the new label to add to metadata
        metadata: Optional comprehensive metadata dict
    """
    store = zarr.open(str(zarr_path), mode="r+")
    pos_group = store[pos_path]
    # Auto-create labels/ if a caller updates metadata before any label array
    # has been initialized. Mirrors _init_organelle_label_array.
    if "labels" not in pos_group:
        labels_group = pos_group.create_group("labels")
        labels_group.attrs.update({
            "labels": [],
            "ome": {"version": "0.5", "labels": []},
        })
    else:
        labels_group = pos_group["labels"]

    existing_attrs = dict(labels_group.attrs)
    existing_labels = existing_attrs.get("labels", [])
    existing_ome = existing_attrs.get("ome", {"version": "0.5", "labels": []})
    existing_ome_labels = existing_ome.get("labels", [])

    if new_label_name not in existing_labels:
        existing_labels.append(new_label_name)
    if new_label_name not in existing_ome_labels:
        existing_ome_labels.append(new_label_name)

    labels_group.attrs["labels"] = existing_labels
    labels_group.attrs["ome"] = {
        "version": "0.5",
        "labels": existing_ome_labels
    }

    if new_label_name in labels_group:
        subgroup = labels_group[new_label_name]
        if metadata:
            subgroup.attrs["segmentation_metadata"] = metadata
            print(f"  Set segmentation_metadata on labels/{new_label_name}")
        else:
            subgroup.attrs["segmentation_metadata"] = {
                "label_name": new_label_name,
                "annotation_type": "organelle_segmentation",
                "is_ome_label": True,
                "description": f"Organelle segmentation: {new_label_name}",
            }
            print(f"  Set basic segmentation_metadata on labels/{new_label_name}")
    else:
        print(f"  WARNING: labels/{new_label_name} subgroup not found, cannot set metadata!")
    print(f"  Updated labels metadata with {new_label_name}")


def _write_label_to_tile(dest_pos, label_name, data, scale, chunks):
    """
    Helper function to create/overwrite a label image within the 'labels'
    group of a Zarr position.

    Args:
        dest_pos: Zarr position group (iohub Position object)
        label_name: Name for the label (e.g., "mitoc_tomm20_seg")
        data: Label data array (typically int32)
        scale: Transformation scale metadata
        chunks: Chunk size for the array
    """
    if TransformationMeta is None:
        raise ImportError("iohub is required for _write_label_to_tile")

    if "labels" not in dest_pos:
        dest_pos.create_group("labels")

    if label_name in dest_pos.labels:
        del dest_pos["labels"][label_name]

    dest_pos.labels.create_label(
        name=label_name,
        data=data,
        chunks=chunks,
        transform=[TransformationMeta(type="scale", scale=scale)],
    )


def _check_label_has_data(labels_group, label_name: str) -> bool:
    """
    Check if a label has actual data, not just an empty folder structure.

    Args:
        labels_group: Zarr group for labels
        label_name: Name of the label to check

    Returns:
        True if label has actual data, False otherwise
    """
    try:
        if label_name not in labels_group:
            return False

        label_group = labels_group[label_name]

        if "0" not in label_group:
            return False

        arr = label_group["0"]

        if not hasattr(arr, 'shape') or arr.shape is None:
            return False

        if any(s == 0 for s in arr.shape):
            return False

        try:
            shape = arr.shape
            center_slices = tuple(slice(max(0, s//2 - 100), min(s, s//2 + 100)) for s in shape)
            sample = arr[center_slices]
            if hasattr(sample, 'max'):
                return sample.max() > 0
            return True
        except Exception:
            return True

    except Exception:
        return False
