import ast
import random
from typing import Callable, List, Literal, Optional

import zarr
import numpy as np
import pandas as pd
import torch
from iohub import open_ome_zarr
from iohub.core.registry import set_default_implementation
import tensorstore as ts

# iohub >=0.3 defaults to the zarrs (Rust) codec pipeline, which raises
# "incompatible offset ... for region" on cross-shard-boundary crops of the
# sharded phenotyping stores. Force the pure-Python zarr backend for all reads.
set_default_implementation("zarr-python")

from torch.utils.data import Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    SpatialPadd,
    ToTensord,
)


# Global cache for tensorstore handles (shared across BaseDataset instances in same process)
_TS_HANDLE_CACHE = {}


def _open_ts_array(store_path: str, well: str, array_path: str = "0") -> ts.TensorStore:
    """
    Open a zarr v3 array using tensorstore (thread-safe) with caching.
    
    Tensorstore handles are cached by (store_path, well, array_path) to avoid
    the overhead of opening new handles for every read.
    
    Parameters
    ----------
    store_path : str
        Path to the zarr store root
    well : str  
        Well/position key (e.g., 'A/1/0')
    array_path : str
        Path within the well (e.g., '0' for intensity, 'labels/cell_seg/0' for mask)
    
    Returns
    -------
    ts.TensorStore
        Thread-safe tensorstore handle (cached)
    """
    cache_key = (store_path, well, array_path)
    if cache_key not in _TS_HANDLE_CACHE:
        full_path = f"{store_path}/{well}/{array_path}"
        spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': full_path},
        }
        _TS_HANDLE_CACHE[cache_key] = ts.open(spec).result()
    return _TS_HANDLE_CACHE[cache_key]


def normalize_bbox(bbox_val) -> tuple | None:
    """
    Normalize bbox from various formats to a tuple of 4 integers (ymin, xmin, ymax, xmax).
    
    Handles:
    - tuple/list: Direct conversion
    - numpy.ndarray: Convert to tuple (common from parquet)
    - str: Parse from string representation like "(100, 200, 300, 400)"
    - None/NaN: Return None
    
    Parameters
    ----------
    bbox_val : any
        Bbox value in any supported format.
        
    Returns
    -------
    tuple | None
        Normalized (ymin, xmin, ymax, xmax) tuple, or None if invalid/missing.
        
    Examples
    --------
    >>> normalize_bbox((10, 20, 30, 40))
    (10, 20, 30, 40)
    >>> normalize_bbox(np.array([10, 20, 30, 40]))
    (10, 20, 30, 40)
    >>> normalize_bbox("(10, 20, 30, 40)")
    (10, 20, 30, 40)
    >>> normalize_bbox(None)
    None
    """
    # Handle None or NaN
    if bbox_val is None:
        return None
    if isinstance(bbox_val, float) and pd.isna(bbox_val):
        return None
    
    # Handle string formats
    if isinstance(bbox_val, str):
        bbox_val = bbox_val.strip()
        if bbox_val.lower() in ("none", "nan", ""):
            return None
        try:
            # Try standard Python literal_eval first (handles "(1, 2, 3, 4)" and "[1, 2, 3, 4]")
            bbox_val = ast.literal_eval(bbox_val)
        except (ValueError, SyntaxError):
            # Fallback: Handle numpy array string format from AnnData HDF5 serialization
            # Format: "[1 2 3 4]" (space-separated, no commas)
            if bbox_val.startswith('[') and bbox_val.endswith(']'):
                try:
                    values = bbox_val[1:-1].split()
                    bbox_val = [int(v) for v in values if v]
                except (ValueError, TypeError):
                    return None
            else:
                return None
    
    # Handle array-like (tuple, list, numpy array, etc.)
    if hasattr(bbox_val, '__iter__') and hasattr(bbox_val, '__len__'):
        try:
            bbox_tuple = tuple(int(v) for v in bbox_val)
            if len(bbox_tuple) == 4:
                return bbox_tuple
        except (TypeError, ValueError):
            return None
    
    return None


class BaseDataset(Dataset):
    """
    Base PyTorch Dataset for loading and preprocessing microscopy image patches with associated labels.

    This dataset handles loading image patches from OME-Zarr stores, applying spatial transformations,
    and preparing data for training deep learning models on microscopy data. It supports single or
    multiple channel selection, cell masking, and flexible patch sizing.

    Attributes:
        stores (dict): Dictionary mapping store keys to OME-Zarr store objects.
        labels_df (pd.DataFrame): DataFrame containing crop information and labels for each sample.
        initial_yx_patch_size (tuple): Initial spatial size (height, width) for cropping patches.
        final_yx_patch_size (tuple): Final spatial size after padding/cropping transformations.
        out_channels (List[str] | Literal["random"] | Literal["all"]): Channel selection strategy.
        label_int_lut (dict): Lookup table mapping gene names to integer labels.
        mask_cell (bool): Whether to apply cell segmentation mask to image data.
        use_original_crop_size (bool): Whether to use original crop size without padding/cropping.
        transform (Compose): MONAI composition of transformations to apply to data.
    """

    def __init__(
        self,
        stores: dict,
        labels_df: pd.DataFrame,
        initial_yx_patch_size: tuple = (128, 128),
        final_yx_patch_size: tuple = (128, 128),
        out_channels: List[str] | Literal["random"] | Literal["all"] = "random",
        label_int_lut: Optional[dict] = None,  # string --> int
        mask_cell: bool = True,
        use_original_crop_size: bool = False,
        store_paths: Optional[dict] = None,  # For tensorstore thread-safe reads
        use_tensorstore: bool = False,  # Enable tensorstore for thread-safe reads
    ):
        """
        Initialize the BaseDataset.

        Args:
            stores (dict): Dictionary mapping store keys to opened OME-Zarr store objects.
            labels_df (pd.DataFrame): DataFrame with columns including 'gene_name', 'bbox',
                'store_key', 'well', 'segmentation_id', and 'total_index'.
            initial_yx_patch_size (tuple, optional): Initial (height, width) size for extracting
                patches before transformations. Defaults to (128, 128).
            final_yx_patch_size (tuple, optional): Final (height, width) size after padding and
                center cropping. Defaults to (128, 128).
            out_channels (List[str] | Literal["random"] | Literal["all"], optional): Strategy for
                channel selection. "random" selects one random channel per sample, "all" uses all
                available channels, or provide a list of specific channel names. Defaults to "random".
            label_int_lut (Optional[dict], optional): Lookup table mapping gene names (str) to
                integer labels (int). If None, automatically generated from unique gene names in
                labels_df. Defaults to None.
            mask_cell (bool, optional): If True, multiply image data by segmentation mask to isolate
                individual cells. Defaults to True.
            use_original_crop_size (bool, optional): If True, skip padding/cropping transformations
                and use original bounding box size. Defaults to False.
            store_paths (Optional[dict], optional): Dictionary mapping store keys to store paths
                (strings). Required when use_tensorstore=True for thread-safe reads.
            use_tensorstore (bool, optional): If True, use tensorstore for thread-safe array reads.
                Requires store_paths to be provided. Defaults to False.
        """
        self.stores = stores
        self.labels_df = labels_df
        self.initial_yx_patch_size = initial_yx_patch_size
        self.final_yx_patch_size = final_yx_patch_size
        self.out_channels = out_channels
        self.mask_cell = mask_cell
        self.use_original_crop_size = use_original_crop_size
        self.store_paths = store_paths or {}
        self.use_tensorstore = use_tensorstore
        if label_int_lut is None:
            # Replace NaN gene names with "NTC" (non-targeting control) before building lookup
            gene_labels = sorted([
                "NTC" if pd.isna(g) else g
                for g in self.labels_df["gene_name"].unique()
            ])
            # Remove duplicates that may arise from multiple NaN -> NTC conversions
            gene_labels = sorted(set(gene_labels))
            self.label_int_lut = {gene: i for i, gene in enumerate(gene_labels)}
        else:
            self.label_int_lut = label_int_lut

        if self.use_original_crop_size:
            self.transform = Compose(
                [
                    ToTensord(keys=["data", "mask"]),
                ]
            )
        else:
            self.transform = Compose(
                [
                    SpatialPadd(
                        keys=["data", "mask"],
                        spatial_size=self.initial_yx_patch_size,
                    ),
                    CenterSpatialCropd(
                        keys=["data", "mask"], roi_size=(self.final_yx_patch_size)
                    ),
                    ToTensord(
                        keys=["data", "mask"],
                    ),
                ]
            )

        return

    def _get_bbox(self, ci, final_shape):
        """
        Extract and optionally expand bounding box to match target shape.

        Parses the bounding box from crop info and pads it equally on all sides if it's
        smaller than the target shape. Padding is distributed symmetrically to keep the
        original crop centered.

        Args:
            ci: Row from labels_df containing crop information with a 'bbox' field.
            final_shape (tuple): Target (height, width) dimensions for the bounding box.

        Returns:
            tuple or None: Bounding box as (ymin, xmin, ymax, xmax). If use_original_crop_size is True,
                returns the original bbox. Otherwise, returns padded bbox matching final_shape.
                Returns None if bbox is missing or invalid.

        Note:
            The bbox format is (ymin, xmin, ymax, xmax) representing top-left and bottom-right
            corners in (y, x) coordinates.
        """
        # Normalize bbox from any format (tuple, list, numpy array, string)
        bbox = normalize_bbox(ci.bbox)
        if bbox is None:
            return None

        if not self.use_original_crop_size:

            if len(final_shape) > 2:
                final_shape = final_shape[-2:]

            ymin, xmin, ymax, xmax = bbox
            target_height, target_width = final_shape

            # Calculate current bbox dimensions
            current_height = ymax - ymin
            current_width = xmax - xmin

            # Calculate padding needed
            height_padding = max(0, target_height - current_height)
            width_padding = max(0, target_width - current_width)

            # Distribute padding equally on both sides
            pad_top = height_padding / 2
            pad_bottom = height_padding / 2
            pad_left = width_padding / 2
            pad_right = width_padding / 2

            # Apply padding
            new_ymin = int(ymin - pad_top)
            new_ymax = int(ymax + pad_bottom)
            new_xmin = int(xmin - pad_left)
            new_xmax = int(xmax + pad_right)
            bbox = (new_ymin, new_xmin, new_ymax, new_xmax)

        return bbox

    def _get_channels(self, ci):
        """
        Determine which channels to load based on the configured strategy.

        Retrieves available channel names from the OME-Zarr metadata and selects channels
        according to the out_channels strategy (random, all, or specific channels).

        Args:
            ci: Row from labels_df containing 'store_key' and 'well' fields to identify
                the data source.

        Returns:
            tuple: A tuple containing:
                - channel_names (list): List of channel name strings to load.
                - channel_index (list): List of integer indices corresponding to the channels
                  in the OME-Zarr store.

        Note:
            If out_channels is "random", selects one random channel per call.
            If out_channels is "all", returns all available channels.
            Otherwise, uses the specific channel names provided in out_channels.
        """

        # Support both zarr v2 (.attrs) and v3 (.zattrs) APIs
        position = self.stores[ci.store_key][ci.well]
        if hasattr(position, 'zattrs'):
            attrs = dict(position.zattrs)
        else:
            attrs = position.attrs.asdict()
        # Handle different metadata structures: v2 has ome->omero->channels, v3 has omero->channels
        if "omero" in attrs:
            omero = attrs["omero"]
        elif "ome" in attrs and "omero" in attrs["ome"]:
            omero = attrs["ome"]["omero"]
        else:
            raise ValueError(f"Could not find omero metadata in attrs: {list(attrs.keys())}")
        all_channel_names = [a["label"] for a in omero["channels"]]

        if self.out_channels == "random":
            channel_names = [random.choice(all_channel_names)]
        elif self.out_channels == "all":
            channel_names = all_channel_names
        else:
            channel_names = self.out_channels
        channel_index = [all_channel_names.index(c) for c in channel_names]

        return channel_names, channel_index

    def __len__(self):
        return len(self.labels_df)

    def add_labels_to_batch(self, ci):
        """
        Extract label information from crop info for the current sample.

        Converts gene name to integer label using the lookup table and retrieves
        additional metadata for tracking.

        Args:
            ci: Row from labels_df containing 'gene_name' and 'total_index' fields.

        Returns:
            tuple: A tuple containing:
                - gene_label (int): Integer label for the gene name.
                - total_index: Unique identifier for this sample.
                - crop_info (dict): Dictionary representation of all crop metadata.
        """
        # Handle NaN gene_name - treat as NTC (non-targeting control)
        gene_name = ci.gene_name
        if pd.isna(gene_name):
            gene_name = "NTC"
        gene_label = self.label_int_lut.get(gene_name, -1)
        total_index = ci.total_index

        return gene_label, total_index, ci.to_dict()

    def add_mask_to_batch(self, ci, bbox):
        """
        Load and extract binary segmentation mask for a specific cell.

        Retrieves the segmentation mask from the OME-Zarr store, crops it to the bounding box,
        and creates a binary mask for the specific cell identified by segmentation_id.

        Args:
            ci: Row from labels_df containing 'store_key', 'well', and 'segmentation_id' fields.
            bbox (tuple): Bounding box as (ymin, xmin, ymax, xmax) defining the region to extract.

        Returns:
            np.ndarray: Binary mask of shape (1, height, width) where True indicates pixels
                belonging to the target cell (segmentation_id) and False elsewhere.
                If segmentation_id is NaN/None, returns a mask of all ones (no masking).
        """
        # Calculate mask shape from bbox
        mask_height = bbox[2] - bbox[0]
        mask_width = bbox[3] - bbox[1]

        # If no segmentation ID, return full mask (all ones)
        if pd.isna(ci.segmentation_id):
            return np.ones((1, mask_height, mask_width), dtype=bool)

        if self.use_tensorstore and ci.store_key in self.store_paths:
            # Thread-safe tensorstore read with caching
            store_path = self.store_paths[ci.store_key]
            ts_mask = _open_ts_array(store_path, ci.well, "labels/cell_seg/0")
            # Zarr v3: (T, C, Z, Y, X) -> read slice
            mask = ts_mask[0, 0, 0, bbox[0]:bbox[2], bbox[1]:bbox[3]].read().result()
            mask = np.expand_dims(mask, axis=0)
        else:
            # Original iohub read
            position = self.stores[ci.store_key][ci.well]
            # Access labels via zgroup for zarr v3 API
            mask_fov = position.zgroup['labels']['cell_seg']['0']

            mask = np.asarray(
                mask_fov[0:1, :, 0:1, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
            ).copy()
            mask = np.squeeze(mask)
            mask = np.expand_dims(mask, axis=0)
        
        sc_mask = mask == ci.segmentation_id

        return sc_mask

    def load_label_array(self, store_key: str, well: str, label_name: str, bbox: tuple) -> np.ndarray:
        """
        Load any label array from the zarr store using the robust read pattern.

        This method uses the same np.asarray(...).copy() pattern that works correctly
        for zarr v3 sharded arrays, avoiding blank chunk issues.

        Args:
            store_key: Key identifying which store to use (e.g., 'pheno_assembled_v3')
            well: Well identifier (e.g., 'A/1/0')
            label_name: Name of the label array (e.g., 'cell_seg', 'cp_cell_seg', 'focus3d_vesicular')
            bbox: Bounding box as (ymin, xmin, ymax, xmax)

        Returns:
            np.ndarray: Label array of shape (height, width)
        """
        if self.use_tensorstore and store_key in self.store_paths:
            store_path = self.store_paths[store_key]
            ts_label = _open_ts_array(store_path, well, f"labels/{label_name}/0")
            label_array = ts_label[0, 0, 0, bbox[0]:bbox[2], bbox[1]:bbox[3]].read().result()
        else:
            position = self.stores[store_key][well]
            if "labels" not in position.zgroup or label_name not in position.zgroup["labels"]:
                return None
            label_fov = position.zgroup["labels"][label_name]["0"]

            # Use robust pattern: np.asarray(...).copy() with explicit slice() objects
            label_array = np.asarray(
                label_fov[0:1, :, 0:1, slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])]
            ).copy()
            label_array = np.squeeze(label_array)

        return label_array

    def add_data_to_batch(self, ci, bbox, channel_index):
        """
        Load and extract image data for specified channels and bounding box.

        Retrieves raw microscopy image data from the OME-Zarr store, crops it to the
        bounding box region, and extracts the specified channels.

        Args:
            ci: Row from labels_df containing 'store_key' and 'well' fields to identify
                the data source.
            bbox (tuple): Bounding box as (ymin, xmin, ymax, xmax) defining the region to extract.
            channel_index (list): List of integer indices specifying which channels to load.

        Returns:
            np.ndarray: Image data as float32 array with shape (C, height, width) where C is
                the number of channels. Single channel images are expanded to (1, height, width).
        """
        if self.use_tensorstore and ci.store_key in self.store_paths:
            # Thread-safe tensorstore read with caching
            store_path = self.store_paths[ci.store_key]
            ts_fov = _open_ts_array(store_path, ci.well, "0")
            # Zarr v3: (T, C, Z, Y, X) -> read slice
            data = ts_fov[0, channel_index, 0, bbox[0]:bbox[2], bbox[1]:bbox[3]].read().result()
            # Squeeze T and Z dims (already done by slicing), keep C
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)
        else:
            # Original iohub read
            fov = self.stores[ci.store_key][ci.well]["0"]
            data = np.asarray(
                fov[
                    0:1,
                    channel_index,
                    0:1,
                    slice(bbox[0], bbox[2]),
                    slice(bbox[1], bbox[3]),
                ]
            ).copy()
            data = np.squeeze(data)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)

        return data.astype(np.float32)

    def __getitem__(self, index):
        """
        Load and preprocess a single sample from the dataset.

        This is the main data loading method called by PyTorch DataLoader. It orchestrates
        loading the image patch, segmentation mask, and labels, applies transformations,
        and returns a dictionary containing all sample data.

        Args:
            index (int): Index of the sample to retrieve from labels_df.

        Returns:
            dict: Dictionary containing:
                - 'data' (torch.Tensor): Image data of shape (C, H, W) or (1, C, H, W) if
                  final_yx_patch_size has 3 dimensions. Optionally masked by cell segmentation.
                - 'mask' (torch.Tensor): Binary segmentation mask of shape (1, H, W) or
                  (1, 1, H, W), with same dimensionality as data.
                - 'marker_label' (list): List of channel names loaded for this sample.
                - 'gene_label' (int): Integer label for the gene associated with this cell.
                - 'total_index' (int): Unique identifier for this sample.
                - 'crop_info' (dict): Complete metadata for this crop from labels_df.

        Note:
            If mask_cell is True, the returned data will be element-wise multiplied by the mask.
            Transformations (padding, cropping, tensor conversion) are applied based on the
            transform pipeline configured during initialization.
        """
        batch = {}
        ci = self.labels_df.iloc[index]  # crop info
        bbox = self._get_bbox(ci, self.initial_yx_patch_size)
        c_names, c_index = self._get_channels(ci)
        batch["marker_label"] = c_names

        gene_label, total_index, crop_info = self.add_labels_to_batch(ci)
        batch["gene_label"] = gene_label
        batch["total_index"] = int(total_index)
        batch["crop_info"] = crop_info
        batch["bbox"] = bbox  # Include computed bbox for downstream use (may be None)

        # Handle missing bbox - return batch with empty data/mask for downstream to skip
        if bbox is None:
            batch["data"] = np.zeros((1, 1, 1), dtype=np.float32)
            batch["mask"] = np.zeros((1, 1, 1), dtype=np.uint8)
            return batch

        batch["data"] = self.add_data_to_batch(ci, bbox, c_index)
        batch["mask"] = self.add_mask_to_batch(ci, bbox)

        if self.mask_cell:
            batch["data"] = batch["data"] * batch["mask"]

        if len(self.final_yx_patch_size) == 3:
            batch["data"] = np.expand_dims(batch["data"], axis=0)
            batch["mask"] = np.expand_dims(batch["mask"], axis=0)

        if self.transform is not None:
            batch = self.transform(batch)

        return batch