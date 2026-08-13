from typing import Callable, Optional, List, Tuple
import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from skimage.measure import regionprops
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.hpc.resource_manager import get_optimal_workers


def _worker_get_bounding_boxes_for_tile(
    tile_path_str: str, tile_cells_dict: dict, morphology_path: Path
) -> List[dict]:
    """
    Dask worker to calculate bounding boxes for all cells in a single tile.

    This function reads a tile's segmentation mask, computes region properties,
    and returns bounding box information for each cell assigned to that tile.
    """
    tile_cells = pd.DataFrame(tile_cells_dict)
    bbox_data = []

    try:
        # Open position directly with layout="fov" to skip plate metadata parsing.
        with open_ome_zarr(Path(morphology_path) / tile_path_str, layout="fov", mode="r") as tile_pos:

            if "cell_mask" not in tile_pos:
                print(
                    f"Warning: No 'cell_mask' found for tile {tile_path_str}, skipping {len(tile_cells)} cells."
                )
                return bbox_data

            seg_mask = np.asarray(tile_pos["cell_mask"][0, 0, 0, :, :])
            props = regionprops(seg_mask)
            props_dict = {prop.label: prop for prop in props}

            for _, cell in tile_cells.iterrows():
                cell_id = cell["segmentation_id"]
                original_index = cell["index"]

                if cell_id in props_dict:
                    prop = props_dict[cell_id]
                    bbox_data.append(
                        {
                            "index": original_index,
                            "bbox_min_row": prop.bbox[0],
                            "bbox_min_col": prop.bbox[1],
                            "bbox_max_row": prop.bbox[2],
                            "bbox_max_col": prop.bbox[3],
                            "cell_area": prop.area,
                        }
                    )
                else:
                    print(
                        f"Warning: Cell {cell_id} not found in segmentation for tile {tile_path_str}"
                    )
                    bbox_data.append(
                        {
                            "index": original_index,
                            "bbox_min_row": int(cell["y_local_pheno"] - 64),
                            "bbox_min_col": int(cell["x_local_pheno"] - 64),
                            "bbox_max_row": int(cell["y_local_pheno"] + 64),
                            "bbox_max_col": int(cell["x_local_pheno"] + 64),
                            "cell_area": 128 * 128,
                        }
                    )
    except (KeyError, IndexError):
        print(
            f"Warning: Tile path '{tile_path_str}' could not be resolved in Zarr store. Skipping {len(tile_cells)} cells."
        )
    except Exception as e:
        import traceback

        print(
            f"Error processing tile {tile_path_str} in worker: {e}\n{traceback.format_exc()}"
        )

    return bbox_data


class CellDataLoader:
    """
    A flexible cell data loader that can extract cells at their natural boundaries
    or crop them to specific patch sizes.
    """

    def __init__(self, experiment: str, wells: List[str] = None):
        """
        Initialize the cell data loader.

        Parameters
        ----------
        experiment : str
            The experiment name
        wells : List[str], optional
            List of wells to load. If None, loads all available wells.
        """
        self.experiment = experiment
        self.dataset = OpsDataset(experiment)
        self.wells = wells
        self._store = None
        self._cells_df = None

    def load_cells_metadata(self) -> pd.DataFrame:
        """
        Load cell metadata from linked_results files.

        Returns
        -------
        pd.DataFrame
            DataFrame with cell metadata including coordinates, gene names, etc.
        """
        if self._cells_df is not None:
            return self._cells_df

        # Open the phenotyping store to get available wells
        pheno_store = open_ome_zarr(
            self.dataset.store_paths["pheno_assembled_v3"], mode="r"
        )

        if self.wells is None:
            # Get all available wells
            self.wells = [f"A/{i}/0" for i in pheno_store["A"].group_keys()]

        cell_records = []
        for well in self.wells:
            results_path = self.dataset.append_well("linked_results", well)
            if results_path.exists():
                well_df = pd.read_csv(results_path)
                well_df["well"] = well
                cell_records.append(well_df)

        if not cell_records:
            raise ValueError(f"No linked_results found for wells: {self.wells}")

        self._cells_df = pd.concat(cell_records, ignore_index=True)
        pheno_store.close()

        return self._cells_df

    def get_cell_bounding_boxes(
        self, cells_df: pd.DataFrame = None, debug_tile_count: int = None
    ) -> pd.DataFrame:
        """
        Calculate bounding boxes for each cell based on segmentation masks.

        Parameters
        ----------
        cells_df : pd.DataFrame, optional
            Cell metadata DataFrame. If None, uses loaded metadata.
        debug_tile_count : int, optional
            If set, only process cells from the first N tiles for debugging purposes.

        Returns
        -------
        pd.DataFrame
            DataFrame with added bounding box information
        """
        if cells_df is None:
            cells_df = self.load_cells_metadata()

        # Apply debug tile filtering if specified
        if debug_tile_count is not None:
            # Sort to ensure reproducibility
            unique_tiles = sorted(list(cells_df["tile_pheno"].unique()))[
                :debug_tile_count
            ]
            cells_df = cells_df[cells_df["tile_pheno"].isin(unique_tiles)]
            print(
                f"DEBUG MODE: Processing bounding boxes for {len(cells_df)} cells from {len(unique_tiles)} tiles"
            )

        cells_with_bbox = cells_df.copy()

        morphology_path = self.dataset.result_paths["phenotyping_vs_tiles"]
        grouped = cells_df.groupby("tile_pheno")

        num_workers = get_optimal_workers(use_gpu=False)
        print(
            f"Calculating bounding boxes for {len(grouped)} tiles with {num_workers} Joblib workers..."
        )

        results = Parallel(n_jobs=num_workers)(
            delayed(_worker_get_bounding_boxes_for_tile)(
                tile_path_str=tile_path_str,
                tile_cells_dict=tile_cells.reset_index().to_dict("list"),
                morphology_path=morphology_path,
            )
            for tile_path_str, tile_cells in tqdm(
                grouped, desc="Calculating cell bounding boxes"
            )
        )

        bbox_data = [item for sublist in results if sublist for item in sublist]

        if bbox_data:
            bbox_df = pd.DataFrame(bbox_data).set_index("index")
            cells_with_bbox = cells_with_bbox.join(bbox_df)

        return cells_with_bbox

    def extract_cell_data(
        self,
        cell_row: pd.Series,
        store=None,
        include_mask: bool = True,
        normalize: bool = False,
        use_bounding_box: bool = True,
        patch_size: Tuple[int, int] = (128, 128),
    ) -> dict:
        """
        Extract data for a single cell from the tile-based morphology dataset.

        Parameters
        ----------
        cell_row : pd.Series
            Row from cells DataFrame containing cell metadata and bounding box info.
        store : zarr store, optional
            Morphology dataset store (will open if not provided).
        include_mask : bool, optional
            Whether to include cell mask.
        normalize : bool, optional
            Whether to normalize image data.
        use_bounding_box : bool, optional
            Whether to crop to cell bounding box (True) or use fixed patch size (False).
        patch_size : Tuple[int, int], optional
            The size of the patch to extract if use_bounding_box is False.

        Returns
        -------
        dict
            Dictionary containing 'image', 'mask', 'metadata', and 'bounding_box'.
        """
        close_store = False
        if store is None:
            store = open_ome_zarr(
                self.dataset.result_paths["phenotyping_vs_tiles"], mode="r"
            )
            close_store = True

        try:
            tile_path_str = cell_row["tile_pheno"]
            try:
                path_parts = Path(tile_path_str).parts
                well_name = str(Path(*path_parts[:-1]))
                position_name = path_parts[-1]
                tile_pos = store[well_name][position_name]
            except (KeyError, IndexError):
                raise ValueError(
                    f"Tile position {tile_path_str} not found in morphology dataset"
                )

            tile_shape = tile_pos.data.shape  # (T, C, Z, Y, X)

            if use_bounding_box:
                if not all(
                    k in cell_row
                    for k in [
                        "bbox_min_row",
                        "bbox_min_col",
                        "bbox_max_row",
                        "bbox_max_col",
                    ]
                ):
                    raise ValueError(
                        "Cell metadata must contain bounding box info for use_bounding_box=True."
                    )
                min_r, min_c = int(cell_row["bbox_min_row"]), int(
                    cell_row["bbox_min_col"]
                )
                max_r, max_c = int(cell_row["bbox_max_row"]), int(
                    cell_row["bbox_max_col"]
                )
            else:
                y_center, x_center = int(cell_row["y_local_pheno"]), int(
                    cell_row["x_local_pheno"]
                )
                y_half, x_half = patch_size[0] // 2, patch_size[1] // 2
                min_r, max_r = y_center - y_half, y_center + y_half
                min_c, max_c = x_center - x_half, x_center + x_half

            # Ensure crop boundaries are within the tile dimensions
            min_r, max_r = max(0, min_r), min(tile_shape[-2], max_r)
            min_c, max_c = max(0, min_c), min(tile_shape[-1], max_c)

            # Crop the image data (all channels)
            # Squeeze T dim, keep C, Z, and crop Y, X
            image_data = tile_pos.data[0, :, :, min_r:max_r, min_c:max_c]

            result = {
                "image": np.squeeze(image_data),  # Squeeze Z if it's 1
                "metadata": cell_row.to_dict(),
                "bounding_box": (min_r, min_c, max_r, max_c),
            }

            # Extract and crop the cell mask
            if include_mask and "cell_mask" in tile_pos:
                full_mask = tile_pos["cell_mask"][:]  # (T, C, Z, Y, X)
                # Squeeze T, C, Z dimensions for mask
                cropped_mask = np.squeeze(full_mask[0, 0, :, min_r:max_r, min_c:max_c])

                # Isolate the specific cell's mask from the label image
                cell_id = cell_row["segmentation_id"]
                instance_mask = (cropped_mask == cell_id).astype(np.uint8)
                result["mask"] = instance_mask

            return result

        finally:
            if close_store:
                store.close()

    def extract_multiple_cells(
        self,
        cell_indices: List[int] = None,
        max_cells: int = None,
        include_mask: bool = True,
        normalize: bool = False,
        use_bounding_box: bool = True,
        filter_by_gene: str = None,
        patch_size: Tuple[int, int] = (128, 128),
    ) -> List[dict]:
        """
        Extract data for multiple cells.

        Parameters
        ----------
        cell_indices : List[int], optional
            Specific cell indices to extract. If None, extracts all or up to max_cells.
        max_cells : int, optional
            Maximum number of cells to extract
        include_mask : bool, optional
            Whether to include cell masks
        normalize : bool, optional
            Whether to normalize image data
        use_bounding_box : bool, optional
            Whether to crop to cell bounding box
        filter_by_gene : str, optional
            Filter cells by gene name
        patch_size : Tuple[int, int], optional
            The size of the patch to extract if use_bounding_box is False.

        Returns
        -------
        List[dict]
            List of cell data dictionaries
        """
        cells_df = self.load_cells_metadata()

        if filter_by_gene:
            cells_df = cells_df[cells_df["gene_name"] == filter_by_gene]

        # Always get bounding boxes as they are needed for edge filtering,
        # and for cropping if use_bounding_box is True.
        cells_df = self.get_cell_bounding_boxes(cells_df)

        if cell_indices is not None:
            cells_df = cells_df.iloc[cell_indices]
        elif max_cells is not None:
            cells_df = cells_df.head(max_cells)

        print(f"Extracting data for {len(cells_df)} cells...")

        cell_data_list = []

        # Use the tile-based store
        with open_ome_zarr(
            self.dataset.result_paths["phenotyping_vs_tiles"], mode="r"
        ) as store:

            for _, cell_row in cells_df.iterrows():
                try:
                    cell_data = self.extract_cell_data(
                        cell_row,
                        store=store,
                        include_mask=include_mask,
                        normalize=normalize,
                        use_bounding_box=use_bounding_box,
                        patch_size=patch_size,
                    )
                    cell_data_list.append(cell_data)
                except Exception as e:
                    print(
                        f"Error extracting cell {cell_row.get('segmentation_id', 'unknown')}: {e}"
                    )
                    continue

        return cell_data_list


class FlexibleCellDataset(Dataset):
    """
    A PyTorch Dataset that can work with either fixed patches or full cell boundaries.
    """

    def __init__(
        self,
        experiment: str,
        wells: List[str] = None,
        transform: Optional[Callable] = None,
        use_bounding_box: bool = True,
        patch_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
        filter_by_gene: str = None,
        max_cells: int = None,
    ):
        """
        Initialize the flexible cell dataset.

        Parameters
        ----------
        experiment : str
            The experiment name
        wells : List[str], optional
            List of wells to include
        transform : Callable, optional
            Transform to apply to data
        use_bounding_box : bool, optional
            Whether to use actual cell bounding boxes (True) or fixed patches (False)
        patch_size : Tuple[int, int], optional
            Fixed patch size if use_bounding_box is False
        normalize : bool, optional
            Whether to normalize image data
        filter_by_gene : str, optional
            Filter cells by gene name
        max_cells : int, optional
            Maximum number of cells to include
        """
        self.loader = CellDataLoader(experiment, wells)
        self.transform = transform
        self.use_bounding_box = use_bounding_box
        self.patch_size = patch_size
        self.normalize = normalize

        # Load and prepare cell metadata
        self.cells_df = self.loader.load_cells_metadata()

        if filter_by_gene:
            self.cells_df = self.cells_df[self.cells_df["gene_name"] == filter_by_gene]

        if max_cells is not None:
            self.cells_df = self.cells_df.head(max_cells)

        if use_bounding_box:
            self.cells_df = self.loader.get_cell_bounding_boxes(self.cells_df)

        # Remove cells that are too close to edges
        self.cells_df = self._filter_edge_cells()

        print(f"Dataset initialized with {len(self.cells_df)} cells")

    def _filter_edge_cells(self) -> pd.DataFrame:
        """Filter out cells too close to tile edges."""
        if not self.use_bounding_box:
            # Use the same edge filtering as original data_loader
            y_half, x_half = (d // 2 for d in self.patch_size)

            with open_ome_zarr(
                self.loader.dataset.store_paths["pheno_assembled_v3"], mode="r"
            ) as store:
                sample_tile = self.cells_df["tile_pheno"].iloc[0]
                array = store[sample_tile][0][0, :, 0, :, :]

                mask_2d = np.all(array > 0, axis=0)
                ys, xs = np.nonzero(mask_2d)
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()

                y_range = (y_min + y_half, y_max - y_half)
                x_range = (x_min + x_half, x_max - x_half)

                filtered_df = self.cells_df[
                    self.cells_df["x_local_pheno"].between(
                        *x_range, inclusive="neither"
                    )
                    & self.cells_df["y_local_pheno"].between(
                        *y_range, inclusive="neither"
                    )
                ]

                return filtered_df
        else:
            # For bounding box mode, we could add more sophisticated edge filtering
            # For now, just return all cells
            return self.cells_df

    def __len__(self):
        return len(self.cells_df)

    def __getitem__(self, idx):
        cell_row = self.cells_df.iloc[idx]

        cell_data = self.loader.extract_cell_data(
            cell_row,
            include_mask=True,
            normalize=self.normalize,
            use_bounding_box=self.use_bounding_box,
        )

        # Prepare batch dictionary
        batch = {
            "data": cell_data["image"],
            "mask": cell_data.get(
                "mask",
                np.zeros((cell_data["image"].shape[1], cell_data["image"].shape[2])),
            ),
            "metadata": cell_data["metadata"],
            "bounding_box": cell_data["bounding_box"],
        }

        if self.transform is not None:
            batch = self.transform(batch)

        return batch
