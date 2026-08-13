"""Tile splitting utilities for array partitioning."""

from typing import List, Tuple


def split_into_tiles(arr_shape: Tuple, n: int, overlap: int) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
    """Divide a 2D array into n x n overlapping tiles.

    Parameters
    ----------
    arr_shape : tuple
        (height, width) of the array.
    n : int
        Number of tiles per dimension (n x n grid).
    overlap : int
        Pixel overlap between adjacent tiles.

    Returns
    -------
    tiles : list of (row_start, row_stop, col_start, col_stop)
    indices : list of (i, j) grid positions
    """
    tiles = []
    height, width = arr_shape
    tile_height = height // n
    tile_width = width // n

    row_stride = tile_height - overlap
    col_stride = tile_width - overlap
    index = []
    for i in range(n):
        row_start = i * row_stride
        row_stop = row_start + tile_height
        if row_stop > height:
            row_stop = height

        for j in range(n):
            col_start = j * col_stride
            col_stop = col_start + tile_width
            if col_stop > width:
                col_stop = width
            index.append((i, j))
            tiles.append((row_start, row_stop, col_start, col_stop))

    return tiles, index
