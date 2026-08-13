"""Shared image utility functions (pure NumPy, no GPU dependencies)."""

import numpy as np


def augment_tile(tile, flipud: bool, fliplr: bool, rot90: int):
    """Augment a tile with flips and rotations.

    Parameters
    ----------
    tile : np.ndarray
        Input image array.
    flipud : bool
        Flip vertically.
    fliplr : bool
        Flip horizontally.
    rot90 : int
        Number of 90-degree rotations.

    Returns
    -------
    np.ndarray
        Augmented tile.
    """
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile
