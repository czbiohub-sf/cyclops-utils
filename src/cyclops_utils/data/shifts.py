"""Utilities for reading tile registration shifts."""

from collections import defaultdict

from stitch.connect import read_shifts_biahub


def read_shifts(dataset, config_key: str, well: str):
    """Read calibration shifts for a specific well from a stitching config.

    Parameters
    ----------
    dataset : OpsDataset
        Dataset instance providing config_paths.
    config_key : str
        Config key identifying which stitching config to use.
    well : str
        Well identifier (e.g. "A/1/0").

    Returns
    -------
    dict
        Mapping of tile names to (y_shift, x_shift) tuples.
    """
    all_shifts = read_shifts_biahub(dataset.config_paths[config_key])
    grouped_shifts = defaultdict(dict)
    for key, value in all_shifts.items():
        group = key.split("/")[1]
        grouped_shifts[group][key] = value
    return grouped_shifts[well[2]]
