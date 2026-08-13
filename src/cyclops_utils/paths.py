"""Central storage roots for cyclops_utils / shared OPS storage.

All shared data lives under ``BASE_PATH``, which must be supplied via the
``OPS_BASE_PATH`` environment variable — there is no default, so the library
never silently reads or writes somebody else's storage:

    export OPS_BASE_PATH=/path/to/ops_data

Raw acquisitions live on separate instrument mounts, supplied via
``OPS_INSTRUMENT_ROOT`` (ISS tiles) and ``OPS_DRAGONFLY_ROOT`` (live-cell
Dragonfly acquisitions). Those two are optional and fall back to a location
under ``BASE_PATH``, because :class:`~cyclops_utils.data.experiment.OpsDataset`
builds their paths for every experiment while only the conversion steps read
them.

Existing specific env vars (OPS_OUTPUT_BASE_DIR, OPS_CONFIGS_DIR, ...) still
take precedence where used; they default to locations under BASE_PATH.
"""
import os


def _require(var: str) -> str:
    """Return env var ``var``, or raise with a usable message if it is unset."""
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f"{var} is not set. Point it at your storage root, e.g. "
            f"`export {var}=/path/to/ops_data`."
        )
    return value


BASE_PATH = _require("OPS_BASE_PATH")

# Root holding the raw ISS tif acquisitions, one directory per experiment.
INSTRUMENT_ROOT = os.environ.get("OPS_INSTRUMENT_ROOT", f"{BASE_PATH}/raw/iss")

# Root holding the raw live-cell Dragonfly acquisitions, one directory per
# OPS key (e.g. OPS0141/).
DRAGONFLY_ROOT = os.environ.get("OPS_DRAGONFLY_ROOT", f"{BASE_PATH}/raw/dragonfly")
