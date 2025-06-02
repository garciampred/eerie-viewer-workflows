import importlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import xarray

from eerieview.data_models import EERIEMember
from eerieview.logger import get_logger

logger = get_logger(__name__)

native_grids = {
    "icon-esm-er": {
        "atmos": "/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc",
        "ocean": "/pool/data/ICON/grids/public/mpim/0016/icon_grid_0016_R02B09_O.nc",
    },
    "ifs-fesom2-sr": {
        "ocean": "/path/to/eerie-data-viewer-pipeline/scripts/NG5_griddes_nodes_IFS.nc*"
    },
}


@dataclass
class CDORegridPaths:
    cdo_binary: Path
    native_grid: Path | None
    target_grid_file: Path
    tmpfile_native: Path
    tmpfile_regular: Path
    weights_file: Path


def get_dest_grid(label: str = "0125") -> Path:
    gridfile = Path(
        str(importlib.resources.files("eerieview")),
        f"resources/grid{label}_cdo.txt",
    )
    return gridfile


def cdo_regrid(dataset: xarray.Dataset, member: str, dest_grid="0125"):
    member_obj = EERIEMember.from_string(member)
    cdo_binary = Path(os.environ["CDO_LOCATION"])
    target_grid_file = get_dest_grid(dest_grid)
    try:
        native_grid = Path(native_grids[member_obj.model][member_obj.realm])
    except KeyError:
        native_grid = None
    weights_file = Path(f"{member_obj.model}_{member_obj.realm}_{dest_grid}_weights.nc")
    with tempfile.NamedTemporaryFile() as tmpfile_native:
        with tempfile.NamedTemporaryFile() as tmpfile_regular:
            logger.info(
                f"Writing native grid product to {tmpfile_native.name} to regrid"
            )
            print(dataset.encoding)
            dataset.to_netcdf(tmpfile_native.name)
            regrid_paths = CDORegridPaths(
                cdo_binary,
                native_grid,
                target_grid_file,
                Path(tmpfile_native.name),
                Path(tmpfile_regular.name),
                weights_file,
            )
            if not Path(weights_file).exists():
                get_weights(regrid_paths)
            remap(regrid_paths)
            dataset_regridded = xarray.open_dataset(
                tmpfile_regular.name, chunks=dict(lat=500, lon=500)
            )
    return dataset_regridded


def remap(regrid_paths: CDORegridPaths) -> Path:
    """Regrid from native to regular grid using precomputed weights."""
    cdo_binary = regrid_paths.cdo_binary
    target_grid_file = regrid_paths.target_grid_file
    weights_file = regrid_paths.weights_file
    tmpfile_native = regrid_paths.tmpfile_native
    tmpfile_regular = regrid_paths.tmpfile_regular
    native_grid = regrid_paths.native_grid
    command_part1 = [str(cdo_binary), f"remap,{target_grid_file},{weights_file}"]
    command_part2 = [tmpfile_native, tmpfile_regular]
    if native_grid is None:
        command = command_part1 + command_part2
    else:
        command = (
            command_part1
            + [
                f"-setgrid,{native_grid}",
            ]
            + command_part2
        )
    subprocess.run(command)
    return tmpfile_regular


def get_weights(regrid_paths: CDORegridPaths) -> Path:
    """Get the weightst using conservative regridding or NN if no grid definition is
    available.
    """
    cdo_binary = regrid_paths.cdo_binary
    weights_file = regrid_paths.weights_file
    native_grid = regrid_paths.native_grid
    target_grid_file = regrid_paths.target_grid_file
    tmpfile_native = regrid_paths.tmpfile_native

    logger.info(f"Generating weights in {weights_file}")
    if native_grid is None:
        logger.info("Gridfile not available, generating weights for NN interpolation.")
        command: list[str] = [
            str(cdo_binary),
            f"gennn,{target_grid_file}",
            str(tmpfile_native),
            str(weights_file),
        ]
    else:
        logger.info("Generating weights for conservative interpolation.")
        command = [
            str(cdo_binary),
            f"genycon,{target_grid_file}",
            f"-setgrid,{native_grid}",
            str(tmpfile_native),
            str(weights_file),
        ]
    subprocess.run(command)
    return weights_file
