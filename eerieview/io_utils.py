import shutil
import tempfile
from pathlib import Path

import xarray
from dask.diagnostics import ProgressBar

from eerieview.logger import get_logger

logger = get_logger(__name__)


def safe_to_netcdf(
    dataset: xarray.Dataset,
    output_path: Path,
    encoding: dict | None = None,
    show_progress: bool = False,
):
    """Safely write a netCDF so it does not remain incomplete in case of an
    interruption.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output_path = Path(tmpdir, output_path.name)
        logger.info(f"Writing netCDF to {temp_output_path}")
        if show_progress:
            with ProgressBar():
                dataset.to_netcdf(temp_output_path, encoding=encoding)
        else:
            dataset.to_netcdf(temp_output_path, encoding=encoding)
        logger.info(f"Write complete. Moving {temp_output_path} to {output_path}")
        shutil.move(temp_output_path, output_path)
