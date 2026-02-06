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

        # ProgressBar only works with the local scheduler.
        # It can interfere with distributed, so we check if a client is active.
        use_progress = show_progress
        if use_progress:
            try:
                from distributed import get_client

                get_client()
                logger.debug("Distributed client detected, disabling ProgressBar")
                use_progress = False
            except (ImportError, ValueError):
                pass

        if use_progress:
            with ProgressBar():
                dataset.to_netcdf(temp_output_path, encoding=encoding)
        else:
            dataset.to_netcdf(temp_output_path, encoding=encoding)
        logger.info(f"Write complete. Moving {temp_output_path} to {output_path}")
        shutil.move(temp_output_path, output_path)


def safe_to_zarr(
    dataset: xarray.Dataset,
    output_path: Path,
    encoding: dict | None = None,
    show_progress: bool = False,
):
    """Safely write a Zarr store so it does not remain incomplete in case of
    an interruption.
    """
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_output_path.exists():
        shutil.rmtree(temp_output_path)

    # Filter encoding for Zarr: NetCDF keys are invalid
    zarr_encoding = None
    if encoding:
        zarr_encoding = {}
        for var, enc in encoding.items():
            z_enc = enc.copy()
            # NetCDF-specific keys that Zarr doesn't support
            # We also remove chunks/chunksizes to avoid alignment issues with Dask
            for key in ["zlib", "complevel", "shuffle", "chunksizes", "chunks"]:
                z_enc.pop(key, None)
            zarr_encoding[var] = z_enc

    logger.info(f"Writing Zarr to {temp_output_path}")

    use_progress = show_progress
    if use_progress:
        try:
            from distributed import get_client

            get_client()
            logger.debug("Distributed client detected, disabling ProgressBar")
            use_progress = False
        except (ImportError, ValueError):
            pass

    if use_progress:
        with ProgressBar():
            dataset.to_zarr(temp_output_path, encoding=zarr_encoding, consolidated=True)
    else:
        dataset.to_zarr(temp_output_path, encoding=zarr_encoding, consolidated=True)

    logger.info(f"Write complete. Renaming {temp_output_path} to {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
    temp_output_path.rename(output_path)
