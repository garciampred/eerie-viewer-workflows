"""Get the monthly observed EKE and Sea level from AVISO data."""

import os
from pathlib import Path

import dask
import xarray
from dotenv import load_dotenv

from eerieview.eke import DEFAULT_ENCODING, compute_monthly_eke
from eerieview.io_utils import safe_to_netcdf, safe_to_zarr


# Crear el cluster Dask ANTES de cargar datos

load_dotenv()

def main_eke():
    dask.config.set(scheduler="synchronous")
    storage_dir = Path(os.environ["DOWNLOADIR"], "aviso")

    daily_anom_zos_file = Path(storage_dir, "zos_anom_aviso_daily.nc")
    aviso_daily_zos_file = Path(storage_dir, "adt_aviso_daily.nc")

    dataset = xarray.open_dataset(
        aviso_daily_zos_file,
        chunks=dict(time=1000, latitude=100, longitude=100),
    ).rename(adt="zos", longitude="lon", latitude="lat")

    zos_daily_climatology_file = Path(storage_dir, "zos_clim_aviso_dayofyear.nc")

    eke_monthly = compute_monthly_eke(
        dataset,
        daily_anom_zos_file,
        zos_daily_climatology_file,
    )

    timeindex = eke_monthly.time.to_index()
    mintime = f"{timeindex[0]:%Y%m}"
    maxtime = f"{timeindex[-1]:%Y%m}"
    output_path = Path(storage_dir, f"eke_AVISO_mon_{mintime}-{maxtime}.zarr")

    safe_to_zarr(eke_monthly, output_path, encoding=dict(eke=DEFAULT_ENCODING))


def main_zos():
    storage_dir = Path(os.environ["DOWNLOADIR"], "aviso")

    aviso_daily_zos_file = Path(storage_dir, "adt_aviso_daily.nc")
    aviso_monthly_zos_file = Path(storage_dir, "zos_AVISO_mon_199301-202206.nc")

    dataset = xarray.open_dataset(
        aviso_daily_zos_file,
        chunks=dict(time=1000, latitude=100, longitude=100),
    ).rename(adt="zos", longitude="lon", latitude="lat")

    dataset_monthly = dataset.resample(time="MS").mean()

    safe_to_netcdf(
        dataset_monthly,
        aviso_monthly_zos_file,
        encoding=dict(zos=DEFAULT_ENCODING),
        show_progress=True,
    )


def main_eke_to_netcdf():
    storage_dir = Path(os.environ["DOWNLOADIR"], "aviso")
    obsdir = Path(os.environ["OBSDIR"])

    zarr_path = Path(storage_dir, "eke_AVISO_mon_199301-202206.zarr")
    dataset = xarray.open_zarr(zarr_path)

    timeindex = dataset.time.to_index()
    mintime = f"{timeindex[0]:%Y%m}"
    maxtime = f"{timeindex[-1]:%Y%m}"
    output_path = Path(obsdir, f"eke_AVISO_mon_{mintime}-{maxtime}.nc")

    safe_to_netcdf(dataset, output_path, encoding=dict(eke=DEFAULT_ENCODING), show_progress=True)


if __name__ == "__main__":
    #main_eke()
    main_eke_to_netcdf()

