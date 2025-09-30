"""Get the monthly observed EKE and Sea level from AVISO data."""

import os
from pathlib import Path

import xarray
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dotenv import load_dotenv

from eerieview.eke import DEFAULT_ENCODING, compute_monthly_eke
from eerieview.io_utils import safe_to_netcdf

load_dotenv()


def main_eke():
    client = Client()
    print(client)
    print(client.dashboard_link)
    # Monthly eddy kinetic energy.
    storage_dir = Path(os.environ["DOWNLOADIR"], "aviso")
    # TODO: Downloaded by hand from DKRZ, make it reproducible
    daily_anom_zos_file = Path(storage_dir, "zos_anom_aviso_daily.nc")
    aviso_daily_zos_file = Path(storage_dir, "adt_aviso_daily.nc")
    dataset = xarray.open_dataset(
        aviso_daily_zos_file, chunks=dict(time=-1, latitude=100, longitude=100)
    ).rename(adt="zos", longitude="lon", latitude="lat")
    # Compute 30-day rolling daily climatology of SSH
    zos_daily_climatology_file = Path(storage_dir, "zos_clim_aviso_dayofyear.nc")
    zos_daily_climatology_file_intermediate = Path(
        storage_dir, "zos_clim_aviso_dayofyear_intermediate.nc"
    )
    eke_monthly = compute_monthly_eke(
        dataset,
        daily_anom_zos_file,
        zos_daily_climatology_file,
        zos_daily_climatology_file_intermediate,
    )
    # Compute Monthly Mean EKE:  N.B.: Cannot use monthly SSH to do this
    timeindex = eke_monthly.time.to_index()
    mintime = f"{timeindex[0]:%Y%m}"
    maxtime = f"{timeindex[-1]:%Y%m}"
    output_path = Path(storage_dir, f"eke_AVISO_mon_{mintime}-{maxtime}.nc")
    with ProgressBar():
        eke_monthly.to_netcdf(output_path, encoding=dict(eke=DEFAULT_ENCODING))


def main_zos():
    # Monthly sea level, renamed from adt to zos
    storage_dir = Path(os.environ["DOWNLOADIR"], "aviso")
    aviso_daily_zos_file = Path(storage_dir, "adt_aviso_daily.nc")
    aviso_monthly_zos_file = Path(storage_dir, "zos_AVISO_mon_199301-202206.nc")
    dataset = xarray.open_dataset(
        aviso_daily_zos_file, chunks=dict(time=1000, latitude=100, longitude=100)
    ).rename(adt="zos", longitude="lon", latitude="lat")
    dataset_monthly = dataset.resample(time="MS").mean()
    safe_to_netcdf(
        dataset_monthly,
        aviso_monthly_zos_file,
        encoding=dict(zos=DEFAULT_ENCODING),
        show_progress=True,
    )


if __name__ == "__main__":
    main_eke()
