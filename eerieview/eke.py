"""
Function to compute the Eddy Kinetic Energy from the sea level using the geostrophic
equations.

Code by Aaron Wienkers with modifications from Markel García-Díez
"""

from pathlib import Path

import numpy
import xarray

from eerieview.io_utils import safe_to_netcdf
from eerieview.logger import get_logger

logger = get_logger(__name__)
DEFAULT_ENCODING = dict(
    zlib=True, complevel=1, shuffle=True, chunksizes=(100, 100, 100)
)


def rolling_smooth_annual_cycly(da: xarray.DataArray) -> xarray.DataArray:
    if da.isnull().all():
        return da
    decadal_window_size = 21  # years
    smooth_window_len = 5  # daysa
    print(da)
    # Rolling average for each day of year, with some years around
    da_rolling_clims = da.groupby("time.dayofyear").map(
        lambda x: x.rolling(
            time=decadal_window_size,
            min_periods=decadal_window_size // 2,
            center=True,
        ).mean()
    )
    # Rolling average for each time with a few days around
    da_rolling_clims_smoothed = da_rolling_clims.rolling(
        time=smooth_window_len, min_periods=smooth_window_len // 2, center=True
    ).mean().load()
    return da_rolling_clims_smoothed


def remove_smooth_climatology(da: xarray.DataArray, da_clim_file: Path):
    """Compute daily Climatology & Smooth.

    We follow https://www.nature.com/articles/s41558-022-01478-3
    21 year moving average for each day of year + 5 day moving average in time for
    smoothing.
    """
    if not da_clim_file.exists():
        # 21-year moving average for each dayofyear (assuming 'time' is daily data)
        # We must ensure that each block has the full time series for the rolling mean to work.
        # Otherwise, map_blocks will return NaNs at the borders of time chunks.
        # We also use smaller spatial blocks here to avoid OOM, as time=-1 can be very large.
        da_full_time = da.chunk(dict(time=-1, lat=50, lon=50))
        print(da_full_time)
        da_dayofyear_rolling_clim = xarray.map_blocks(
            rolling_smooth_annual_cycly, da_full_time, template=da_full_time
        )
        print(da_dayofyear_rolling_clim)
        safe_to_netcdf(
            da_dayofyear_rolling_clim.to_dataset(),
            da_clim_file,
            encoding=dict(zos=DEFAULT_ENCODING),
            show_progress=True,
         )
    else:
        logger.info(f"Reading {da_clim_file}")
        # Use the same chunks as the input data to avoid expensive rechunking
        # when calculating da - da_clim
        da_dayofyear_rolling_clim = xarray.open_dataset(
            da_clim_file, chunks=dict(time=1000, lon=100, lat=100)
        ).zos

    # Remove Rolling daily Climatology from Signal
    da_detrend = da - da_dayofyear_rolling_clim
    return da_detrend


def compute_geostrophic_velocities(
    ssh: xarray.DataArray, latlon_units: str = "degrees"
) -> tuple[xarray.DataArray, xarray.DataArray]:
    # Constants
    Omega = 7.2921e-5  # Earth's rotation rate (rad/s)
    g = 9.81  # Gravitational acceleration (m/s²)
    R_earth = 6371.0e3  # Earth's radius (m)

    # Create periodic boundary in longitude using xr.pad
    padded_ssh = (
        (ssh.pad(lon=(1, 1), mode="wrap"))
        .pad(lat=(1, 1), mode="wrap")
        .chunk({"lat": -1, "lon": -1})
    )

    # Compute SSH gradients (2nd order centered differences), _relative to lat/lon UNITS_
    dssh_dlon = (
        padded_ssh.differentiate("lon").isel(lat=slice(1, -1)).isel(lon=slice(1, -1))
    )
    dssh_dlat = (
        padded_ssh.differentiate("lat").isel(lat=slice(1, -1)).isel(lon=slice(1, -1))
    )

    ## Convert to Cartesian gradients:  i.e. dssh/dx = (dssh/dlon) * (dlon/dx)
    lat = ssh.lat
    if latlon_units == "degrees":  # Convert lat to radians
        lat = numpy.deg2rad(lat)

    dx_dlon = R_earth * numpy.cos(lat)  # _Units of metre / radian_
    dy_dlat = R_earth  # _Units of metre / radian_

    if latlon_units == "degrees":
        dx_dlon = dx_dlon * numpy.pi / 180.0  # _Units of metre / degree_
        dy_dlat = dy_dlat * numpy.pi / 180.0  # _Units of metre / degree_

    dssh_dx = dssh_dlon / dx_dlon
    dssh_dy = dssh_dlat / dy_dlat

    f = 2.0 * Omega * numpy.sin(lat)

    # Compute geostrophic velocities
    u_g = -(g / f) * dssh_dy
    v_g = (g / f) * dssh_dx

    return u_g, v_g


def compute_monthly_eke(
    dataset: xarray.Dataset,
    daily_anom_zos_file: Path,
    zos_daily_climatology_file: Path,
) -> xarray.Dataset:
    """Compute monthly Eddy Kinetic Energy from the sea level.

    It will persist two files to ease memory pressure.
    """
    if not daily_anom_zos_file.exists():
        zos_daily_anom = remove_smooth_climatology(
            dataset.zos,
            zos_daily_climatology_file,
        )
        print(zos_daily_anom)
        safe_to_netcdf(
            zos_daily_anom.to_dataset(),
            daily_anom_zos_file,
            encoding=dict(zos=DEFAULT_ENCODING),
            show_progress=True,
        )
    else:
        zos_daily_anom = xarray.open_dataset(
            daily_anom_zos_file, chunks=dict(time=10, lon=-1, lat=-1)
        ).zos
    # Compute Geostrophic Velocities
    u_g, v_g = compute_geostrophic_velocities(zos_daily_anom, latlon_units="degrees")
    eke = 0.5 * (u_g**2 + v_g**2)
    # Compute Full Time-Mean EKE
    nan_mask = dataset.zos.isel(time=0).notnull().squeeze()
    nan_mask.loc[dict(lat=slice(-3, 3))] = 0
    eke_monthly = (
        eke.resample(time="MS")
        .mean()
        .where(nan_mask)
        .transpose("time", "lat", "lon")
        .to_dataset(name="eke")
    )
    return eke_monthly
