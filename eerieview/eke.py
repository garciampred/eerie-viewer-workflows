"""
Function to compute the Eddy Kinetic Energy from the sea level using the geostrophic
equations.

Code by Aaron Wienkers with modifications from Markel García-Díez
"""

from pathlib import Path

import numpy
import xarray

from eerieview.io_utils import safe_to_netcdf

DEFAULT_ENCODING = dict(
    zlib=True, complevel=1, shuffle=True, chunksizes=(100, 100, 100)
)


def remove_smooth_climatology(da: xarray.DataArray, da_clim_file: Path):
    """Compute daily Climatology, Extend Coordinate, & Smooth."""
    if not da_clim_file.exists():
        da_clim = da.groupby("time.dayofyear").mean("time")
        T_climatology_smooth = (
            30  # days,  to smooth daily climatology to detrend the signal
        )
        pad_size = T_climatology_smooth // 2 + 1

        da_padded = xarray.concat(
            [
                da_clim.isel(dayofyear=slice(-pad_size, -1)),
                da_clim,
                da_clim.isel(dayofyear=slice(0, pad_size)),
            ],
            dim="dayofyear",
        )
        dataset_clim = (
            da_padded.rolling(
                dayofyear=T_climatology_smooth + 1, center=True
            )  # This makes it +-15 days including the present day
            .mean()
            .isel(dayofyear=slice(pad_size - 1, -pad_size))
        ).to_dataset()
        safe_to_netcdf(
            dataset_clim,
            da_clim_file,
            encoding=dict(zos=DEFAULT_ENCODING),
            show_progress=True,
        )
    else:
        da_clim = (
            xarray.open_dataset(da_clim_file, chunks=dict(time=-1))
            .persist()
            .to_datarray()
        )

    # Remove Climatology from Signal
    da_detrend = da.groupby("time.dayofyear") - da_clim
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
    dataset: xarray.Dataset, daily_anom_zos_file: Path, zos_daily_climatology_file: Path
) -> xarray.Dataset:
    """Compute monthly Eddy Kinetic Energy from the sea level.

    It will persist two files to ease memory pressure.
    """
    if not daily_anom_zos_file.exists():
        zos_daily_anom = remove_smooth_climatology(
            dataset.zos, zos_daily_climatology_file
        )
        safe_to_netcdf(
            zos_daily_anom,
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
        .chunk(dict(time=-1))
    )
    return eke_monthly
