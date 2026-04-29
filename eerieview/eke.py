"""
Function to compute the Eddy Kinetic Energy from the sea level using the geostrophic
equations.

Code by Aaron Wienkers with modifications from Markel García-Díez and Ignasi Vallès-Casanova
"""

from pathlib import Path

import numpy
import xarray

from eerieview.io_utils import safe_to_zarr
from eerieview.logger import get_logger

logger = get_logger(__name__)
DEFAULT_ENCODING = dict(
    zlib=True, complevel=1, shuffle=True, chunksizes=(100, 100, 100)
)


def rolling_smooth_annual_cycly(da: xarray.DataArray) -> xarray.DataArray:
    """Compute a smoothed rolling annual climatology.

    Step 1: 21-year rolling mean per day-of-year (center=True, min_periods=11).
    Step 2: 5-day temporal smoothing (center=True, min_periods=3).

    Uses sliding_window_view for a fully-vectorized step 1 — no Python loop
    over years, one nanmean call per Day Of Year (365 DOYs).
    """
    import pandas
    from numpy.lib.stride_tricks import sliding_window_view

    if da.isnull().all():
        return da

    arr = da.values.astype(float)
    original_shape = arr.shape
    n_time = original_shape[0]
    spatial_shape = original_shape[1:]

    doys = pandas.DatetimeIndex(da.time.values).dayofyear  # 1-indexed
    clim = numpy.full_like(arr, numpy.nan)

    # ── Step 1: 21-year rolling mean per DOY (vectorized over years) ──────
    half = 10         # window = 2*half+1 = 21
    min_periods = 11  # matches original decadal_window_size // 2 + 1

    nan_pad = numpy.full((half, *spatial_shape), numpy.nan)
    for doy in range(1, 366):
        idx = numpy.where(doys == doy)[0]
        if len(idx) == 0:
            continue
        sub = arr[idx]  # (n_years, ...)
        # NaN-pad both ends so the window is centered (no edge shrinkage)
        padded = numpy.concatenate([nan_pad, sub, nan_pad], axis=0)
        # windows: (n_years, ..., 21) — zero-copy view
        windows = sliding_window_view(padded, window_shape=21, axis=0)
        count = numpy.sum(~numpy.isnan(windows), axis=-1)
        with numpy.errstate(invalid="ignore"):
            mean_val = numpy.nanmean(windows, axis=-1)
        mean_val[count < min_periods] = numpy.nan
        clim[idx] = mean_val

    # ── Step 2: 5-day rolling smooth along time ───────────────────────────
    flat = clim.reshape(n_time, -1)
    smoothed_flat = (
        pandas.DataFrame(flat).rolling(5, min_periods=3, center=True).mean().to_numpy()
    )
    return da.copy(data=smoothed_flat.reshape(original_shape))


def _fast_time_coord(da: xarray.DataArray) -> numpy.ndarray:
    """Return the time coordinate reading only the first two values.
       Falls back to reading all values if the cadence is not strictly 1 day.
    """
    import pandas

    t0 = da.time[0].values
    t1 = da.time[1].values
    if t1 - t0 != numpy.timedelta64(1, "D"):
        return da.time.values
    return pandas.date_range(pandas.Timestamp(t0), periods=len(da.time), freq="D").values


def _init_zarr_store(
    da: xarray.DataArray, path: Path, chunk_size: int, varname: str
) -> None:
    """Create an empty zarr store using the zarr API directly.

    Avoids xarray's to_zarr(compute=False) + dask.array.empty which writes
    thousands of empty chunk files (one per zarr chunk), each a metadata op.
    The zarr API creates only the metadata files (.zgroup, .zarray, .zattrs).
    """
    import json
    import shutil

    import zarr

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    chunk_tuple = (500, chunk_size, chunk_size)
    n_time, n_lat, n_lon = len(da.time), len(da.lat), len(da.lon)

    root = zarr.open_group(str(path), mode="w")

    # Data variable (no data written — missing chunks return fill_value=NaN)
    root.create_dataset(
        varname,
        shape=(n_time, n_lat, n_lon),
        chunks=chunk_tuple,
        dtype=da.dtype,
        fill_value=float("nan"),
    )
    root[varname].attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time", "lat", "lon"],
        }
    )

    # Coordinate arrays
    time_vals = _fast_time_coord(da).astype("int64")  # ns since epoch
    root.create_dataset("time", data=time_vals, dtype="int64")
    root["time"].attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time"],
            "units": "nanoseconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
        }
    )
    root.create_dataset("lat", data=da.lat.values)
    root["lat"].attrs["_ARRAY_DIMENSIONS"] = ["lat"]
    root.create_dataset("lon", data=da.lon.values)
    root["lon"].attrs["_ARRAY_DIMENSIONS"] = ["lon"]

    # Group-level xarray metadata
    root.attrs["_ARRAY_DIMENSIONS"] = []  # not needed at group level
    zarr.consolidate_metadata(str(path))


def write_clim_and_anom(
    da: xarray.DataArray,
    clim_file: Path,
    anom_file: Path,
    chunk_size: int = 50,
    num_workers: int | None = None,
) -> None:
    """Write rolling climatology and daily anomalies in a single spatial-block pass.

    Reads each spatial block once, computes the
    21-year rolling climatology and the anomaly, then writes both to their
    respective zarr stores. This avoids the alternative of writing clim first
    and then re-reading it for the subtraction.
    """
    import gc

    lat_size = len(da.lat)
    lon_size = len(da.lon)
    clim_tmp = clim_file.with_suffix(clim_file.suffix + ".tmp")
    anom_tmp = anom_file.with_suffix(anom_file.suffix + ".tmp")

    logger.info(f"Initialising clim zarr store at {clim_tmp}")
    _init_zarr_store(da, clim_tmp, chunk_size, varname="zos")
    logger.info(f"Initialising anom zarr store at {anom_tmp}")
    _init_zarr_store(da, anom_tmp, chunk_size, varname="zos")

    n_lat = (lat_size + chunk_size - 1) // chunk_size
    n_lon = (lon_size + chunk_size - 1) // chunk_size
    n_total = n_lat * n_lon
    block = 0
    for lat_start in range(0, lat_size, chunk_size):
        lat_sl = slice(lat_start, min(lat_start + chunk_size, lat_size))
        for lon_start in range(0, lon_size, chunk_size):
            lon_sl = slice(lon_start, min(lon_start + chunk_size, lon_size))
            block += 1
            logger.info(f"Clim+anom block {block}/{n_total}")

            raw = da.isel(lat=lat_sl, lon=lon_sl).chunk({"time": -1}).compute(
                scheduler="threads", num_workers=num_workers
            )
            clim = rolling_smooth_annual_cycly(raw)
            anom = (raw - clim).to_dataset()
            clim_ds = clim.to_dataset()

            # Drop coords already written during store initialisation
            write_kwargs = dict(consolidated=False)
            region = {"lat": lat_sl, "lon": lon_sl}
            clim_ds.drop_vars(list(clim_ds.coords)).to_zarr(
                clim_tmp, region=region, **write_kwargs
            )
            anom.drop_vars(list(anom.coords)).to_zarr(
                anom_tmp, region=region, **write_kwargs
            )
            del raw, clim, clim_ds, anom
            gc.collect()

    logger.info(f"Renaming {clim_tmp} → {clim_file}")
    clim_tmp.rename(clim_file)
    logger.info(f"Renaming {anom_tmp} → {anom_file}")
    anom_tmp.rename(anom_file)


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
    num_workers: int | None = None,
) -> xarray.Dataset:
    """Compute monthly Eddy Kinetic Energy from the sea level.

    It will persist two files to ease memory pressure.
    """
    if not daily_anom_zos_file.exists():
        write_clim_and_anom(
            dataset.zos,
            zos_daily_climatology_file,
            daily_anom_zos_file,
            num_workers=num_workers,
        )
    zos_daily_anom = xarray.open_zarr(
        daily_anom_zos_file, chunks=dict(time=500, lon=-1, lat=-1)
    ).zos
    # Compute Geostrophic Velocities
    u_g, v_g = compute_geostrophic_velocities(zos_daily_anom, latlon_units="degrees")
    eke = 0.5 * (u_g**2 + v_g**2)
    # Compute Full Time-Mean EKE
    nan_mask = dataset.zos.isel(time=0).notnull().squeeze().compute()
    nan_mask.loc[dict(lat=slice(-3, 3))] = 0
    eke_monthly = (
        eke.resample(time="MS")
        .mean()
        .where(nan_mask)
        .transpose("time", "lat", "lon")
        .to_dataset(name="eke")
        .chunk({"time": 120, "lat": 180, "lon": 360})
    )
    return eke_monthly
