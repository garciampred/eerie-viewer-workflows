import importlib
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Callable, Union

import numpy
import xarray
import xarray_regrid  # noqa: F401 # Imported for its side effects (adds .regrid accessor)

from eerieview.constants import OCEAN_VARIABLES
from eerieview.data_models import EERIEProduct, InputLocation, PeriodsConfig, TimeFilter
from eerieview.exceptions import EmptySliceError
from eerieview.logger import get_logger
from eerieview.regions import SpatialAggregation

logger = get_logger(__name__)


def slice_period(dataset: xarray.Dataset, period: tuple[int, int]) -> xarray.Dataset:
    """Slice a dataset to a specified time period.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    period : tuple[int, int]
        A tuple representing the start and end years (inclusive) of the period.

    Returns
    -------
    xarray.Dataset
        The dataset sliced to the given period.

    Raises
    ------
    EmptySliceError
        If the time slice results in an empty dataset.
    """
    idate = datetime(period[0], 1, 1)
    fdate = datetime(period[1], 12, 31)
    ds_period = dataset.sel(time=slice(idate.isoformat(), fdate.isoformat()))
    if ds_period.time.size == 0:
        raise EmptySliceError(f"Empty time slice detected for {dataset=} and {period=}")
    return ds_period


def aggregate_period(dataset: xarray.Dataset, years: tuple[int, int]) -> xarray.Dataset:
    """Aggregate a dataset by computing the mean over a specified period.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    years : tuple[int, int]
        A tuple representing the start and end years (inclusive) for aggregation.

    Returns
    -------
    xarray.Dataset
        The dataset with the mean computed over the specified period.
    """
    ds_period = slice_period(dataset, years)
    ds_mean = ds_period.mean(dim="time", keep_attrs=True)
    return ds_mean


def fix_360_longitudes(dataset: xarray.Dataset) -> xarray.Dataset:
    """Transform longitude values from (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.

    Returns
    -------
    xarray.Dataset
        The dataset with fixed longitude coordinates.
    """
    lonname = "lon"
    lon = dataset[lonname]
    if lon.max().values > 180 and lon.min().values >= 0:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
        dataset = dataset.reindex(**{lonname: sorted(dataset[lonname])})  # type: ignore
    return dataset


def get_time_filters() -> tuple[TimeFilter, ...]:
    """Retrieve a predefined set of time filters including 'year' and standard seasons.

    Returns
    -------
    tuple[TimeFilter, ...]
        A tuple of `TimeFilter` objects.
    """
    seasons = ["MAM", "JJA", "SON", "DJF"]
    time_filters = tuple(
        chain(
            [
                TimeFilter("year", "year"),
            ],
            [TimeFilter(s, "season") for s in seasons],
        )
    )
    return time_filters


def seltime(
    ids: xarray.Dataset,
    time_coord: str,
    **kwargs: Union[str, list[int]],
) -> xarray.Dataset:
    """Select time steps by groups of years, months, etc.

    Parameters
    ----------
    ids : xarray.Dataset
        The input dataset .
    time_coord : str
        The name of the time coordinate to use.
    **kwargs : str or list[int]
        Time units and their corresponding values for selection (e.g., `year=[2000, 2001]`,
        `season=['JJA']`, `month=[6, 7, 8]`). Supported units: 'year', 'season', 'month', 'day', 'hour'.

    Returns
    -------
    xarray.Dataset
        The dataset filtered by the specified time constraints.

    Raises
    ------
    KeyError
        If an unsupported time unit is provided in `kwargs`.
    """
    ntimes: int = len(ids.coords[time_coord])
    time_mask = numpy.repeat(True, ntimes)

    for time_unit, time_values in kwargs.items():
        if time_unit not in ("year", "season", "month", "day", "hour"):
            raise KeyError(f"Time unit '{time_unit}' not supported.")
        time_str = f"{time_coord}.{time_unit}"
        time_mask_temp = numpy.in1d(ids[time_str], time_values)
        time_mask = numpy.logical_and(time_mask, time_mask_temp)
    ods = ids.isel(**{time_coord: time_mask})  # type: ignore
    return ods


def filter_time_axis(
    dataset: xarray.Dataset, time_filter: TimeFilter
) -> xarray.Dataset:
    """Filter the time axis of a dataset based on a given TimeFilter configuration.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    time_filter : TimeFilter
        The time filter configuration to apply.

    Returns
    -------
    xarray.Dataset
        The dataset with its time axis filtered.

    Raises
    ------
    RuntimeError
        If the `time_filter.units` are not supported.
    """
    if time_filter.units == "season":
        dataset = seltime(dataset, "time", season=time_filter.freq)  # type: ignore
    elif time_filter.units == "month":
        dataset = seltime(dataset, "time", month=time_filter.freq)  # type: ignore
    elif time_filter.units == "year":
        pass  # No filtering is needed
    else:
        raise RuntimeError(f"{time_filter.units} are not supported")
    return dataset


def define_extra_dimensions(
    dataset: xarray.Dataset,
    member: str,
    time_filter: TimeFilter,
    period: tuple[int, int] | str,
) -> xarray.Dataset:
    """Define and add extra dimensions (member, period, time_filter) to a dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    member : str
        The name of the member (e.g., 'ERA5', 'eerie-hist-1').
    time_filter : TimeFilter
        The time filter configuration.
    period : tuple[int, int] or str
        The period for the data, either as a (start_year, end_year) tuple or a string (e.g., 'reference').

    Returns
    -------
    xarray.Dataset
        The dataset with the new dimensions added and expanded.
    """
    period_str: str
    if isinstance(period, str):
        period_str = period
    else:
        period_str = f"{period[0]}-{period[1]}"
    dims = ["member", "period", "time_filter"]
    dataset = dataset.assign(
        member=xarray.DataArray([member], dims=["member"]),
        period=xarray.DataArray([period_str], dims=["period"]),
        time_filter=xarray.DataArray([time_filter.to_str()], dims=["time_filter"]),
    ).set_coords(dims)
    for varname in dataset.data_vars:
        dataset[varname] = dataset[varname].expand_dims(dims)
    return dataset


def _get_yearseas(year: int, seas: str, month: int) -> datetime:
    """Determine the datetime for the start of a given season and year."""
    seas2firstmonth = {"DJF": 12, "MAM": 3, "JJA": 6, "SON": 9}
    if month in (1, 2):
        year = year - 1
    firstmonth = seas2firstmonth[seas]
    return datetime(year, firstmonth, 1)


def aggtime(
    dataset: xarray.Dataset,
    freq: str,
    aggfun: Callable,
    minvalues: int | None,
    varname: str | None,
) -> xarray.Dataset:
    """Aggregate a dataset in time, supporting season-aware grouping.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    freq : str
        The frequency to aggregate by ('year', 'season', or 'month').
    aggfun : Callable
        The aggregation function to apply (e.g., `numpy.mean`, `numpy.sum`).
    minvalues : int or None
        If provided, set aggregated values to NaN if the number of non-NaN values
        in a group is below this threshold.
    varname : str or None
        If `minvalues` is set, specifies the variable to apply the `minvalues` check to.
        If None, the check applies to all variables.

    Returns
    -------
    xarray.Dataset
        The time-aggregated dataset.

    Raises
    ------
    RuntimeError
        If the aggregation frequency is not supported.
    """
    year_list = dataset["time.year"].values
    month_list = dataset["time.month"].values

    group_list: list[datetime]
    if freq == "year":
        group_list = [datetime(yy, 1, 1) for yy in year_list]
    elif freq == "month":
        group_list = [datetime(yy, mm, 1) for yy, mm in zip(year_list, month_list)]
    elif freq == "season":
        season_list = dataset["time.season"].values
        group_list = [
            _get_yearseas(yy, ss, mm)
            for yy, ss, mm in zip(year_list, season_list, month_list)
        ]
    else:
        raise RuntimeError(f"Frequency '{freq}' not supported.")

    group_data_array = xarray.DataArray(
        group_list, coords=dict(time=dataset.time), dims="time", name="aggtime"
    )
    gby = dataset.groupby(group_data_array)
    aggregated_dataset = gby.reduce(aggfun, dim="time")
    aggregated_dataset = aggregated_dataset.rename({"aggtime": "time"})

    if minvalues is not None:
        ds_count = gby.count(dim="time").rename({"aggtime": "time"})
        if varname is not None:
            aggregated_dataset[varname] = aggregated_dataset[varname].where(
                ds_count[varname] >= minvalues
            )
        else:
            aggregated_dataset = aggregated_dataset.where(ds_count >= minvalues)

    return aggregated_dataset


def aggregate_regions(
    dataset: xarray.Dataset, region_set: str = "ipcc"
) -> xarray.Dataset:
    """Aggregate a dataset spatially over predefined regions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset with spatial dimensions.
    region_set : str, optional
        The name of the region set to use ('EDDY' or 'IPCC'). Defaults to 'ipcc'.

    Returns
    -------
    xarray.Dataset
        The dataset aggregated by regions.

    Raises
    ------
    KeyError
        If the specified `region_set` is not recognized.
    """
    set2filename = {
        "EDDY": "Eddy-rich-regions.geojson",
        "IPCC": "IPCC-WGI-reference-regions-v4_areas.geojson",
    }
    regions_file = Path(
        str(importlib.resources.files("eerieview")),
        "resources",
        set2filename[region_set],
    )
    logger.info(f"Aggregate in regions from {region_set=}, using file: {regions_file}.")
    sa = SpatialAggregation(dataset, regions_file)
    dataset_regions = sa.compute()
    return dataset_regions


def retry_get_entry_with_fixes(
    catalogue: dict,
    get_entry_dataset_fun,
    location: InputLocation,
    member: str,
    rawname: str,
    varname: str,
) -> tuple[xarray.Dataset, str, str]:
    """Attempt to retrieve a dataset with common fixes if the initial attempt fails."""
    member = member.replace("monthly", "daily")

    # Specific fixes for tasmax/tasmin with fesom
    if varname == "tasmax" and "fesom" in member:
        member = member.replace("avg", "max")
        if "24" not in rawname:
            rawname += "24"  #  Append '24' to rawname for daily maximum
    if varname == "tasmin" and "fesom" in member:
        member = member.replace("avg", "min")
        if "24" not in rawname:
            rawname += "24"  # Append '24' to rawname for daily minimum

    # Retry getting the dataset with the applied fixes
    dataset = get_entry_dataset_fun(catalogue, member, rawname, location=location)
    return dataset, member, rawname


def fix_coords(
    dataset_cmor_filtered: xarray.Dataset, dataset_product: xarray.Dataset
) -> xarray.Dataset:
    """Fix attributes of latitude and longitude coordinates in a dataset."""
    for coord in ["lat", "lon"]:
        # Copy attributes from the original CMOR dataset if the coordinate exists
        if coord in dataset_cmor_filtered:
            dataset_product[coord].attrs = dataset_cmor_filtered[coord].attrs.copy()

        # Convert radian units to degrees for lat/lon if necessary
        if (
            coord in dataset_product
            and "units" in dataset_product[coord].attrs
            and dataset_product[coord].attrs["units"] == "radian"
        ):
            if coord == "lon":
                dataset_product[coord].attrs["units"] = "degrees_east"
            if coord == "lat":
                dataset_product[coord].attrs["units"] = "degrees_north"
    return dataset_product


def delete_wrong_attrs(dataset_product: xarray.Dataset) -> xarray.Dataset:
    """Delete problematic attributes from variables and coordinates in an xarray Dataset."""
    # List of attributes known to cause issues or be irrelevant
    wrong_attrs_to_delete = [
        "chunksizes",
        "complevel",
        "zlib",
        "szip",
        "blosc",
        "contiguous",
        "shuffle",
        "zstd",
        "fletcher32",
        "bzip2",
        "CDI_grid_type",
        "number_of_grid_in_referenc",
    ]
    # Iterate through all data variables and coordinates
    for var in list(dataset_product.data_vars) + list(dataset_product.coords):
        for attr in wrong_attrs_to_delete:
            if attr in dataset_product[var].attrs:
                logger.debug(
                    f"Deleting attribute '{attr}' from variable/coordinate '{var}'"
                )
                del dataset_product[var].attrs[attr]
    return dataset_product


def add_anomalies(
    final_dataset: xarray.Dataset, periods: PeriodsConfig, varname: str
) -> xarray.Dataset:
    """Calculate and add anomaly data to the final dataset based on a reference period."""
    reference_period = periods.reference_period
    # Format the reference period as a string for selection
    ref_period_str = f"{reference_period[0]}-{reference_period[1]}"
    anomaly = final_dataset[varname] - final_dataset[varname].sel(period=ref_period_str)
    final_dataset[varname + "_anom"] = anomaly.astype("float32")
    return final_dataset


def fix_units(
    dataset: xarray.Dataset, varname: str, product: EERIEProduct
) -> xarray.Dataset:
    """Fix units of certain variables to a common standard (e.g., K to degC, m/s to mm/day)."""
    if varname == "pr":
        units = dataset[varname].attrs.get("units", "")
        if units != "mm":
            factor = 86400  # seconds in a day
            # If ICON precipitation is in meters per second, convert to mm/day
            if "m s**-1" in dataset[varname].attrs.get("units", ""):
                factor *= 1000  # meters to millimeters
            dataset[varname] = dataset[varname] * factor
        dataset[varname].attrs["units"] = "mm day-1"  # Set the correct units
    # Convert temperature variables from Kelvin to Celsius, unless it's a trend product
    if (
        varname in ["tasmax", "tasmin", "tas", "tos"]
        and dataset[varname].isel(time=5).max().compute().item() > 200
    ):
        if (
            product != "trend"
        ):  # Trends in K are the same as in degC, no conversion needed
            dataset[varname] = dataset[varname] - 273.15
        dataset[varname].attrs["units"] = "degC"
    return dataset


def rename_realm(member: str, varname: str) -> str:
    """Adjust the member string based on the variable's realm (e.g., atmos to ocean)."""
    # For ocean variables, change 'atmos' to 'ocean' in member string
    if varname in OCEAN_VARIABLES and "amip" not in member:
        member = member.replace("atmos", "ocean")
        # Specific fix for 'ifs-fesom2-sr' ocean data
        if "ifs-fesom2-sr" in member and ("hist" in member or "control" in member):
            member = member.replace("monthly", "daily")
            member += "_1950-2014"
    # Adjust member string for ICON tasmax/tasmin variables
    if "icon" in member and varname in ["tasmax", "tasmin"]:
        extreme = "max" if varname == "tasmax" else "min"
        member = member.replace("monthly_mean", f"daily_{extreme}")
    return member
