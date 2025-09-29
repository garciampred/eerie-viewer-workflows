from datetime import datetime
from functools import reduce
from pathlib import Path

import dask
import numpy
import xarray

from eerieview.cdo import cdo_regrid
from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import futuremember2hist
from eerieview.data_access import get_entry_dataset, get_main_catalogue
from eerieview.data_models import (
    DecadalProduct,
    EERIEMember,
    InputLocation,
    PeriodsConfig,
    TimeFilter,
)
from eerieview.data_processing import (
    add_anomalies,
    aggregate_period,
    aggregate_regions,
    aggtime,
    define_extra_dimensions,
    delete_wrong_attrs,
    filter_time_axis,
    fix_coords,
    fix_units,
    get_time_filters,
    rename_realm,
    retry_get_entry_with_fixes,
    slice_period,
)
from eerieview.exceptions import EmptySliceError
from eerieview.grids import get_grid_dataset
from eerieview.io_utils import safe_to_netcdf
from eerieview.logger import get_logger
from eerieview.metadata import fix_attributes
from eerieview.trends.api import compute_trend

logger = get_logger(__name__)


def get_decadal_product(
    dataset: xarray.Dataset,
    period: tuple[int, int],
    product: DecadalProduct,
    varname: str,
) -> xarray.Dataset:
    if product == "clim":
        dataset_product = aggregate_period(dataset, period)
    elif product == "trend":
        ds_period = slice_period(dataset, period)
        if "lon" in ds_period.dims:
            chunks = dict(time=-1, lon=100, lat=100)
        elif "longitude" in ds_period.dims:
            chunks = dict(time=-1, longitude=100, latitude=100)
        elif "ncells" in ds_period.dims:
            chunks = dict(time=-1, ncells=300)
        else:
            chunks = dict(time=-1, value=300)
        dataset_yearly = ds_period.resample(time="YS").mean().chunk(chunks)
        datarray_clim, pvalue = compute_trend(varname, dataset_yearly, as_decadal=True)
        dataset_product = datarray_clim.to_dataset(name=varname)
        dataset_product[varname + "_pvalue"] = pvalue
    else:
        raise RuntimeError(f"Unknown product type {product}")
    return dataset_product


def get_time_series(
    dataset_cmor_filtered: xarray.Dataset,
    time_filter: TimeFilter,
    varname: str,
    region_set: str,
) -> xarray.Dataset:
    dataset_cmor_resampled = aggtime(
        dataset_cmor_filtered,
        time_filter.units,
        numpy.nanmean,
        minvalues=time_filter.get_minvalues("monthly"),
        varname=varname,
    )
    varattrs = dataset_cmor_filtered[varname].attrs
    dataset_ts = aggregate_regions(dataset_cmor_resampled, region_set).compute()
    # Set time to the first day of year year so all the time filters are aligned
    dataset_ts["time"] = [datetime(d.year, 1, 1) for d in dataset_ts.time.to_index()]
    dataset_ts[varname].attrs.update(varattrs)
    return dataset_ts


def get_model_decadal_product(
    varname: str,
    location: InputLocation,
    output_dir: Path,
    members: list[str],
    periods: PeriodsConfig,
    product: DecadalProduct = "clim",
    experiment: str = "control",
    clobber: bool = False,
    get_entry_dataset_fun=get_entry_dataset,
) -> Path:
    """Compute and save decadal products (e.g., climatologies, trends) for a given variable.

    Parameters
    ----------
    varname : str
        The name of the variable to process (e.g., 'tas', 'pr').
    location : InputLocation
        The location where input data is stored ('levante', 'cloud', etc.).
    output_dir : Path
        The directory where the output NetCDF file will be saved.
    members : list[str]
        A list of model members (e.g., 'eerie-control-1', 'eerie-hist-1').
    periods : PeriodsConfig
        Configuration for time periods, including reference and analysis periods.
    product : DecadalProduct, optional
        The type of product to compute ('clim', 'trend'). By default 'clim'.
    experiment : str, optional
        The experiment name (e.g., 'control', 'hist', 'hist-amip'). By default 'control'.
    clobber : bool, optional
        If True, overwrite existing output files. By default False.
    get_entry_dataset_fun : callable, optional
        Function to retrieve the initial dataset. By default `get_entry_dataset`.

    Returns
    -------
    Path
        The path to the generated output NetCDF file.
    """
    # Get time filtering configurations
    time_filters = get_time_filters()
    output_path = Path(output_dir, f"{varname}_{experiment}_EERIE_{product}.nc")
    if output_path.exists() and not clobber:
        logger.info(f"{output_path} already exists, refusing to overwrite.")
        return output_path

    # Get the main data catalogue
    catalogue = get_main_catalogue()

    climatologies = []

    # Iterate over each model member
    for member_str in members:
        # Rename the member string based on variable realm
        member = rename_realm(member_str, varname)
        # Get the raw variable name from the CMOR mapping
        rawname = get_raw_variable_name(member, varname)
        if "future" in member:
            dataset_future, member, rawname = _get_member(
                catalogue, get_entry_dataset_fun, location, member, rawname, varname
            )
            member_hist = futuremember2hist[member_str]
            member_hist = rename_realm(member_hist, varname)
            dataset_hist, _ , rawname = _get_member(
                catalogue,
                get_entry_dataset_fun,
                location,
                member_hist,
                rawname,
                varname,
            )
            dataset = xarray.concat([dataset_hist, dataset_future], dim="time")
        else:
            dataset, member, rawname = _get_member(
                catalogue, get_entry_dataset_fun, location, member, rawname, varname
            )

        # Squeeze out singleton dimensions
        dataset = dataset.squeeze()
        dataset_cmor = to_cmor_names(dataset, rawname, varname)
        # Fix units if necessary (e.g., K to degC, m/s to mm/day)
        dataset_cmor = fix_units(dataset_cmor, varname, product)

        # Get the standardized member slug for dimension naming
        final_member = EERIEMember.from_string(member.replace("ocean", "atmos")).slug

        # Iterate through each time filter and period
        for time_filter in time_filters:
            dataset_cmor_filtered = filter_time_axis(dataset_cmor, time_filter)
            for period in periods.all_list:
                logger.info(
                    f"Computing {product} for {varname} {member} {period} and "
                    f"{time_filter}"
                )
                # Get the product (e.g., climatology) or fill with NaN if empty
                dataset_product = get_decadal_product_or_fill_with_nan(
                    dataset_cmor_filtered, period, product, varname
                )
                # Fix coordinates attributes (e.g., units, bounds)
                dataset_product = fix_coords(dataset_cmor_filtered, dataset_product)
                # Remove problematic attributes from the dataset
                dataset_product = delete_wrong_attrs(dataset_product)

                # Regrid the dataset based on member type (native grid or common grid)
                if "native" in member:
                    dataset_product = cdo_regrid(dataset_product, member)
                else:
                    grid_dataset = get_grid_dataset(
                        0.25
                    )  # Get a 0.25 degree target grid
                    dataset_product = dataset_product.regrid.conservative(
                        grid_dataset, latitude_coord="lon"
                    )

                # Define extra dimensions for metadata (member, time filter, period)
                dataset_product = define_extra_dimensions(
                    dataset_product,
                    final_member,
                    time_filter,
                    period,
                ).drop_vars(["depth"], errors="ignore")
                climatologies.append(dataset_product)

    # Merge all individual climatologies/products into a single dataset
    # Using reduce for efficient merging of multiple xarray Datasets
    final_dataset = reduce(lambda x, y: xarray.merge([x, y]), climatologies)
    # Anomalies
    if product == "clim":
        final_dataset = add_anomalies(final_dataset, periods, varname)

    # Fix global and variable attributes for compliance
    final_dataset = fix_attributes(final_dataset, varname)

    # Optimize Dask graph and write the final dataset to NetCDF
    logger.info(f"Writing climatologies to {output_path}")
    final_dataset = dask.optimize(final_dataset)[0]
    safe_to_netcdf(final_dataset, output_path, show_progress=True)
    return output_path


def _get_member(catalogue, get_entry_dataset_fun, location, member, rawname, varname):
    try:
        # Attempt to retrieve the dataset for the current member and variable
        dataset = get_entry_dataset_fun(catalogue, member, rawname, location=location)
    except KeyError:
        # If a KeyError occurs, retry with common fixes
        dataset, member, rawname = retry_get_entry_with_fixes(
            catalogue, get_entry_dataset_fun, location, member, rawname, varname
        )
    # Handle realization dimension if present by averaging
    if "realization" in dataset:
        logger.info("Realization dimension detected. Averaging the ensemble members.")
        dataset = dataset.mean(dim="realization")
    return dataset, member, rawname


def get_decadal_product_or_fill_with_nan(
    dataset_cmor_filtered: xarray.Dataset,
    period: tuple[int, int],
    product: DecadalProduct,
    varname: str,
) -> xarray.Dataset:
    """Retrieve the decadal product for a given period, or fill with NaN if the slice is empty."""
    try:
        # Attempt to get the product (e.g., climatology or trend)
        dataset_product = get_decadal_product(
            dataset_cmor_filtered, period, product, varname
        )
    except EmptySliceError:
        logger.warning(
            f"Empty slice detected for {varname} during period {period}. Filling with NaN."
        )
        # If an EmptySliceError occurs, create a NaN-filled dataset with appropriate dimensions
        dataset_product = (
            dataset_cmor_filtered.copy().isel(time=0).squeeze().drop_vars("time")
        )
        dataset_product[varname][:] = numpy.nan
    return dataset_product
