import copy
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Callable

import dask
import intake
import numpy
import xarray

from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import OCEAN_VARIABLES
from eerieview.data_access import get_entry_dataset, get_main_catalogue
from eerieview.data_models import (
    CmorEerieMember,
    DecadalProduct,
    EERIEMember,
    InputLocation,
    Member,
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
    member_class: type[Member] = EERIEMember,
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
    member_class : Member, optional, default is EERIEMember
        Member class to use.

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
        member_obj = member_class.from_string(member_str)
        # Rename the member string based on variable realm
        if varname in OCEAN_VARIABLES:
            member_obj = member_obj.to_ocean()
            member_str = member_obj.to_string()
        # Get the raw variable name from the CMOR mapping
        if isinstance(member_obj, CmorEerieMember):
            rawname = varname
        else:
            rawname = get_raw_variable_name(member_str, varname)
        dataset, member, rawname = get_complete_input_dataset(
            catalogue,
            get_entry_dataset_fun,
            location,
            member_obj,
            rawname,
            varname,
        )
        # Squeeze out singleton dimensions
        dataset = dataset.squeeze()
        dataset_cmor = to_cmor_names(dataset, rawname, varname)
        # Fix units if necessary (e.g., K to degC, m/s to mm/day)
        dataset_cmor = fix_units(dataset_cmor, varname, product)

        # Get the standardized member slug for dimension naming
        final_member = member.slug

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
                # Get a 0.25 degree target grid
                grid_dataset = get_grid_dataset(0.25)
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


def get_complete_input_dataset(
    catalogue: intake.Catalog,
    get_entry_dataset_fun: Callable,
    location: InputLocation,
    member: Member,
    rawname: str,
    varname: str,
) -> tuple[xarray.Dataset, Member, str]:
    if "future" in member.simulation:
        dataset_future, member, rawname = get_member_dataset(
            catalogue, get_entry_dataset_fun, location, member, rawname, varname
        )
        member_hist = copy.replace(member, simulation="hist-1950")
        member_hist = rename_realm(member_hist, varname)
        dataset_hist, _, rawname = get_member_dataset(
            catalogue,
            get_entry_dataset_fun,
            location,
            member_hist,
            rawname,
            varname,
        )
        dataset = xarray.concat([dataset_hist, dataset_future], dim="time")
    else:
        dataset, member, rawname = get_member_dataset(
            catalogue, get_entry_dataset_fun, location, member, rawname, varname
        )
    return dataset, member, rawname


def get_member_dataset(
    catalogue: intake.Catalog,
    get_entry_dataset_fun: Callable,
    location: InputLocation,
    member: Member,
    rawname: str,
    varname: str,
) -> tuple[xarray.Dataset, Member, str]:
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


def get_model_time_series(
    varname: str,
    location: InputLocation,
    output_dir: Path,
    members: list[str],
    experiment: str,
    reference_period: tuple[int, int],
    region_set: str,
    get_entry_dataset_fun=get_entry_dataset,
    clobber: bool = False,
    member_class: type[Member] = EERIEMember,
) -> Path:
    """Compute and save regional mean time series for a given model variable.

    Parameters
    ----------
    varname : str
        The CMOR variable name (e.g., 'tas', 'pr').
    location : InputLocation
        The input data location (e.g., 'levante', 'cloud').
    output_dir : Path
        The directory to save the output NetCDF file.
    members : list[str]
        A list of model members to process (e.g., 'eerie-hist-1', 'eerie-amip-1').
    experiment : str
        The experiment name (e.g., 'hist', 'hist-amip').
    reference_period : tuple[int, int]
        The start and end years defining the reference period for anomaly calculation.
    region_set : str
        The set of regions to use for spatial aggregation (e.g., 'EDDY', 'IPCC').
    get_entry_dataset_fun : callable, optional
        A function to retrieve the initial dataset. Defaults to `get_entry_dataset`.
    clobber : bool, optional
        If True, overwrite existing output files. Defaults to False.
    member_class : Member, optional, default is EERIEMember
        Member class to use.

    Returns
    -------
    Path
        The path to the generated output NetCDF file.
    """
    # Get the list of time filters (e.g., annual, seasonal).
    time_filters = get_time_filters()
    # Construct the full path for the output time series file.
    output_path = Path(output_dir, f"{varname}_{experiment}_EERIE_{region_set}_ts.nc")

    # Check if the output file already exists and if overwriting is not allowed.
    if output_path.exists() and not clobber:
        logger.info(f"{output_path} already exists, refusing to overwrite")
        return output_path

    # Get the main data catalogue.
    catalogue = get_main_catalogue()

    time_series_datasets = []
    # Process each model member.
    for member_str in members:
        # Adjust the member string based on variable realm (e.g., 'atmos' to 'ocean').
        member_obj = member_class.from_string(member_str)
        # Rename the member string based on variable realm
        if varname in OCEAN_VARIABLES:
            member_obj = member_obj.to_ocean()
            member_str = member_obj.to_string()
        # Get the raw variable name as it appears in the source files.
        if isinstance(member_obj, CmorEerieMember):
            rawname = varname
            if varname in ["tasmax", "tasmin"] and "icon" in member_obj.model:
                member_obj = copy.replace(member_obj, cmor_table="day")
        else:
            rawname = get_raw_variable_name(member_str, varname)
        dataset, member, rawname = get_complete_input_dataset(
            catalogue,
            get_entry_dataset_fun,
            location,
            member_obj,
            rawname,
            varname,
        )
        # Rename the raw variable to its CMOR-compliant name.
        dataset_cmor = to_cmor_names(dataset, rawname, varname)
        # Convert the member string to a standardized slug for consistent dimension naming.
        final_member = member_obj.to_atmos().slug
        # Iterate through each time filter to generate different time series.
        for time_filter in time_filters:
            logger.info(
                f"Computing time series for {member} and {time_filter.to_str()}"
            )
            # Filter the dataset's time axis based on the current time filter.
            dataset_cmor_filtered = filter_time_axis(dataset_cmor, time_filter)
            # Generate the time series, including regional aggregation.
            dataset_ts = get_time_series(
                dataset_cmor_filtered, time_filter, varname, region_set
            )
            # Fix units if necessary (e.g., temperature from K to degC).
            # 'product="series"' indicates that it's a time series for unit conversion logic.
            dataset_ts = fix_units(dataset_ts, varname, product="series")
            # Define and add extra dimensions (member, time_filter, period as 'reference').
            # The period is set to 'reference' as this is a continuous time series.
            dataset_ts = define_extra_dimensions(
                dataset_ts, final_member, time_filter, "reference"
            )

            # --- Anomaly Calculation ---
            # Define the start and end dates for the reference period.
            start_date_str = f"{reference_period[0]}-01-01"
            end_date_str = f"{reference_period[1]}-12-31"
            # Calculate the climatological mean over the reference period.
            ref_clim = (
                dataset_ts[varname]
                .sel(time=slice(start_date_str, end_date_str))
                .mean(dim="time")
            ).compute()
            # Compute the anomaly by subtracting the reference climatology.
            dataset_ts[varname + "_anom"] = dataset_ts[varname] - ref_clim

            # Stack realizations with the member dimension for ensemble runs (e.g., AMIP).
            if "realization" in dataset_ts:
                logger.info("Stacking realizations into a single member dimension.")
                # Combine 'member' and 'realization' into a new 'memberrea' multi-index dimension.
                dataset_ts = dataset_ts.stack(memberrea=["member", "realization"])
                # Create a string representation for the combined member-realization index.
                dataset_ts["memberrea_str"] = xarray.DataArray(
                    [
                        "_".join([str(i) for i in mi])
                        for mi in dataset_ts.memberrea.to_index()
                    ],
                    dims=["memberrea"],
                )
                # Drop the original 'member', 'realization', and the temporary 'memberrea' dimensions.
                dataset_ts = dataset_ts.drop_vars(
                    ["member", "realization", "memberrea"]
                )
                # Rename the 'memberrea' dimension to 'member' for consistency.
                # Transpose dimensions to a desired order.
                dataset_ts = (
                    dataset_ts.rename_dims(memberrea="member")
                    .rename(memberrea_str="member")
                    .transpose("period", "member", "time_filter", "time", "region")
                )
                # Set the new 'member' as a coordinate for easier indexing.
                dataset_ts = dataset_ts.set_coords(["member"]).set_xindex("member")
            time_series_datasets.append(dataset_ts)

    # Merge all individual time series datasets into a single xarray Dataset.
    final_dataset = xarray.merge(time_series_datasets)
    # Fix global and variable attributes for compliance.
    final_dataset = fix_attributes(final_dataset, varname).squeeze()
    # Optimize the Dask graph for the final dataset before writing.
    logger.info(f"Writing time series to {output_path}")
    final_dataset = dask.optimize(final_dataset)[0]

    # Define encoding options for the NetCDF variables to optimize storage.
    # Chunking is specified for better I/O performance.
    encoding_variable = dict(
        dtype="float32",
        zlib=True,
        complevel=1,
        chunksizes=(final_dataset.member.size, 1, final_dataset.time.size, 3),
    )
    # Apply encoding to both the main variable and its anomaly.
    encoding = {varname: encoding_variable, varname + "_anom": encoding_variable}

    # Safely write the final dataset to a NetCDF file with progress bar.
    safe_to_netcdf(final_dataset, output_path, encoding=encoding, show_progress=True)
    return output_path
