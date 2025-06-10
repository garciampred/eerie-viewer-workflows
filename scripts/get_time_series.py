import os
from pathlib import Path
from dotenv import load_dotenv
import dask
import xarray

from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import (
    members_eerie_hist_amip, members_eerie_hist, members_eerie_control
)
from eerieview.data_access import get_diagnostic, get_entry_dataset, get_main_catalogue
from eerieview.data_models import EERIEMember, InputLocation
from eerieview.data_processing import (
    define_extra_dimensions,
    filter_time_axis,
    fix_units,
    get_time_filters,
    rename_realm,
    retry_get_entry_with_fixes,
)
from eerieview.io_utils import safe_to_netcdf
from eerieview.logger import get_logger
from eerieview.metadata import fix_attributes
from eerieview.product_computation import get_time_series
load_dotenv()
logger = get_logger(__name__)


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
        member = rename_realm(member_str, varname)
        # Get the raw variable name as it appears in the source files.
        rawname = get_raw_variable_name(member, varname)

        try:
            # Attempt to retrieve the dataset for the current member and variable.
            dataset = get_entry_dataset_fun(
                catalogue, member, rawname, location=location
            )
        except KeyError:
            # If initial retrieval fails, retry with common fixes.
            dataset, member, rawname = retry_get_entry_with_fixes(
                catalogue, get_entry_dataset_fun, location, member, rawname, varname
            )

        # Rename the raw variable to its CMOR-compliant name.
        dataset_cmor = to_cmor_names(dataset, rawname, varname)
        # Convert the member string to a standardized slug for consistent dimension naming.
        final_member = EERIEMember.from_string(member.replace("ocean", "atmos")).slug

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


def get_exp_time_series(experiment: str, region_set: str):
    """Run the time series generation process for specific model experiments and variables."""
    location: InputLocation = "levante"
    exp2ref_period = {"hist-amip": (1981, 2000), "hist": (1951, 1980), "control": (1951, 1980)}
    exp2members = {"hist-amip": members_eerie_hist_amip, "hist": members_eerie_hist, "control": members_eerie_control}
    members = exp2members[experiment]
    reference_period = exp2ref_period[experiment]
    output_dir = Path(os.environ["PRODUCTSDIR"], "time_series")

    # Define the list of variables to process.
    variables_to_process = [
        "sfcWind",
        "uas",
        "vas",
        "tas",
        "pr",
        "tos",
        "clt",
        "zos",
        "tasmax",
        "tasmin",
        "eke",
    ]

    # Iterate through each variable to process.
    for varname in variables_to_process:
        # Skip specific variables for the 'hist-amip' experiment if they are not relevant.
        if experiment == "hist-amip" and varname in ["eke", "zos"]:
            logger.info(f"Skipping {varname} for {experiment} experiment.")
            continue

        # Determine the appropriate function to get the initial dataset.
        # 'eke' often requires a special diagnostic function.
        if varname in ["eke"]:
            get_entry_dataset_fun = get_diagnostic
        else:
            get_entry_dataset_fun = get_entry_dataset

        logger.info(f"Processing {varname} data.")
        # Call the main time series generation function.
        get_model_time_series(
            varname,
            location,
            output_dir,
            members,
            experiment,
            reference_period,
            region_set,
            clobber=False,
            get_entry_dataset_fun=get_entry_dataset_fun,
        )


def main():
    region_sets = ["IPCC", "EDDY"]
    experiments = ["control", "hist", "hist-amip"]
    for region_set in region_sets:
        for exp in experiments:
            get_exp_time_series(exp, region_set)


if __name__ == "__main__":
    main()
