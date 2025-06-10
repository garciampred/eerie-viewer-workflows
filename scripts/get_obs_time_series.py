import os
from pathlib import Path
from dotenv import load_dotenv
import dask
import xarray

from eerieview.cmor import to_cmor_names
from eerieview.constants import CMOR2C3SATLAS
from eerieview.data_access import get_obs_dataset
from eerieview.data_processing import (
    define_extra_dimensions,
    filter_time_axis,
    get_time_filters,
)
from eerieview.io_utils import safe_to_netcdf
from eerieview.logger import get_logger
from eerieview.metadata import fix_attributes
from eerieview.product_computation import get_time_series
load_dotenv()
# Initialize logger for the module.
logger = get_logger(__name__)


def get_obs_time_series(
    varname: str,
    obsdir: Path,
    output_dir: Path,
    source: str,
    region_set: str,
    get_obs_dataset_fun=get_obs_dataset,
    reference_period: tuple[int, int] = (1951, 1970),
    clobber: bool = False,
) -> Path:
    """Compute and save regional mean time series for an observational dataset.

    Parameters
    ----------
    varname : str
        The CMOR variable name (e.g., 'tas', 'pr').
    obsdir : Path
        The directory containing the raw observational input data.
    output_dir : Path
        The directory to save the output NetCDF file.
    source : str
        The source of the observational data (e.g., 'era5', 'aviso').
    region_set : str
        The set of regions to use for spatial aggregation (e.g., 'IPCC', 'EDDY').
    get_obs_dataset_fun : callable, optional
        A function to retrieve the raw observational dataset. Defaults to `get_obs_dataset`.
    reference_period : tuple[int, int], optional
        The start and end years defining the reference period for anomaly calculation.
        Defaults to (1951, 1970).
    clobber : bool, optional
        If True, overwrite existing output files. Defaults to False.

    Returns
    -------
    Path
        Output netcdf file path.
    """
    # Get predefined time filters (e.g., for annual, seasonal means).
    time_filters = get_time_filters()
    # Construct the full path for the output time series file.
    output_path = Path(output_dir, f"{varname}_{source}_{region_set}_ts.nc")

    # If the output file exists and clobbering is not allowed, log and return.
    if output_path.exists() and not clobber:
        logger.info(f"{output_path} exists, skipping processing.")
        return output_path

    # Initialize a list to hold time series datasets for merging.
    time_series_datasets = []

    # Retrieve the raw variable name from the CMOR-to-C3S_ATLAS mapping.
    rawname = CMOR2C3SATLAS[varname]
    # Load the observational dataset using the determined raw name.
    dataset = get_obs_dataset_fun(obsdir, rawname)
    # Rename the raw variable to its CMOR-compliant name.
    dataset_cmor = to_cmor_names(dataset, rawname, varname)

    # Process the dataset for each defined time filter.
    for time_filter in time_filters:
        logger.info(
            f"Computing time series for {varname} {source} and {time_filter.to_str()}."
        )
        # Filter the dataset's time axis based on the current time filter (e.g., select seasons).
        dataset_cmor_filtered = filter_time_axis(dataset_cmor, time_filter)
        # Generate the time series, including spatial aggregation by region.
        dataset_ts = get_time_series(
            dataset_cmor_filtered, time_filter, varname, region_set=region_set
        )
        # Define and add extra dimensions (member, period, time_filter) to the dataset.
        # 'reference' is used for the period as these are continuous time series.
        # The 'source' is used as the 'member' for observational data.
        dataset_ts = define_extra_dimensions(
            dataset_ts, source, time_filter, period="reference"
        )

        # --- Anomaly Calculation ---
        # Define the start and end dates for the reference period slice.
        start_date_str = f"{reference_period[0]}-01-01"
        end_date_str = f"{reference_period[1]}-12-31"

        # Calculate the climatological mean over the specified reference period.
        # `.compute()` is called here to trigger immediate Dask computation for the climatology.
        ref_clim = (
            dataset_ts[varname]
            .sel(time=slice(start_date_str, end_date_str))
            .mean(dim="time")
        ).compute()
        # Compute the anomaly by subtracting the reference climatology from the time series.
        dataset_ts[varname + "_anom"] = dataset_ts[varname] - ref_clim
        # Append the processed time series dataset to the list.
        time_series_datasets.append(dataset_ts)

    # Merge all individual time series datasets into a single xarray Dataset.
    final_dataset = xarray.merge(time_series_datasets)
    # Add the reference period to the global attributes for metadata.
    final_dataset.attrs["reference_period"] = str(reference_period)
    # Fix global and variable attributes for compliance.
    final_dataset = fix_attributes(final_dataset, varname).squeeze()
    # Optimize the Dask computation graph for the final dataset.
    logger.info(f"Writing time series to {output_path}.")
    final_dataset = dask.optimize(final_dataset)[0]

    # Define encoding options for the NetCDF variables to optimize storage.
    # `chunksizes` are set for efficient I/O, assuming certain dimension orders.
    encoding_variable = dict(
        dtype="float32",
        zlib=True,
        complevel=1,
        chunksizes=(1, final_dataset.time.size, 3),
    )
    # Apply encoding to both the main variable and its anomaly variable.
    encoding = {varname: encoding_variable, varname + "_anom": encoding_variable}

    # Print statement for debugging, can be removed in production.
    print(f"Writing output to {output_path}")
    # Safely write the final dataset to a NetCDF file with progress bar.
    safe_to_netcdf(final_dataset, output_path, encoding=encoding, show_progress=True)
    return output_path


def main():
    """Run the time series generation process for observational datasets."""
    # Define the base directory for observational input data.
    # Using direct dictionary access for environment variables, assuming they are set.
    obsdir = Path(os.environ["OBSDIR"])
    # Define the output directory for processed time series.
    output_dir = Path(os.environ["PRODUCTSDIR"], "time_series")
    region_set = "IPCC"  # Define the default region set for spatial aggregation.

    # Define reference periods for different datasets.
    reference_period_era5 = (1951, 1980)
    reference_period_aviso = (1991, 2020)

    # --- ERA5 Data Processing ---
    # List of variables to process for ERA5.
    variables_era5 = [
        "sfcWind",
        "tas",
        "pr",
        "tos",
        "clt",
        "tasmax",
        "tasmin",
        "uas",
        "vas",
    ]
    # Process each variable using ERA5 data.
    for varname in variables_era5:
        logger.info(f"Processing {varname} data from ERA5.")
        get_obs_time_series(
            varname,
            obsdir,
            output_dir,
            "era5",  # Source is ERA5
            region_set,
            reference_period=reference_period_era5,
            clobber=True,
        )

    # --- AVISO Data Processing ---
    # List of variables to process for AVISO.
    variables_aviso = ["zos", "eke"]
    # Process each variable using AVISO data.
    for varname in variables_aviso:
        logger.info(f"Processing {varname} data from AVISO.")
        get_obs_time_series(
            varname,
            obsdir,
            output_dir,
            "aviso",  # Source is AVISO
            region_set,
            reference_period=reference_period_aviso,
            clobber=True,  # Do not overwrite existing files for AVISO data.
        )


if __name__ == "__main__":
    main()
