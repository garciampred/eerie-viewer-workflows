import os
from pathlib import Path

import dask
import xarray
from dotenv import load_dotenv

from eerieview.constants import CMOR2C3SATLAS
from eerieview.data_access import get_obs_dataset
from eerieview.data_models import DecadalProduct, PeriodsConfig
from eerieview.data_processing import (
    define_extra_dimensions,
    filter_time_axis,
    fix_units,
    get_time_filters,
)
from eerieview.grids import get_grid_dataset
from eerieview.io_utils import safe_to_netcdf
from eerieview.logger import get_logger
from eerieview.metadata import fix_attributes
from eerieview.product_computation import get_decadal_product_or_fill_with_nan

logger = get_logger(__name__)
load_dotenv()


def get_obs_decadal_product(
    varname: str,
    obsdir: Path,
    output_dir: Path,
    periods: PeriodsConfig,
    source: str,
    product: DecadalProduct = "clim",
    get_obs_dataset_fun=get_obs_dataset,
    clobber: bool = False,
) -> Path:
    """Compute and save decadal products (e.g., climatologies, trends) for observational datasets.

    Parameters
    ----------
    varname : str
        The CMOR-compliant name of the variable to process (e.g., 'tasmax', 'pr').
    obsdir : Path
        The directory containing the raw observational input data.
    output_dir : Path
        The directory where the final processed NetCDF output will be stored.
    periods : PeriodsConfig
        An object defining the reference period and other analysis periods for decadal products.
    source : str
        The source of the observational data (e.g., 'era5', 'aviso').
    product : DecadalProduct, optional
        The type of decadal product to compute ('clim' for climatology, 'trend' for trend).
        Defaults to 'clim'.
    get_obs_dataset_fun : callable, optional
        A function to retrieve the raw observational dataset. Defaults to `get_obs_dataset`.
        This allows for dependency injection and easier testing.
    clobber : bool, optional
        If True, an existing output file will be overwritten. If False and the file exists,
        the function will skip processing and return the path to the existing file.
        Defaults to False.

    Returns
    -------
    Path
        The path to the generated NetCDF output file.
    """
    # Get predefined time filters (e.g., for seasons, annual).
    time_filters = get_time_filters()
    # Construct the full path for the output NetCDF file.
    output_path = Path(output_dir, f"{varname}_{source}_{product}.nc")

    # Check if the output file already exists and if overwriting is not allowed.
    if output_path.exists() and not clobber:
        logger.info(f"{output_path} exists, skipping processing.")
        return output_path

    climatologies = []
    rawname = CMOR2C3SATLAS[varname]
    dataset = get_obs_dataset_fun(obsdir, rawname).rename({rawname: varname})
    dataset = fix_units(dataset, varname, product)

    # Iterate through each defined time filter (e.g., annual, seasonal).
    for time_filter in time_filters:
        logger.info(f"Computing {product} for time filter: {time_filter.to_str()}")
        # Filter the dataset's time axis based on the current time filter.
        dataset_filtered = filter_time_axis(dataset, time_filter)

        # Iterate through each period defined in PeriodsConfig.
        for period in periods.all_list:
            # Get the specific product (climatology or trend)
            dataset_product = get_decadal_product_or_fill_with_nan(
                dataset_filtered, period, product, varname
            ).compute()

            # Define and add extra dimensions (member, period, time_filter) for metadata.
            # 'ERA5' is hardcoded as the member name here for observational data.
            dataset_product = define_extra_dimensions(
                dataset_product, "ERA5", time_filter, period
            )

            # Get a regular grid dataset for regridding.
            grid_dataset = get_grid_dataset(0.25)
            # Regrid the product dataset to the common high-resolution grid using conservative regridding.
            dataset_product = dataset_product.regrid.conservative(
                grid_dataset, latitude_coord="lon"
            )

            # Fix units of the variable if necessary (e.g., temperature from K to degC, precipitation units).
            
            climatologies.append(dataset_product)

    # Merge all individual product datasets into a single xarray Dataset.
    final_dataset = xarray.merge(climatologies).squeeze()
    final_dataset = dask.optimize(final_dataset)[0]

    # --- Anomaly Calculation ---
    # Retrieve the reference period from the PeriodsConfig.
    reference_period = periods.reference_period
    # Format the reference period for selection.
    ref_period_str = f"{reference_period[0]}-{reference_period[1]}"

    # If only one period was processed, expand the 'period' dimension to allow for anomaly calculation
    # (even if anomalies will be zero in this case).
    if final_dataset.period.size == 1:
        final_dataset = final_dataset.expand_dims("period")

    # Define encoding options for NetCDF variables to optimize storage.
    encoding_opts = dict(zlib=True, complevel=1, shuffle=True)
    # Calculate anomalies if the product is 'clim'.
    if product == "clim":
        final_dataset[varname + "_anom"] = final_dataset[varname] - final_dataset[
            varname
        ].sel(period=ref_period_str)
        # Set encoding for both the original variable and the anomaly variable.
        encoding = {varname: encoding_opts, varname + "_anom": encoding_opts}
    else:
        # If the product is 'trend', encoding for the trend variable and its p-value.
        encoding = {varname: encoding_opts, varname + "_pvalue": encoding_opts}

    # Fix global and variable attributes for compliance.
    final_dataset = fix_attributes(final_dataset, varname)

    logger.info(f"Writing final dataset to {output_path}")
    safe_to_netcdf(final_dataset, output_path, show_progress=True, encoding=encoding)
    return output_path


def main():
    """Run the processing of observational decadal products."""
    # Define the base directory for observational data.
    obsdir = Path(os.environ["OBSDIR"])
    # Define the output directory for processed products. Use environment variable for flexibility.
    output_dir = Path(os.environ["PRODUCTSDIR"], "decadal")

    # --- ERA5 Data Processing ---
    # Define periods for ERA5 data processing.
    reference_period_era5 = (1951, 1980)
    periods_era5 = [(1971, 2000), (1991, 2020)]
    periods_config_era5 = PeriodsConfig(reference_period_era5, periods_era5)
    product: DecadalProduct = "trend"

    # List of variables to process for ERA5.
    variables_era5 = [
        "sfcWind",
        "tas",
        "tasmax",
        "tasmin",
        "pr",
        "tos",
        "clt",
        "uas",
        "vas",
    ]
    # Process each variable for ERA5.
    for varname in variables_era5:
        logger.info(f"Processing ERA5 data for variable: {varname}")
        get_obs_decadal_product(
            varname,
            obsdir,
            output_dir,
            periods_config_era5,
            "era5",
            product,
            clobber=False,
        )

    # --- AVISO Data Processing ---
    # Define periods for AVISO data processing (typically shorter timeframes).
    variables_aviso = ["zos", "eke"]
    reference_period_aviso = (1991, 2020)
    periods_aviso = [
        (1991, 2020)
    ]  # This implies processing the reference period itself as a product
    periods_config_aviso = PeriodsConfig(reference_period_aviso, periods_aviso)

    # Process each variable for AVISO.
    for varname in variables_aviso:
        logger.info(f"Processing AVISO data for variable: {varname}")
        get_obs_decadal_product(
            varname,
            obsdir,
            output_dir,
            periods_config_aviso,
            "aviso",
            product,
            clobber=False,
        )


if __name__ == "__main__":
    main()
