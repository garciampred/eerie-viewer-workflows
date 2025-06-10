import os
from pathlib import Path

from dotenv import load_dotenv

from eerieview.constants import (
    members_eerie_control,
    members_eerie_hist,
    members_eerie_hist_amip,
)
from eerieview.data_access import get_diagnostic, get_entry_dataset
from eerieview.data_models import (
    DecadalProduct,
    InputLocation,
    PeriodsConfig,
)
from eerieview.logger import get_logger
from eerieview.product_computation import get_model_decadal_product

# Initialize logger for the module
logger = get_logger(__name__)
# Load environment variables from a .env file
load_dotenv()

# Define a list of variables to be processed
VARIABLES = [
    "sfcWind",
    "uas",
    "vas",
    "tas",
    "pr",
    "tos",
    "zos",
    "clt",
    "tasmax",
    "tasmin",
    "eke",
]

"""
Main Processing Functions

These functions launch the data processing workflow
for different experimental setups (control, historical, AMIP).
"""


def main_control():
    location: InputLocation = "levante"
    product: DecadalProduct = "trend"
    reference_period = (1951, 1980)
    periods = [(1951, 1980), (1971, 2000), (1991, 2020), (2021, 2050)]
    periods_config = PeriodsConfig(reference_period, periods)
    output_dir = Path(os.environ["PRODUCTSDIR"], "decadal")

    for varname in VARIABLES:
        logger.info(f"Processing {varname} data for 'control' experiment")
        # Determine which function to use to get the initial dataset
        get_entry_dataset_fun = (
            get_diagnostic if varname == "eke" else get_entry_dataset
        )

        get_model_decadal_product(
            varname=varname,
            output_dir=output_dir,
            members=members_eerie_control,
            periods=periods_config,
            product=product,
            experiment="control",
            clobber=True,
            location=location,
            get_entry_dataset_fun=get_entry_dataset_fun,
        )


def main_hist():
    location: InputLocation = "levante"
    product: DecadalProduct = "trend"
    reference_period = (1951, 1980)
    periods = [(1971, 2000), (1991, 2020)]
    members = members_eerie_hist
    periods_config = PeriodsConfig(reference_period, periods)
    output_dir = Path(os.environ["PRODUCTSDIR"], "decadal")

    for varname in VARIABLES:
        logger.info(f"Processing {varname} data for 'hist' experiment")

        get_entry_dataset_fun = (
            get_diagnostic if varname == "eke" else get_entry_dataset
        )

        get_model_decadal_product(
            varname=varname,
            location=location,
            output_dir=output_dir,
            members=members,
            periods=periods_config,
            product=product,
            experiment="hist",
            clobber=False,
            get_entry_dataset_fun=get_entry_dataset_fun,
        )


def main_amip():
    location: InputLocation = "levante"
    product: DecadalProduct = "clim"
    reference_period = (1981, 2010)
    periods = [(1991, 2020)]
    periods_config = PeriodsConfig(reference_period, periods)
    output_dir = Path(os.environ["PRODUCTSDIR"], "decadal")
    variables_amip = [v for v in VARIABLES if v not in ["eke", "zos"]]
    for varname in variables_amip:
        logger.info(f"Processing {varname} data for 'hist-amip' experiment")
        get_model_decadal_product(
            varname=varname,
            location=location,
            output_dir=output_dir,
            members=members_eerie_hist_amip,
            periods=periods_config,
            product=product,
            experiment="hist-amip",
            clobber=True,
        )


if __name__ == "__main__":
    main_amip()
