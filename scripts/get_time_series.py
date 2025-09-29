import os
from pathlib import Path

from dotenv import load_dotenv

from eerieview.constants import (
    members_eerie_control,
    members_eerie_hist,
    members_eerie_hist_amip,
)
from eerieview.data_access import get_diagnostic, get_entry_dataset
from eerieview.data_models import InputLocation
from eerieview.logger import get_logger
from eerieview.product_computation import get_model_time_series

load_dotenv()
logger = get_logger(__name__)


def get_exp_time_series(experiment: str, region_set: str):
    """Run the time series generation process for specific model experiments and variables."""
    location: InputLocation = "levante"
    exp2ref_period = {
        "hist-amip": (1981, 2000),
        "hist": (1951, 1980),
        "control": (1951, 1980),
    }
    exp2members = {
        "hist-amip": members_eerie_hist_amip,
        "hist": members_eerie_hist,
        "control": members_eerie_control,
    }
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
