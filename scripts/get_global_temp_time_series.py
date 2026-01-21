import os
from pathlib import Path

from dotenv import load_dotenv

from eerieview.constants import (
    members_eerie_control_cmor,
    members_eerie_future_cmor,
    members_eerie_hist_amip,
    members_eerie_hist_cmor,
)
from eerieview.data_access import get_entry_dataset
from eerieview.data_models import CmorEerieMember, InputLocation
from eerieview.logger import get_logger
from eerieview.product_computation import get_model_time_series

load_dotenv()
logger = get_logger(__name__)


def get_exp_time_series(experiment: str, region_set: str):
    """Run the time series generation process for specific model experiments and variables."""
    location: InputLocation = "levante_cmor"
    exp2ref_period = {
        "hist-amip": (1981, 2000),
        "hist": (1951, 1980),
        "future": (1951, 1980),
        "control": (1951, 1980),
    }
    exp2members = {
        "hist-amip": members_eerie_hist_amip,
        "hist": members_eerie_hist_cmor,
        "control": members_eerie_control_cmor,
        "future": members_eerie_future_cmor,
    }
    members = exp2members[experiment]
    reference_period = exp2ref_period[experiment]
    output_dir = Path(os.environ["PRODUCTSDIR"], "time_series")

    # Define the list of variables to process.
    variables_to_process = [
        "tas",
    ]

    # Iterate through each variable to process.
    for varname in variables_to_process:
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
            member_class=CmorEerieMember,
        )


def main():
    region_sets = ["Global"]
    experiments = ["control", "future"]
    for region_set in region_sets:
        for exp in experiments:
            get_exp_time_series(exp, region_set)


if __name__ == "__main__":
    main()
