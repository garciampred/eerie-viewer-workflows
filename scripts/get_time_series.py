import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from eerieview.constants import (
    members_eerie_control_cmor,
    members_eerie_future_cmor,
    members_eerie_hist_amip,
    members_eerie_hist_cmor,
)
from eerieview.data_access import get_diagnostic, get_entry_dataset
from eerieview.data_models import CmorEerieMember, InputLocation
from eerieview.logger import get_logger
from eerieview.product_computation import get_model_time_series

load_dotenv()
logger = get_logger(__name__)


def get_exp_time_series(experiment: str, region_set: str, members: list | None = None):
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
    members = members if members is not None else exp2members[experiment]
    reference_period = exp2ref_period[experiment]
    output_dir = Path(os.environ["PRODUCTSDIR"], "time_series")

    # Define the list of variables to process.
    '''
    variables_to_process = [
        "sfcWind",
        "uas",
        "vas",
        "tas",
        "pr",
        "tos",
        #"sos",
        "clt",
        "zos",
        "tasmax",
        "tasmin",
    ]
    '''

    variables_to_process = ['eke']
    
    # Iterate through each variable to process.
    for varname in variables_to_process:
        # Skip specific variables for the 'hist-amip' experiment if they are not relevant.
        if experiment == "hist-amip" and varname in ["eke", "zos", "sos"]:
            logger.info(f"Skipping {varname} for {experiment} experiment.")
            continue

        # Determine the appropriate function to get the initial dataset.
        # 'eke' often requires a special diagnostic function.
        get_entry_dataset_fun: Callable
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
            clobber=True,
            get_entry_dataset_fun=get_entry_dataset_fun,
            member_class=CmorEerieMember,
        )


def members_with_eke_data(members: list) -> list:
    """Filter member list to only those with a pre-computed eke file (zarr or nc) in DIAGSDIR."""
    diagdir = os.environ["DIAGSDIR"]
    result = []
    for m in members:
        slug = CmorEerieMember.from_string(m).slug
        zarr_exists = Path(diagdir, f"eke_{slug}_monthly.zarr").exists()
        nc_exists = Path(diagdir, f"eke_{slug}_monthly.nc").exists()
        if zarr_exists or nc_exists:
            result.append(m)
    return result


def main():
    region_sets = ["IPCC", "EDDY"]
    for region_set in region_sets:
        get_exp_time_series("hist", region_set, members=members_with_eke_data(members_eerie_hist_cmor))
        get_exp_time_series("future", region_set, members=members_with_eke_data(members_eerie_future_cmor))


if __name__ == "__main__":
    main()
