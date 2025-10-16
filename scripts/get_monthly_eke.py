"""Compute monthly EDDY kinetic energy from daily EERIE data.

This needs significant resources (especially memory), try with 64G or 128G
"""

import os
from pathlib import Path

from dask.distributed import Client, LocalCluster
from dotenv import load_dotenv

from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import (
    members_eerie_control,
    members_eerie_future,
    members_eerie_hist,
)
from eerieview.data_access import get_entry_dataset, get_main_catalogue
from eerieview.data_models import EERIEMember, InputLocation
from eerieview.data_processing import rename_realm
from eerieview.eke import DEFAULT_ENCODING, compute_monthly_eke
from eerieview.io_utils import safe_to_netcdf
from eerieview.logger import get_logger

load_dotenv()
logger = get_logger(__name__)
import logging
logging.getLogger("distributed").setLevel(logging.ERROR)


def compute_eke_for_member(member: str, location: InputLocation, clobber: bool = False):
    varname = "zos"
    output_dir = os.environ["DIAGSDIR"]
    # Input data must be daily and ocean
    member = member.replace("monthly", "daily")
    member = rename_realm(member, varname)
    # Get intermediate and final file names
    final_member = EERIEMember.from_string(member.replace("ocean", "atmos")).slug
    output_path = Path(output_dir, f"eke_{final_member}_monthly.nc")
    zos_daily_climatology_file = Path(
        output_dir, f"zos_clim_{final_member}_dayofyear.nc"
    )
    daily_anom_zos_file = Path(output_dir, f"zos_anom_{final_member}_daily.nc")
    if output_path.exists() and not clobber:
        logger.info(f"{output_path} already exists")
    else:
        # Open the catalogue entry
        catalogue = get_main_catalogue()
        rawname = get_raw_variable_name(member, varname)
        dataset = get_entry_dataset(
            catalogue, member, rawname, location=location
        ).chunk(dict(time=1000, lat=100, lon=100))
        # Rename to CMOR names
        dataset_cmor = to_cmor_names(dataset, rawname, varname)

        # Run computation
        eke_monthly = compute_monthly_eke(
            dataset_cmor, daily_anom_zos_file, zos_daily_climatology_file
        )
        safe_to_netcdf(
            eke_monthly,
            output_path,
            encoding=dict(eke=DEFAULT_ENCODING),
            show_progress=True,
        )


def main():
    cluster = LocalCluster()
    print(cluster)
    client = Client(cluster)
    print("Dashboard URL:", client.dashboard_link)
    location: InputLocation = "levante"
    all_members = members_eerie_future + members_eerie_hist + members_eerie_control
    for member in all_members:
        logger.info(f"Computing monthly EKE for {member}")
        try:
            compute_eke_for_member(member, location)
        except Exception as e:
            logger.warning(f"EKE computation failed for {member} with error {e}")


if __name__ == "__main__":
    main()
