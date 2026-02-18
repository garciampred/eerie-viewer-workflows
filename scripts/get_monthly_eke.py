"""Compute monthly EDDY kinetic energy from daily EERIE data.

This needs significant resources (especially memory), try with 64G or 128G
"""

import os
from dataclasses import replace
from pathlib import Path

import dask
from dask.distributed import Client, LocalCluster
from dotenv import load_dotenv

from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import (
    members_eerie_control_cmor,
    members_eerie_future_cmor,
    members_eerie_hist_cmor,
)
from eerieview.data_access import get_entry_dataset, get_main_catalogue
from eerieview.data_models import CmorEerieMember, InputLocation, Member
from eerieview.data_processing import retry_get_entry_with_fixes
from eerieview.eke import DEFAULT_ENCODING, compute_monthly_eke
from eerieview.io_utils import safe_to_netcdf
from eerieview.io_utils import safe_to_zarr
from eerieview.logger import get_logger

load_dotenv()
logger = get_logger(__name__)
import logging

logging.getLogger("distributed").setLevel(logging.INFO)


def compute_eke_for_member(
    member: Member, location: InputLocation, clobber: bool = False
):
    varname = "zos"
    output_dir = os.environ["DIAGSDIR"]
    # Input data must be daily and ocean
    member = member.to_ocean().to_daily()
    member_str = member.to_string()
    if (
        "ifs-nemo" in member.model
        or "ifs-fesom2-sr" in member.model
        or "icon" in member.model
    ):
        member = replace(member, cmor_table="HROday")
    # Get intermediate and final file names
    final_member = member.to_atmos().slug
    output_path = Path(output_dir, f"eke_{final_member}_monthly.nc")
    zos_daily_climatology_file = Path(
        output_dir, f"zos_clim_{final_member}_dayofyear.zarr"
    )
    daily_anom_zos_file = Path(output_dir, f"zos_anom_{final_member}_daily.zarr")
    if output_path.exists() and not clobber:
        logger.info(f"{output_path} already exists")
    else:
        # Open the catalogue entry
        catalogue = get_main_catalogue()
        if isinstance(member, CmorEerieMember):
            rawname = varname
        else:
            rawname = get_raw_variable_name(member_str, varname)
        try:
            # Attempt to retrieve the dataset for the current member and variable
            dataset = get_entry_dataset(catalogue, member, rawname, location=location)
        except KeyError:
            # If a KeyError occurs, retry with common fixes
            dataset, member, rawname = retry_get_entry_with_fixes(
                catalogue, get_entry_dataset, location, member, rawname, varname
            )
        # dataset = dataset.chunk(dict(time=100, lat=100, lon=100))
        # Rename to CMOR names
        dataset_cmor = to_cmor_names(dataset, rawname, varname)
        # Run computation
        print(dataset)
        eke_monthly = compute_monthly_eke(
            dataset_cmor, daily_anom_zos_file, zos_daily_climatology_file
        )
        safe_to_zarr(
            eke_monthly,
            Path(str(output_path).replace(".nc", ".zarr")),
            encoding=dict(eke=DEFAULT_ENCODING),
            show_progress=True,
        )


def main():
    # dask.config.set({"scheduler": "threads"})
    dask.config.set(
        {
            "temporary-directory": "/scratch/b/b382819/tmp/dask-scratch-space/",
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "120s",
            "distributed.worker.connections.outgoing": 2,
            "distributed.worker.connections.incoming": 2,
            "distributed.comm.compression": "auto",
            "distributed.scheduler.worker-saturation": 0.9,
            "distributed.worker.memory.target": 0.70,  # start spilling to disk
            "distributed.worker.memory.spill": 0.80,  # spill aggressively
            "distributed.worker.memory.pause": 0.95,  # only pause as a last resort
        }
    )
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="25GB")
      # n_workers=1, memory_limit="40GB", threads_per_worker=4)
    print(cluster)
    client = Client(cluster, timeout="120s")
    print("Dashboard URL:", client.dashboard_link)
    location: InputLocation = "levante_cmor"
    all_members = (
        members_eerie_future_cmor + members_eerie_hist_cmor + members_eerie_control_cmor
    )
    for member_str in all_members:
        if "HadGEM3" in member_str or "nemo" in member_str:
            continue
        logger.info(f"Computing monthly EKE for {member_str}")
        try:
            member = CmorEerieMember.from_string(member_str)
            compute_eke_for_member(member, location, clobber=False)
        except Exception as e:
            raise
            #            logger.warning(f"EKE computation failed for {member_str} with error {e}")


if __name__ == "__main__":
    main()
