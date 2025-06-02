"""Compute monthly EDDY kinetic energy from daily EERIE data.

This needs significant resources (especially memory), try with 64G or 128G
"""

import os
from pathlib import Path

from dask.distributed import Client, LocalCluster

from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import members_eerie_control
from eerieview.data_access import get_entry_dataset, get_main_catalogue
from eerieview.data_models import EERIEMember
from eerieview.data_processing import rename_realm
from eerieview.ekf import DEFAULT_ENCODING, compute_monthly_eke
from eerieview.io_utils import safe_to_netcdf


def main():
    cluster = LocalCluster()
    print(cluster)
    client = Client(cluster)
    print("Dashboard URL:", client.dashboard_link)
    location = "levante"
    varname = "zos"
    output_dir = os.environ["DIAGSDIR"]
    member = members_eerie_control[1].replace("monthly", "daily")
    member = rename_realm(member, varname)
    catalogue = get_main_catalogue()
    rawname = get_raw_variable_name(member, varname)
    dataset = get_entry_dataset(catalogue, member, rawname, location=location).chunk(
        dict(time=1000, lat=100, lon=100)
    )
    dataset_cmor = to_cmor_names(dataset, rawname, varname)
    # Get intermediate and final file names
    final_member = EERIEMember.from_string(member.replace("ocean", "atmos")).slug
    output_path = Path(output_dir, f"eke_{final_member}_monthly.nc")
    zos_daily_climatology_file = Path(
        output_dir, f"zos_clim_{final_member}_dayofyear.nc"
    )
    daily_anom_zos_file = Path(output_dir, f"zos_anom_{final_member}_daily.nc")
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


if __name__ == "__main__":
    main()
