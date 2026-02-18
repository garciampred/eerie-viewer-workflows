import json
from pathlib import Path

import pandas as pd

from eerieview.cmor import get_raw_variable_name, to_cmor_names
from eerieview.constants import OCEAN_VARIABLES, members_eerie_future_cmor
from eerieview.data_access import get_diagnostic, get_entry_dataset, get_main_catalogue
from eerieview.data_models import (
    CmorEerieMember,
    EERIEMember,
    InputLocation,
    Member,
)
from eerieview.data_processing import fix_units
from eerieview.logger import get_logger
from eerieview.product_computation import get_complete_input_dataset
from scripts.get_climatologies import VARIABLES

logger = get_logger(__name__)


def check_variable_data(
    varname: str,
    location: InputLocation,
    members: list[str],
    get_entry_dataset_fun=get_entry_dataset,
    member_class: type[Member] = EERIEMember,
):
    catalogue = get_main_catalogue()

    nan_records = {}

    # Iterate over each model member
    for member_str in members:
        member_obj = member_class.from_string(member_str)
        # Rename the member string based on variable realm
        if varname in OCEAN_VARIABLES:
            member_obj = member_obj.to_ocean()
            member_str = member_obj.to_string()
        # Get the raw variable name from the CMOR mapping
        if isinstance(member_obj, CmorEerieMember):
            rawname = varname
            if varname in ["tasmax", "tasmin"] and member_obj.model in [
                "icon-esm-er",
                "ifs-fesom2-sr",
            ]:
                member_obj = member_obj.to_daily()
        else:
            rawname = get_raw_variable_name(member_str, varname)
        dataset, member, rawname = get_complete_input_dataset(
            catalogue,
            get_entry_dataset_fun,
            location,
            member_obj,
            rawname,
            varname,
        )
        times = dataset.time.to_index()
        if "mon" in member.cmor_table:
            freq = "MS"
            dataset["time"] = [t.replace(day=1, hour=0, minute=0) for t in times]
        else:
            dataset["time"] = [t.replace(hour=0, minute=0) for t in times]
            freq = "D"
        mask = ~dataset.time.to_index().duplicated()
        dataset = dataset.sel(time=mask)
        new_times = pd.date_range(
            start=dataset.time.to_index()[0], end=dataset.time.to_index()[-1], freq=freq
        )
        dataset = dataset[rawname].reindex(time=new_times).to_dataset(name=rawname)
        print(dataset)
        # Squeeze out singleton dimensions
        dataset = dataset.squeeze()
        dataset_cmor = to_cmor_names(dataset, rawname, varname)
        # Fix units if necessary (e.g., K to degC, m/s to mm/day)
        dataset_cmor = fix_units(dataset_cmor, varname)
        nan_mask = dataset_cmor[varname].isnull().all(dim=("lat", "lon"))
        nan_times = dataset_cmor.time.to_index()[nan_mask]
        print(nan_times)
        nan_records[member.slug] = [t.strftime("%Y-%m-%d") for t in nan_times]

    return nan_records


def main():
    location: InputLocation = "levante_cmor"
    members = members_eerie_future_cmor
    experiment = "future"
    for varname in VARIABLES:
        logger.info(f"Processing {varname} data for 'future' experiments")

        get_entry_dataset_fun = (
            get_diagnostic if varname == "eke" else get_entry_dataset
        )

        nan_records_variable = check_variable_data(
            varname=varname,
            location=location,
            members=members,
            get_entry_dataset_fun=get_entry_dataset_fun,
            member_class=CmorEerieMember,
        )

        with Path(f"nans_{varname}_{experiment}.json").open(mode="w") as nansfile:
            json.dump(nan_records_variable, nansfile)


if __name__ == "__main__":
    main()
