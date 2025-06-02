from datetime import datetime, timezone

import xarray

lon_attr_dict = {
    "long_name": "Longitude",
    "standard_name": "longitude",
    "units": "degrees_east",
}

lat_attr_dict = {
    "long_name": "Latitudes",
    "standard_name": "latitude",
    "units": "degrees_north",
}


def fix_attributes(final_dataset: xarray.Dataset, varname: str) -> xarray.Dataset:
    for attr in final_dataset[varname].attrs.copy():
        if "GRIB" in attr:
            del final_dataset[varname].attrs[attr]
    for attr in final_dataset.attrs.copy():
        if "GRIB" in attr or "intake_esm" in attr:
            del final_dataset.attrs[attr]
    if "institution" in final_dataset.attrs:
        del final_dataset.attrs["institution"]
    final_dataset.attrs["history"] = (
        f"Created by Predictia Intelligent Data Solutions for the EERIE Data viewer "
        f"with {__file__} at {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}."
    )
    if "lon" in final_dataset:
        final_dataset["lon"].attrs = lon_attr_dict
        final_dataset["lat"].attrs = lat_attr_dict
    final_dataset = final_dataset.drop_vars(["height", "height2"], errors="ignore")
    final_dataset = final_dataset.drop_vars(["surface"], errors="ignore")
    return final_dataset
