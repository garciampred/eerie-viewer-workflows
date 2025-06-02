import xarray

from eerieview.constants import CMOR2EERIE, CMOR2EERIEAMIP


def get_raw_variable_name(member: str, varname: str) -> str:
    """Map a CMOR variable name to its raw name used in the original datasets,
    based on the model member.
    """
    if "icon-esm-er" in member:
        # Specific mappings for ICON-ESM-ER model
        if varname == "tos":
            rawname = "to"
        elif varname == "zos":
            rawname = "ssh"
        elif varname == "sfcWind":
            rawname = "sfcwind"
        elif varname in ["tasmax", "tasmin"]:
            rawname = "tas"
        else:
            rawname = varname
    elif "amip" in member:
        # Use AMIP-specific mapping for AMIP experiments
        variable_mapping = CMOR2EERIEAMIP
        rawname = variable_mapping[varname]
    else:
        # Use general CMOR to EERIE mapping
        variable_mapping = CMOR2EERIE
        rawname = variable_mapping[varname]
    return rawname


def to_cmor_names(
    dataset: xarray.Dataset, rawname: str, cmorname: str
) -> xarray.Dataset:
    return dataset.rename({rawname: cmorname})
