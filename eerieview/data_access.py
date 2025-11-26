import os
from pathlib import Path

import intake
import xarray

from eerieview.constants import location2prefix
from eerieview.data_models import EERIEMember, InputLocation, Member
from eerieview.logger import get_logger

logger = get_logger(__name__)


def get_main_catalogue():
    """Open the main EERIE Intake catalogue."""
    catalogue = intake.open_catalog(
        "https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml"
    )
    return catalogue


def to_dask(catalogue_entry, **kwargs):
    """Convert an Intake catalogue entry into a Dask-backed xarray Dataset.

    Drops common unnecessary variables and cleans up time coordinate attributes.
    """
    dataset = catalogue_entry.to_dask(**kwargs)
    dataset = dataset.drop_vars(
        ["cell_sea_land_mask", "height_2", "depth"],
        errors="ignore",
    ).squeeze()
    if "units" in dataset.time.attrs:
        del dataset.time.attrs["units"]
        del dataset.time.attrs["calendar"]
    dataset.lon.encoding = {}
    dataset.lat.encoding = {}
    return dataset


def get_entry_dataset(
    catalogue: intake.Catalog,  # Changed type hint from 'str' to 'intake.Catalog' for accuracy
    member: Member,
    rawname: str,
    location: InputLocation,
    to_dask_funct=to_dask,
) -> xarray.Dataset:
    """Retrieve a specific dataset entry from the catalogue for a given member.

    Handles concatenation with spin-up data for 'control' members if applicable.

    Parameters
    ----------
    catalogue : intake.Catalog
        The Intake catalogue object.
    member : Member
        The EERIE ensemble member as a Member instance.
    rawname : str
        The raw variable name in the dataset.
    location : InputLocation
        The input data location (e.g., 'levante', 'cloud').
    to_dask_funct : callable, optional
        A function to convert catalogue entries to Dask-backed datasets. Defaults to `to_dask`.

    Returns
    -------
    xarray.Dataset
        The loaded and prepared xarray Dataset for the specified entry.
    """
    logger.info(f"Read EERIE member {member} to an xarray Dataset.")
    location_prefix = location2prefix[location]
    member_str = location_prefix + "." + member.to_string()
    catalogue_entry = catalogue[member_str](method="kerchunk")  # type: ignore
    dataset = to_dask_funct(catalogue_entry)

    if "control" in member_str and "fesom" not in member_str:
        member_spin_up = member_str.replace("control", "spinup")
        print(f"Reading spinup from {member_spin_up}")
        dataset_spin_up = to_dask_funct(catalogue[member_spin_up])
        dataset = xarray.concat([dataset_spin_up, dataset], dim="time")
        dataset = dataset.sortby("time").drop_duplicates(dim="time")

    dataset = dataset[[rawname]].astype("float32")
    if "lon" not in dataset.dims and "native" not in member:
        dataset = dataset.set_index(value=("lat", "lon")).unstack("value")
    logger.info(dataset)
    time_index = dataset.time.to_index()
    logger.info(
        f"Time span for {member_str} is from {time_index[0]} to {time_index[-1]}."
    )
    return dataset


def get_obs_dataset(obsdir: Path, rawname: str) -> xarray.Dataset:
    """Retrieve an observational dataset based on the raw variable name.

    Parameters
    ----------
    obsdir : Path
        The directory where observational data files are stored.
    rawname : str
        The raw name of the variable to retrieve (e.g., 'tx', 'tn', 'eke').

    Returns
    -------
    xarray.Dataset
        The loaded observational dataset.
    """
    if rawname in ["eke", "zos"]:
        ipath = Path(obsdir, f"{rawname}_AVISO_mon_199301-202206.nc")
    else:
        ipath = Path(obsdir, f"{rawname}_ERA5_mon_194001-202212.nc")

    logger.info(f"Reading observations from {ipath}.")
    dataset = xarray.open_dataset(ipath, chunks="auto")[[rawname]]

    if "longitude" in dataset.dims:
        dataset = dataset.rename(
            dict(longitude="lon", latitude="lat", valid_time="time")
        )
    elif "valid_time" in dataset.dims:
        dataset = dataset.rename(dict(valid_time="time"))

    return dataset


def get_diagnostic(
    catalogue: str,
    member: str,
    rawname: str,
    location: InputLocation,
    to_dask_funct=to_dask,
) -> xarray.Dataset:
    """Retrieve a diagnostic dataset from a specific directory."""
    diagdir = os.environ["DIAGSDIR"]
    final_member = EERIEMember.from_string(member).slug
    diagfile = Path(diagdir, f"{rawname}_{final_member}_monthly.nc")
    dataset = xarray.open_dataset(diagfile)
    dataset = dataset[[rawname]].astype("float32")
    time_index = dataset.time.to_index()
    logger.info(f"Time span for {member} is {time_index[0]} from {time_index[-1]}")
    return dataset
