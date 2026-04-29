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
    catalogue_entry = catalogue[member_str](
        driver="kerchunk", chunks=dict(time=100, lon=100, lat=100)
    )  # type: ignore
    dataset = to_dask_funct(catalogue_entry)

    if (
        "control" in member_str
        and "fesom" not in member_str
        and isinstance(member, EERIEMember)
    ):
        member_spin_up = member_str.replace("control", "spinup")
        print(f"Reading spinup from {member_spin_up}")
        dataset_spin_up = to_dask_funct(catalogue[member_spin_up])
        dataset = xarray.concat([dataset_spin_up, dataset], dim="time")
        dataset = dataset.sortby("time").drop_duplicates(dim="time")

    dataset = dataset[[rawname]].astype("float32")
    if "lon" not in dataset.dims and "native" not in member_str:
        dataset = dataset.set_index(value=("lat", "lon")).unstack("value")
    logger.info(dataset)
    if "time_2" in dataset.coords:
        logger.warning("Renaming time_2 to time")
        if "time" in dataset.dims:
            logger.info("Removing time")
            dataset = dataset.drop_dims("time")
        logger.info(dataset)
        dataset = dataset.rename(time_2="time")
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
    elif rawname == "sos":
        ipath = Path(obsdir, "sos_ORAS5_mon_195801-202212.nc")
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
    elif "time_counter" in dataset.dims:
        dataset = dataset.rename(dict(time_counter="time"))

    return dataset


def _get_or_create_land_mask(diagdir: str, slug: str) -> xarray.DataArray:
    """Return a 2-D land mask (True = land) for an ORCA-grid member.

    The mask is derived from the pre-computed ``zos_anom_*_daily.*`` file in
    $DIAGSDIR and cached as ``land_mask_{slug}.zarr`` so it is only computed
    once.  HadGEM3 uses 0 as a fill value, so zeros are converted to NaN
    before the all-time reduction.
    """
    mask_path = Path(diagdir, f"land_mask_{slug}.zarr")
    if mask_path.exists():
        return xarray.open_zarr(mask_path)["land_mask"]

    zos_zarr = Path(diagdir, f"zos_anom_{slug}_daily.zarr")
    zos_nc = Path(diagdir, f"zos_anom_{slug}_daily.nc")
    if zos_zarr.exists():
        zos_ds = xarray.open_zarr(zos_zarr)
    else:
        zos_ds = xarray.open_dataset(zos_nc, chunks="auto")

    logger.info(f"Computing land mask for {slug} from zos_anom, saving to {mask_path}")
    zos_da = zos_ds["zos"].astype("float32").where(lambda x: x != 0)
    land_mask = zos_da.isnull().all("time").compute()
    land_mask.to_dataset(name="land_mask").to_zarr(mask_path, mode="w")
    return land_mask


def _fix_eke_artifacts(dataset: xarray.Dataset, diagdir: str, slug: str) -> xarray.Dataset:
    """Fix known EKE artifacts in pre-computed diagnostic zarrs (for visualisation purposes only!).

    1. Zero → NaN: HadGEM3 uses 0 as fill value instead of NaN.
    2. Seam fix: roll the array so the 0°/360° lon boundary is not at the edge,
       fill up to 10 pixels in each direction, then roll back. Fixes the black
       meridional line caused by gradient artefacts at the seam in IFS-NEMO and HadGEM3.
    3. Land mask restore: use the zos_anom land mask (free of seam artefacts)
       to re-mask ocean/land after the fill.
    4. Equatorial mask (±3°): geostrophic EKE is unreliable near the equator
       due to vanishing Coriolis. Applied here rather than relying on eke.py
       so we are sure that pre-computed zarrs are corrected consistently.
    """
    dataset = dataset.where(dataset != 0)
    land_mask = _get_or_create_land_mask(diagdir, slug)
    n = dataset.sizes["lon"]
    dataset = (
        dataset.roll(lon=n // 2, roll_coords=True)
        .ffill("lon", limit=10)
        .bfill("lon", limit=10)
        .roll(lon=-(n // 2), roll_coords=True)
    )
    dataset = dataset.where(~land_mask)
    dataset = dataset.where((dataset.lat < -3) | (dataset.lat > 3))
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
    final_member = member.slug if hasattr(member, "slug") else EERIEMember.from_string(member).slug
    zarr_file = Path(diagdir, f"{rawname}_{final_member}_monthly.zarr")
    nc_file = Path(diagdir, f"{rawname}_{final_member}_monthly.nc")
    if zarr_file.exists():
        dataset = xarray.open_zarr(zarr_file)
    else:
        dataset = xarray.open_dataset(nc_file, chunks="auto")
    dataset = dataset[[rawname]].astype("float32")
    if rawname == "eke":
        dataset = _fix_eke_artifacts(dataset, diagdir, final_member)
    time_index = dataset.time.to_index()
    logger.info(f"Time span for {member} is {time_index[0]} from {time_index[-1]}")
    return dataset
