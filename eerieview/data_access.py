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
        # WORKAROUND: Some models (HadGEM3, IFS-NEMO) have artifacts at the
        # ORCA grid seam (0°/360° lon): HadGEM3 uses 0 as fill value instead
        # of NaN, and IFS-NEMO leaves NaN at the seam boundary. Both cause a
        # black meridional line in the viewer.
        # Fix: convert zeros to NaN, then fill the seam by rolling the array
        # so the boundary is not at the edge, interpolating, and rolling back.
        # TODO: proper fix is in eerieview/eke.py compute_monthly_eke — fill
        # the seam in zos_daily_anom and nan_mask before computing geostrophic
        # velocities, then regenerate the EKE diagnostic zarrs. However this approach is 
        # enought for visualisation in eerie-viewer
        dataset = dataset.where(dataset != 0)
        # Land mask from zos_anom (same ORCA grid). The eke seam artifact comes
        # from the gradient calculation at the boundary, so zos has valid values
        # at the seam pixels and its all-NaN mask is free of seam artifacts.
        land_mask = _get_or_create_land_mask(diagdir, final_member)
        n = dataset.sizes["lon"]
        dataset = (
            dataset.roll(lon=n // 2, roll_coords=True)
            .ffill("lon", limit=10)
            .bfill("lon", limit=10)
            .roll(lon=-(n // 2), roll_coords=True)
        )
        # Restore land mask from zos_anom — unaffected by eke seam artifacts
        dataset = dataset.where(~land_mask)
        # WORKAROUND: mask the equatorial band (±3°) which has unreliable
        # geostrophic EKE due to vanishing Coriolis. The mask is applied on
        # the native grid in eke.py but its effective width in the output
        # varies with model resolution. Re-applying here ensures a consistent
        # ±3° band across all models after regridding.
        dataset = dataset.where((dataset.lat < -3) | (dataset.lat > 3))
    time_index = dataset.time.to_index()
    logger.info(f"Time span for {member} is {time_index[0]} from {time_index[-1]}")
    return dataset
