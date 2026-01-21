import importlib
import json
import os
from pathlib import Path

import pandas
import xarray
import zarr
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv

from eerieview.constants import AVISO_VARIABLES, OCEAN_VARIABLES
from eerieview.data_processing import fix_360_longitudes
from eerieview.logger import get_logger
from eerieview.zarr import get_filesystem

load_dotenv()
logger = get_logger(__name__)

member2shortmeber = {
    "icon-esm-er-eerie-control-1950": "icon",
    "ifs-fesom2-sr-eerie-control-1950": "ifs-fesom2",
    "ifs-fesom2-sr-hist-1950": "ifs-fesom2",
    "icon-esm-er-hist-1950": "icon",
    "ifs-nemo-er-hist-1950": "ifs-nemo-er",
    "HadGEM3-GC5-EERIE-N216-ORCA025-eerie-historical": "hadgem3-mediumres",
    "HadGEM3-GC5-EERIE-N640-ORCA12-eerie-historical": "hadgem3-hires",
    "HadGEM3-GC5-EERIE-N96-ORCA1-eerie-historical": "hadgem3-lowres",
    "ifs-amip-tco1279-hist": "ifs-amip-tco1279-hist",
    "ifs-amip-tco1279-hist-c-0-a-lr20": "ifs-amip-tco1279-hist-c-0-a-lr20",
    "ifs-amip-tco399-hist-c-0-a-lr20": "ifs-amip-tco399-hist-c-0-a-lr20",
    "ifs-amip-tco399-hist-c-lr20-a-0": "ifs-amip-tco399-hist-c-lr20-a-0",
    "ifs-amip-tco399-hist": "ifs-amip-tco399-hist",
    "icon-esm-er-highres-future-ssp245": "icon",
    "ifs-fesom2-sr-highres-future-ssp245": "ifs-fesom2",
    "ifs-nemo-er-highres-future-ssp245": "ifs-nemo-er",
    "HadGEM3-GC5-EERIE-N216-ORCA025-eerie-ssp245": "hadgem3-mediumres",
    "HadGEM3-GC5-EERIE-N640-ORCA12-eerie-ssp245": "hadgem3-hires",
    "HadGEM3-GC5-EERIE-N96-ORCA1-eerie-ssp245": "hadgem3-lowres",
}


def get_merged_dataset(ifiles, chunks, drop_member: bool = False):
    to_merge = [
        xarray.open_dataset(f)
        .drop_vars(
            [
                "height2m",
                "height10m",
                "height_2",
                "lev",
                "latitude_longitude",
                "lon_bnds",
                "lat_bnds",
            ],
            errors="ignore",
        )
        .chunk(chunks)
        for f in ifiles
    ]
    if drop_member:
        to_merge = [ds.drop_vars("member") for ds in to_merge]
    dataset = xarray.merge(to_merge)
    return dataset


def get_encoding(variables, product, chunks):
    encoding = {}
    encoding_var = dict(dtype="float32", chunks=tuple(chunks.values()))
    for v in variables:
        encoding[v] = encoding_var
        if product in ["clim", "series"]:
            encoding[v + "_anom"] = encoding_var
        elif product == "trend":
            encoding[v + "_pvalue"] = encoding_var
        else:
            raise RuntimeError(f"Unknown product {product}")

    return encoding


def shorten_members(dataset):
    dataset["member"] = dataset["member"].to_index().map(member2shortmeber)
    assert not dataset.member.isnull().any()
    return dataset


def upload_eerie_climatologies(
    variables: list[str], product: str = "clim", experiment: str = "control", grid="025"
):
    idir = Path(os.environ["PRODUCTSDIR"], "decadal")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = f"s3://{bucket}/test/decadal/{experiment}_EERIE_{product}.zarr"
    ifiles = [
        f"{idir}/{varname}_{experiment}_EERIE_{product}.nc" for varname in variables
    ]
    logger.info(f"Reading  {ifiles}")
    if grid == "025":
        latchunk, lonchunk = 721, 1440
    elif grid == "125":
        latchunk, lonchunk = 1440, 2880
    else:
        raise RuntimeError(f"Unsupported {grid=}")
    chunks = dict(member=1, period=-1, time_filter=1, lat=latchunk, lon=lonchunk)
    dataset = get_merged_dataset(ifiles, chunks)
    dataset = set_cmor_metadata(dataset, product)
    dataset = shorten_members(dataset)
    encoding = get_encoding(variables, product, chunks)
    fs = get_filesystem()
    # Create an S3 store
    store = zarr.storage.FSStore(zarr_url, fs=fs)
    with ProgressBar():
        logger.info(f"Saving {dataset} to {zarr_url}")
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )


def get_obs_file(idir: Path, varname: str, product: str, region_set: str | None = None):
    if varname in AVISO_VARIABLES:
        source = "aviso"
    else:
        source = "era5"
    if region_set is None:
        obs_file = f"{idir}/{varname}_{source}_{product}.nc"
    else:
        obs_file = f"{idir}/{varname}_{source}_{region_set}_{product}.nc"
    return obs_file


def upload_obs_climatologies(variables: list[str], product="clim"):
    logger.info(f"Uploading obs for {product=}")
    idir = Path(os.environ["PRODUCTSDIR"], "decadal")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = f"s3://{bucket}/decadal/obs_{product}.zarr"
    ifiles = [get_obs_file(idir, varname, product) for varname in variables]
    chunks = dict(period=1, time_filter=1, lat=721, lon=1440)
    dataset = get_merged_dataset(ifiles, chunks)
    dataset = dataset.drop_vars(["height2m", "height10m", "height_2"], errors="ignore")
    dataset = set_cmor_metadata(dataset, product)
    encoding = get_encoding(variables, product, chunks)
    fs = get_filesystem()
    # Create an S3 store
    store = zarr.storage.FSStore(zarr_url, fs=fs)
    with ProgressBar():
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )


def upload_eerie_time_series(variables: list[str], experiment: str, region_set: str):
    logger.info(f"Uploading EERIE time series for {experiment=} {region_set=}")
    idir = Path(os.environ["PRODUCTSDIR"], "time_series")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = f"s3://{bucket}/time_series/{experiment}_EERIE_{region_set}_ts.zarr"
    ifiles = [
        f"{idir}/{varname}_{experiment}_EERIE_{region_set}_ts.nc"
        for varname in variables
    ]
    chunks = dict(time_filter=1, time=-1, region=1)
    dataset = get_merged_dataset(ifiles, chunks)
    dataset = dataset.drop_vars(["height2m", "height10m", "height_3"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "ts")
    if experiment != "hist-amip":
        dataset = shorten_members(dataset)
    encoding = get_encoding(variables, "series", chunks)
    fs = get_filesystem()
    # Create an S3 store
    store = zarr.storage.FSStore(zarr_url, fs=fs)
    with ProgressBar():
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )


def upload_obs_time_series(variables: list[str], region_set: str):
    logger.info(f"Uploading obs time series for {region_set=}")
    idir = Path(os.environ["PRODUCTSDIR"], "time_series")
    bucket = os.environ["S3_BUCKET"]
    zarr_url = f"s3://{bucket}/time_series/obs_{region_set}_ts.zarr"
    ifiles = [
        get_obs_file(idir, varname, "ts", region_set=region_set)
        for varname in variables
    ]
    logger.info(f"Reading {ifiles}")
    chunks = dict(time_filter=1, time=-1, region=1)
    dataset = get_merged_dataset(ifiles, chunks, drop_member=True)
    dataset = dataset.drop_vars(["height2m", "height10m"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "ts")
    encoding = get_encoding(variables, "series", chunks)
    fs = get_filesystem()
    # Create an S3 store
    # store = zarr.storage.FSStore(zarr_url, fs=fs)
    store = fs.get_mapper(zarr_url)
    with ProgressBar():
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )


def upload_eddy_rich_zarr():
    variables = ["uo", "vo"]
    ifile = Path(
        os.environ["PRODUCTSDIR"],
        "misc",
        "icon-esm-er.hist-1950_u_v_ocean_197001_19700212_weekly.nc",
    )
    bucket = os.environ["S3_BUCKET"]
    zarr_url = f"s3://{bucket}/misc/icon-esm-er.hist-1950_u_v_ocean_19700101.zarr"
    logger.info(f"Writing {zarr_url}")
    dataset = xarray.open_dataset(ifile).squeeze().rename(u="uo", v="vo")
    dataset = fix_360_longitudes(dataset)
    dataset = dataset.drop_vars(["depth", "time"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "clim")
    chunks = dict(lat=-1, lon=-1)
    encoding = get_encoding(variables, "misc", chunks)
    fs = get_filesystem()
    # Create an S3 store
    store = zarr.storage.FSStore(zarr_url, fs=fs)
    with ProgressBar():
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )


def upload_eddy_rich_zarr_5lev():
    variables = ["uo", "vo"]
    ifile = Path(
        os.environ["PRODUCTSDIR"],
        "misc",
        "icon-esm-er.hist-1950_u_v_ocean_19700101_5lev.nc",
    )
    bucket = os.environ["S3_BUCKET"]
    zarr_url = f"s3://{bucket}/misc/icon-esm-er.hist-1950_u_v_ocean_19700101_5lev.zarr"
    logger.info(f"Writing {zarr_url}")
    dataset = xarray.open_dataset(ifile).squeeze().rename(u="uo", v="vo")
    dataset = dataset.drop_vars(["time"], errors="ignore")
    dataset = set_cmor_metadata(dataset, "clim")
    chunks = dict(depth=1, lat=-1, lon=-1)
    encoding = get_encoding(variables, "misc", chunks)
    fs = get_filesystem()
    # Create an S3 store
    store = zarr.storage.FSStore(zarr_url, fs=fs)
    with ProgressBar():
        dataset.to_zarr(
            store=store, zarr_format=2, consolidated=True, encoding=encoding, mode="w"
        )


def get_variable_cmor_metadata(varname: str):
    realm = "Omon" if varname in OCEAN_VARIABLES else "Amon"
    cmor_json = Path(
        str(importlib.resources.files("eerieview")),
        f"resources/EERIE_{realm}.json",
    )
    with open(cmor_json, "r") as fileobj:
        table = json.load(fileobj)["variable_entry"]
    df = pandas.DataFrame.from_dict(table, orient="index")
    return df.loc[varname]


def set_cmor_metadata(dataset: xarray.Dataset, product) -> xarray.Dataset:
    for varname in dataset.data_vars:
        varname_noanom = str(varname).replace("_anom", "").replace("_pvalue", "")
        if varname_noanom == "eke":
            attrs = dict(
                standard_name="eddy_kinetic_energy",
                long_name="Eddy Kinetic Energy",
                units="m2/s2",
            )
        else:
            attrs = get_variable_cmor_metadata(varname_noanom)
        for attrname in ["long_name", "standard_name", "units"]:
            attrval = attrs[attrname]
            if attrname == "units":
                if varname_noanom in ["tas", "tasmin", "tasmax", "tos"]:
                    attrval = "degC"
                if varname_noanom == "pr":
                    attrval = "mm day-1"
            if "anom" in str(varname):
                if attrname == "standard_name":
                    attrval += "_anomaly"
                if attrname == "long_name":
                    attrval += " Anomaly"
            dataset[varname].attrs[attrname] = attrval
        if product == "trend":
            dataset[varname].attrs["standard_name"] += "_trend"
            dataset[varname].attrs["long_name"] += " Trend"
    return dataset


def upload_time_series(
    variables: list[str], variables_amip: list[str], region_set: str
):
    upload_obs_time_series(variables, region_set)
    # upload_eerie_time_series(variables, "hist", region_set)
    # upload_eerie_time_series(variables_amip, "hist-amip", region_set)
    # upload_eerie_time_series(variables, "control", region_set)
    # upload_eerie_time_series(variables, "future", region_set)


def main():
    variables = [
        "sfcWind",
        "uas",
        "vas",
        "tas",
        "pr",
        "tos",
        "clt",
        "tasmax",
        "tasmin",
        "zos",
        # "eke",
    ]
    variables_amip = [v for v in variables if v not in ["zos", "eke", "so"]]
    # for product in ["clim", "trend"]:
    #     upload_obs_climatologies(variables, product=product)
    #     for experiment in ["future", ]: #"hist", ]: #"control"]:  # , "hist-amip"]:
    #         if experiment == "hist-amip":
    #             variables_exp = variables_amip
    #         else:
    #             variables_exp = variables
    #         logger.info(f"Uploading {product=} for {experiment=}")
    #         upload_eerie_climatologies(
    #             variables_exp, product=product, experiment=experiment, grid="025"
    #         )
    upload_time_series(variables, variables_amip, "IPCC")
    upload_time_series(variables, variables_amip, "EDDY")


if __name__ == "__main__":
    main()
