from datetime import datetime

import numpy
import pandas
import xarray


def mock_dataset(
    varname: str,
    freq: str = "D",
    res: float = 1.0,
    time_start: datetime = datetime(2000, 1, 1),
    time_end: datetime = datetime(2001, 12, 31),
    scale: float | None = None,
    offset: float | None = None,
    trend: float | None = None,
) -> xarray.Dataset:
    varname2units = dict(
        tasmax="degC",
        tasmin="degC",
        t="degC",
        pr="mm day-1",
        tas="degC",
        tos="degC",
        avg_tos="degC",
        to="degC",
        tasrange="degC",
        sst="degC",
        clt="percent",
        rsds="W m-2",
        rlds="W m-2",
        rsus="W m-2",
        rlus="W m-2",
        siconc="fraction",
        prsn="mm",
        sfcwind="m s-1",
        sfcwindmax="m s-1",
        psl="Pa",
        mrsos="kg m-2",
        mrro="kg m-2",
        hurs="%",
        huss="kg kg -1",
        evspsbl="mm",
        tprate="mm",
        mean2t="degC",
        eke="m2 s-2",
    )
    times = pandas.date_range(time_start, time_end, freq=freq)
    lats = numpy.arange(-90, 90 + res, res)
    lons = numpy.arange(-180 + res, 180, res)
    generator = numpy.random.default_rng(42)
    data = generator.standard_normal((len(times), len(lats), len(lons)), "float32")
    if scale is not None:
        data = data * scale
    if offset is not None:
        data = data + offset
    if trend is not None:
        trend_data = trend * (times.year - times.year[0])
        data = data + trend_data.values.reshape([-1, 1, 1])
    da = xarray.DataArray(data, coords=[times, lats, lons], dims=["time", "lat", "lon"])
    # Add some points with nans in all steps
    da[:, 0:2, 0:2] = numpy.nan
    # Add nans in two steps at beggining and end
    da[0, 3:5, 3:5] = numpy.nan
    da[1, 3:5, 3:5] = numpy.nan
    da.attrs["units"] = varname2units[varname]
    dataset = da.to_dataset(name=varname)
    return dataset


def mocked_get_entry_dataset(catalogue, member, rawname, location="cloud", trend=None):
    dataset = mock_dataset(
        rawname,
        freq="MS",
        time_start=datetime(1951, 1, 1),
        time_end=datetime(2010, 12, 31),
        trend=trend,
    )
    return dataset


def mocked_get_obs_dataset(obsdir, rawname):
    dataset = mock_dataset(
        rawname,
        freq="MS",
        time_start=datetime(1951, 1, 1),
        time_end=datetime(2010, 12, 31),
    )
    return dataset
