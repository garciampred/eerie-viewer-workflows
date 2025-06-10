import dask
import numpy
import xarray

from eerieview.trends.regression import ltr_OLSdofrNaN


def _compute_trend(input_arr, years):
    # xarray apply_ufunct moves the core dimensions to the right! That is why time
    # goes at the right.
    res = ltr_OLSdofrNaN(years, input_arr)
    return res[0], res[5]


_compute_trend = numpy.vectorize(
    _compute_trend, otypes=[numpy.float32, numpy.float32], signature="(nt),(nt)->(),()"
)


def compute_trend(
    varname, dataset, as_decadal: bool = False
) -> tuple[xarray.DataArray, xarray.DataArray]:
    """
    Compute the trend and p-value for the specified variable in the dataset.

    Parameters
    ----------
        varname (str): Name of the variable to compute trend for.
        dataset (xarray.Dataset): Input dataset containing the variable and time information.

    Returns
    -------
        tuple: A tuple of xarray.Datasets containing the computed trend and p-value, respectively.
    """
    attrs = dataset[varname].attrs
    years = dataset["time.year"].astype("float32")
    meta_array = numpy.empty(shape=dataset[varname].shape, dtype="float32")
    # This will let numba do the parallelization/scheduling, and use dask only for
    # keeping memory usage low. Both things at the same time work poorly.
    with dask.config.set(scheduler="synchronous"):
        result = xarray.apply_ufunc(
            _compute_trend,
            dataset[varname],
            years,
            input_core_dims=[["time"], ["time"]],
            vectorize=False,
            dask="parallelized",
            output_core_dims=[[], []],
            dask_gufunc_kwargs=dict(meta=[meta_array, meta_array]),
        )
        trend, p_value = dask.compute(result[0], result[1], optimize_graph=True)

    # Create a mask where p_value is not NaN
    not_nan_mask = ~numpy.isnan(p_value.values)

    # Apply this mask to trend, setting NaNs to 0 where p_value is not NaN
    trend.values[numpy.isnan(trend.values) & not_nan_mask] = 0

    if as_decadal:
        trend = trend * 10
        attrs["units"] = attrs.get("units", "") + "decade -1"
    else:
        attrs["units"] = attrs.get("units", "") + "year -1"
    trend.attrs = attrs
    return trend, p_value
