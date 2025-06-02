import numpy as np
import xarray as xr


def get_grid_dataset(res: float) -> xr.Dataset:
    """Generate a global regular grid dataset with dummy temperature data.

    Parameters
    ----------
    res : float
        The resolution of the grid in degrees (e.g., 0.25 for a 0.25-degree grid).

    Returns
    -------
    xarray.Dataset
        A dataset containing a global regular grid with 'lat' and 'lon' coordinates
        and a 'temperature' data variable.
    """
    # Define latitude and longitude values for the center of grid cells.
    lats = np.arange(-90 + res / 2, 90, res)
    lons = np.arange(-180 + res / 2, 180, res)
    # Create dummy data
    generator = np.random.default_rng(42)
    # Create dummy temperature data using a normal distribution.
    # Data is scaled and shifted to represent temperatures in Kelvin.
    temp_data = (
        generator.standard_normal((len(lats), len(lons))) * 30 + 273.15
    )  # Random temperatures in Kelvin

    # Create an xarray Dataset to hold the grid and data.
    ds = xr.Dataset(
        {
            "temperature": (
                ("lat", "lon"),
                temp_data,
                {
                    "units": "K",
                    "long_name": "Surface temperature",
                    "coordinates": "lat lon",
                },
            ),
        },
        coords={
            "lat": (
                "lat",
                lats,
                {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"},
            ),
            "lon": (
                "lon",
                lons,
                {"units": "degrees_east", "standard_name": "longitude", "axis": "X"},
            ),
        },
        attrs={
            "description": "Example netCDF file with a global regular grid (0.125-degree resolution)",
            "history": "Created using Python xarray library",
            "source": "Synthetic data",
        },
    )
    return ds
