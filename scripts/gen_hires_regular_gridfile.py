"""Generate a netCDF file with a regular grid, to be used with the CDO for regridding."""

from eerieview.grids import get_grid_dataset


def create_grid(filename: str = "gr0125.nc"):
    # Define grid resolution
    res = 0.125

    ds = get_grid_dataset(res)

    # Save to netCDF file
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")
    print(f"NetCDF file '{filename}' created successfully.")


if __name__ == "__main__":
    create_grid()
