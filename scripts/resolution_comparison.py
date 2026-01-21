import xarray_regrid
import xarray
from xarray_regrid import Grid


def main():
    ifile_hires = "/work/bm1344/DKRZ/MOHC/HadGEM3-GC5-EERIE-N640-ORCA12/eerie-ssp245/r1i1p1f1/Oday/zos/gr1/v20250425/zos_nemo_u-dq389_1d_20500101-20500201_grid-025.nc"
    ifile_midres = "/work/bm1344/DKRZ/MOHC/HadGEM3-GC5-EERIE-N216-ORCA025/eerie-ssp245/r1i1p1f1/Oday/zos/gr1/v20250425/zos_nemo_u-dg002_1d_20500101-20500201_grid-025.nc"
    ifile_lowres = "/work/bm1344/DKRZ/MOHC/HadGEM3-GC5-EERIE-N96-ORCA1/eerie-ssp245/r1i1p1f1/Oday/zos/gr1/v20250425/zos_nemo_u-dc015_1d_20500101-20500201_grid-025.nc"
    grid_hires = Grid(north=90, east=180, south=-90, west=-180,resolution_lat=0.2, resolution_lon=0.25).create_regridding_dataset(lat_name = 'lat', lon_name='lon')
    grid_midres = Grid(north=90, east=180, south=-90, west=-180, resolution_lat=0.5, resolution_lon=0.5).create_regridding_dataset(lat_name = 'lat', lon_name='lon')
    grid_lowres = Grid(north=90, east=180, south=-90, west=-180, resolution_lat=1.25, resolution_lon=1.25).create_regridding_dataset(lat_name = 'lat', lon_name='lon')
    time_slice = slice("2050-01-01", "2050-02-01")
    ds_hires = xarray.open_dataset(ifile_hires)[["zos"]].sel(time=time_slice).regrid.conservative(grid_hires)
    ds_midres = xarray.open_dataset(ifile_midres)[["zos"]].sel(time=time_slice).regrid.conservative(grid_midres)
    ds_lowres = xarray.open_dataset(ifile_lowres)[["zos"]].sel(time=time_slice).regrid.conservative(grid_lowres)
    ds_hires.to_netcdf("~/hadgem_hires_ssp245_205001.nc")
    ds_midres.to_netcdf("~/hadgem_midres_ssp245_205001.nc")
    ds_lowres.to_netcdf("~/hadgem_lowres_ssp245_205001.nc")


if __name__ == "__main__":
    main()

