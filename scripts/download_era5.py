import zipfile
from pathlib import Path

import cdsapi


def main_c3s():
    variables = [
        "monthly_mean_of_daily_mean_wind_speed",
        "monthly_fraction_of_cloud_cover",
        "monthly_mean_of_daily_maximum_temperature",
        "monthly_mean_of_daily_minimum_temperature",
        "monthly_mean_of_daily_mean_temperature",
        "monthly_mean_of_daily_accumulated_precipitation",
        "monthly_mean_of_sea_ice_area_percentage",
        "monthly_mean_of_sea_surface_temperature",
    ]
    odir = Path("/oldhome/users/garciam/temp/eerie/era5")
    for variable in variables:
        target = Path(odir, f"{variable}era5_monthly_c3s.zip")
        dataset = "multi-origin-c3s-atlas"
        request = {
            "origin": "era5",
            "domain": "global",
            "period": "1940-2022",
            "variable": variable,
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request, target)
        with zipfile.ZipFile(target, "r") as zip_ref:
            zip_ref.extractall(odir)
        target.unlink()


def main_era5():
    odir = "/oldhome/users/garciam/temp/eerie/obs/"
    target = f"{odir}/era5_monthly2.nc"
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": [str(year) for year in range(1940, 2025)],
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request, target)


if __name__ == "__main__":
    main_era5()
