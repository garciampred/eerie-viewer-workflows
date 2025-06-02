"""PLot stripes figures for the menu combinations to help with the QA."""

import os
from pathlib import Path

import pandas
import seaborn as sns
import xarray
from dotenv import load_dotenv
from matplotlib import pyplot
from matplotlib.dates import DateFormatter

from eerieview.zarr import get_filesystem

load_dotenv()


def main():
    menus = pandas.read_excel("eerie_menus.xlsx")
    fs = get_filesystem()
    s3_bucket = os.environ["S3_BUCKET"]

    for row in menus.itertuples(index=False):
        print(row)
        zarr_path = f"{s3_bucket}/{row.zarr}"
        product = row.product
        dataset_name = row.dataset
        print(f"Reading {zarr_path}")
        dataset = xarray.open_zarr(fs.get_mapper(zarr_path), consolidated=True)
        if product == "ts":
            dataset["time"] = dataset.time.dt.year

        for variable in eval(row.variables):
            plot_variable(dataset, dataset_name, product, row, variable)
            if product != "trend":
                plot_variable(dataset, dataset_name, product, row, variable + "_anom")


def plot_variable(dataset, dataset_name, product, row, variable):
    if product in ["clim", "trend"]:
        agg_data = dataset[variable].mean(dim=("lat", "lon"))
        if dataset_name != "obs":
            table = agg_data.stack(period_filter=("period", "time_filter")).to_pandas()
        else:
            table = agg_data.to_pandas()
    elif product == "ts":
        agg_data = dataset[variable].mean(dim="region")
        if dataset_name != "obs":
            table = (
                agg_data.stack(member_filter=("member", "time_filter")).to_pandas().T
            )
        else:
            table = agg_data.to_pandas()
    else:
        raise RuntimeError("Unknown product")
    units = dataset[variable].attrs["units"]
    pyplot.figure(figsize=(8, 4))
    ax = sns.heatmap(table, cbar_kws=dict(label=units))
    pyplot.title(f"{variable=} {dataset_name=} {product=}")
    ax.fmt_xdata = DateFormatter("% Y")
    pyplot.tight_layout()
    if product == "ts":
        figure_path = Path(
            "../figures", f"{variable}_{dataset_name}_{product}_{row.region_set}.png"
        )
    else:
        figure_path = Path("../figures", f"{variable}_{dataset_name}_{product}.png")
    print(f"Writing {figure_path}")
    pyplot.savefig(figure_path)


if __name__ == "__main__":
    main()
