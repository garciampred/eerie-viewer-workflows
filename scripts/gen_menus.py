"""Generates a spreadsheet with the options for the menus in the viewer."""

import pandas


def main():
    products = ["clim", "trend", "ts"]
    # In the future ssp45, ssp85 will replace hist
    # The Real experiment layout is more complex (there are two kind of hist and controls)
    # https://eerie-project.eu/research/modelling/simulations-descriptions/

    datasets = ["hist", "hist-amip", "control", "obs"]
    members_coupled = ["ifs-fesom", "icon"]  # ifs-nemo + 3x HadGEM
    members_amip = [
        "ifs-amip-tco1279.hist",
        "ifs-amip-tco1279.hist-c-0-a-lr20",
        "ifs-amip-tco399.hist-c-0-a-lr20",
        "ifs-amip-tco399.hist-c-lr20-a-0",
        "ifs-amip-tco399.hist",
    ]
    members_amip_series = [
        "ifs-amip-tco1279-hist",
        "ifs-amip-tco1279-hist-c-0-a-lr20",
        "ifs-amip-tco399-hist-c-0-a-lr20_1",
        "ifs-amip-tco399-hist-c-0-a-lr20_10",
        "ifs-amip-tco399-hist-c-0-a-lr20_2",
        "ifs-amip-tco399-hist-c-0-a-lr20_3",
        "ifs-amip-tco399-hist-c-0-a-lr20_4",
        "ifs-amip-tco399-hist-c-0-a-lr20_5",
        "ifs-amip-tco399-hist-c-0-a-lr20_6",
        "ifs-amip-tco399-hist-c-0-a-lr20_7",
        "ifs-amip-tco399-hist-c-0-a-lr20_8",
        "ifs-amip-tco399-hist-c-0-a-lr20_9",
        "ifs-amip-tco399-hist-c-lr20-a-0_1",
        "ifs-amip-tco399-hist-c-lr20-a-0_10",
        "ifs-amip-tco399-hist-c-lr20-a-0_2",
        "ifs-amip-tco399-hist-c-lr20-a-0_3",
        "ifs-amip-tco399-hist-c-lr20-a-0_4",
        "ifs-amip-tco399-hist-c-lr20-a-0_5",
        "ifs-amip-tco399-hist-c-lr20-a-0_6",
        "ifs-amip-tco399-hist-c-lr20-a-0_7",
        "ifs-amip-tco399-hist-c-lr20-a-0_8",
        "ifs-amip-tco399-hist-c-lr20-a-0_9",
        "ifs-amip-tco399-hist_1",
        "ifs-amip-tco399-hist_10",
        "ifs-amip-tco399-hist_2",
        "ifs-amip-tco399-hist_3",
        "ifs-amip-tco399-hist_4",
        "ifs-amip-tco399-hist_5",
        "ifs-amip-tco399-hist_6",
        "ifs-amip-tco399-hist_7",
        "ifs-amip-tco399-hist_8",
        "ifs-amip-tco399-hist_9",
    ]
    variables_coupled = [
        "sfcWind",
        "uas",
        "vas",
        "tas",
        "pr",
        "tos",
        "clt",
        "zos",
        "tasmax",
        "tasmin",
        "eke",
    ]
    region_sets = ["IPCC", "EDDY"]
    variables_amip = [v for v in variables_coupled if v not in ["zos", "eke"]]
    zarr_template = "{folder}/{dataset}_EERIE_{product}.zarr"

    rows = []
    for product in products:
        folder = "time_series" if product == "ts" else "decadal"
        for dataset in datasets:
            zarr = zarr_template.format(folder=folder, dataset=dataset, product=product)
            # Remove EERIE name from observations
            if dataset == "obs":
                zarr = zarr.replace("_EERIE", "")
            # Set the AMIP members and variables
            if dataset == "hist-amip":
                members = members_amip
                variables = variables_amip
            else:
                members = members_coupled
                variables = variables_coupled
            # Set the quantity
            if product in ["ts", "clim"]:
                quantity = ["value (variable)", "anomaly (variable_anom)"]
            else:
                quantity = [
                    "value (variable with hatching over variable_pvalue > 0.05)"
                ]
            # Set the extra dimensions
            if product in ["clim", "trend"]:
                extra_dims = ["period", "time_filter"]
            else:
                extra_dims = ["region", "time_filter"]
            # For regions, loop over region sets
            if product == "ts":
                if dataset == "hist-amip":
                    members_ts = members_amip_series
                else:
                    members_ts = members
                for region_set in region_sets:
                    zarr_regions = zarr.replace("_ts", f"_{region_set}_ts")
                    row = (
                        dataset,
                        product,
                        members_ts,
                        variables,
                        zarr_regions,
                        quantity,
                        region_set,
                        extra_dims,
                    )
                    rows.append(row)
            else:
                row = (
                    dataset,
                    product,
                    members,
                    variables,
                    zarr,
                    quantity,
                    None,
                    extra_dims,
                )
                rows.append(row)

    df = pandas.DataFrame(
        rows,
        columns=[
            "dataset",
            "product",
            "members",
            "variables",
            "zarr",
            "quantity",
            "region_set",
            "extra_dims",
        ],
    ).sort_values(["dataset", "product"])
    df.to_excel("eerie_menus.xlsx")


if __name__ == "__main__":
    main()
