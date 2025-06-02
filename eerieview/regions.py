from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import geopandas
import numpy
import regionmask
import xarray


@dataclass
class SpatialAggregation:
    """
    Aggregates xarray.Dataset data over regions.

    Attributes
    ----------
    dataset : xarray.Dataset
       The xarray dataset containing the data to be aggregated.
    polygons_info : Union[pathlib.PosixPath, str]
       The path to a GeoJSON file, Shapefile or a Well-Known Text (WKT)
       representation of the regions to be aggregated.
    """

    dataset: xarray.Dataset
    polygons_info: Union[Path, Dict[str, str]]

    def __post_init__(self):
        """Initialize the RegionAggregation object."""
        # Load the GeoJSON file into a GeoDataFrame
        self.polygons_gdf = geopandas.read_file(self.polygons_info)

    def compute(self):
        """

        Compute the aggregation  over the region defined by the polygons and season.

        Returns
        -------
        xarray.Dataset
           A dataset containing the aggregated data for each region and year
           of the given season.
        """
        # Convert the GeoDataFrame to a regionmask object
        regions = regionmask.from_geopandas(
            self.polygons_gdf, names="Name", abbrevs="id"
        )
        regions_mask = regions.mask_3D(self.dataset)
        if regions_mask.values.size == 0:
            bounds = self.polygons_gdf.geometry.bounds.values[0]
            region_dataset = self.dataset.sel(
                lat=(bounds[1] + bounds[3]) / 2,
                lon=(bounds[0] + bounds[2]) / 2,
                method="nearest",
            )
            region_dataset = region_dataset.expand_dims(
                {"region": [self.polygons_gdf.id.values[0]]}
            )
            region_dataset = region_dataset.drop_vars(["lat", "lon"])
        else:
            # Compute latitude weights using the cosine function
            latitude_weights = numpy.cos(numpy.deg2rad(self.dataset["lat"]))
            nlat, nlon = len(self.dataset.lat), len(self.dataset.lon)
            # Weights must havve lat, lon dimensions
            latitude_weights = xarray.DataArray(
                numpy.tile(latitude_weights, nlon).reshape((nlat, nlon), order="F"),
                dims=("lat", "lon"),
                coords=dict(lon=self.dataset.lon, lat=self.dataset.lat),
            )

            # Groups by region using the sum as aggregation function
            region_dataset = self.dataset.weighted(
                regions_mask * latitude_weights
            ).mean(dim=("lon", "lat"), skipna=True)

            region_dataset = region_dataset.swap_dims({"region": "abbrevs"})
            region_dataset = region_dataset.drop_vars(["region", "names"])
            region_dataset = region_dataset.rename_dims({"abbrevs": "region"})
            region_dataset = region_dataset.rename_vars({"abbrevs": "region"})
            region_dataset["region"] = region_dataset["region"].astype(int)
        return region_dataset
