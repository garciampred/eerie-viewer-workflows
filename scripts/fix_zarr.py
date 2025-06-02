import xarray
import zarr
from dotenv import load_dotenv

from eerieview.zarr import get_filesystem

load_dotenv()

def main():
    fs = get_filesystem()
    store = zarr.storage.FSStore("s3://eerie/decadal/hist_EERIE_clim.zarr", fs=fs)
    dataset = xarray.open_zarr(store=store)
    print(dataset)
    # Save original encoding
    encoding = {"tas": dataset["tas"].encoding }

    # Modify the variable
    attrs = dataset['tas'].attrs.copy()
    dataset['tas'] = dataset['tas'] - (273.15 * 2)
    dataset['tas'].attrs = attrs

    # Write it back, preserving encoding
    dataset.to_zarr(store=store, mode='r+', compute=True, consolidated=True)


if __name__ == "__main__":
    main()
