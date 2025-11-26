from functools import partial

import pytest

from eerieview.constants import members_eerie_control_cmor
from eerieview.data_models import CmorEerieMember, PeriodsConfig
from eerieview.product_computation import get_model_decadal_product
from scripts.get_obs_climatologies import get_obs_decadal_product
from tests.conftest import mocked_get_entry_dataset, mocked_get_obs_dataset


@pytest.mark.parametrize(
    "varname,product",
    [
        ("tos", "clim"),
    ],
)  # ("tas", "clim"), ("pr", "trend")])
def test_get_model_decadal_product(tmp_path, varname, product):
    reference_period = (1951, 1970)
    periods = [(1971, 1990), (1991, 2010)]
    periods_config = PeriodsConfig(reference_period, periods)
    if product == "trend":
        _mocked_get_entry_dataset = partial(mocked_get_entry_dataset, trend=0.2)
    else:
        _mocked_get_entry_dataset = mocked_get_entry_dataset
    dataset = get_model_decadal_product(
        varname=varname,
        location="cloud",
        output_dir=tmp_path,
        product=product,
        get_entry_dataset_fun=_mocked_get_entry_dataset,
        periods=periods_config,
        members=members_eerie_control_cmor,
        member_class=CmorEerieMember,
    )
    print(dataset)


@pytest.mark.parametrize("varname,product", [("tas", "clim"), ("pr", "trend")])
def test_get_obs_decadal_product(tmp_path, varname, product):
    reference_period = (1951, 1970)
    periods = [(1971, 1990), (1991, 2010)]
    periods_config = PeriodsConfig(reference_period, periods)

    get_obs_decadal_product(
        varname=varname,
        obsdir=tmp_path,
        output_dir=tmp_path,
        product=product,
        get_obs_dataset_fun=mocked_get_obs_dataset,
        periods=periods_config,
        source="cloud",
    )
