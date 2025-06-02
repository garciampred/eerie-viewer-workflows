from itertools import product

import pytest

from eerieview.constants import members_eerie_hist
from scripts.get_obs_time_series import get_obs_time_series
from scripts.get_time_series import get_model_time_series
from tests.conftest import mocked_get_entry_dataset, mocked_get_obs_dataset


@pytest.mark.parametrize("varname,region_set", product(["tas", "pr"], ["IPCC", "EDDY"]))
def test_get_model_time_series(tmp_path, varname, region_set):
    get_model_time_series(
        varname,
        "cloud",
        tmp_path,
        members_eerie_hist,
        experiment="control",
        reference_period=(1951, 1980),
        region_set=region_set,
        get_entry_dataset_fun=mocked_get_entry_dataset,
    )


@pytest.mark.parametrize("varname", ["tas", "pr", "eke"])
def test_get_obs_time_series(tmp_path, varname):
    source = "AVISO" if varname == "eke" else "ERA5"
    obsdir = tmp_path
    get_obs_time_series(
        varname,
        obsdir,
        tmp_path,
        source=source,
        region_set="IPCC",
        get_obs_dataset_fun=mocked_get_obs_dataset,
    )
