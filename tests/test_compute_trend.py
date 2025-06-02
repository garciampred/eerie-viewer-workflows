from datetime import datetime

from eerieview.trends.api import compute_trend
from tests.conftest import mock_dataset


def test_compute_trend():
    varname = "tas"
    trend = 0.02
    dataset = mock_dataset(
        "tas",
        freq="YS",
        trend=trend,
        time_start=datetime(1951, 1, 1),
        time_end=datetime(1970, 12, 31),
    )
    trend_dataset = compute_trend(varname, dataset, as_decadal=False)
    assert (trend_dataset[0].mean().item() - trend) < 0.01

    trend_dataset = compute_trend(varname, dataset, as_decadal=True)
    assert (trend_dataset[0].mean().item() - trend * 10) < 0.01
