from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

from eerieview.constants import NUM2MONTH

InputLocation = Literal["levante", "cloud"]
DecadalProduct = Literal["clim", "trend"]
EERIEProduct = Literal["series"] | DecadalProduct
Period = Tuple[int, int]


@dataclass
class EERIEMember:
    # 'ifs-fesom2-sr.hist-1950.v20240304.atmos.gr025.2D_monthly_avg'
    model: str
    simulation: str
    version: str
    grid: str
    cmor_table: str

    @classmethod
    def from_string(cls, member_str: str) -> "EERIEMember":
        pieces = member_str.split(".")
        if len(pieces) != 5:
            raise RuntimeError("Member string must have 5 sections separated by points")
        return cls(*pieces)

    @property
    def slug(self) -> str:
        return f"{self.model}-{self.simulation}"


@dataclass
class ObservationPaths:
    era5: Path


@dataclass
class ModelsPaths:
    control: str
    hist: str
    amip: str


@dataclass
class OutputPaths:
    obs: Path
    models: Path


@dataclass
class Paths:
    obs: ObservationPaths
    output: OutputPaths


@dataclass
class TimeFilter:
    """Int of month str if season."""

    freq: str | int
    units: str

    def to_str(self):
        return NUM2MONTH[self.freq] if self.units == "month" else self.freq

    def get_minvalues(self, original_freq: str = "daily"):
        if original_freq == "daily":
            units2minvalues = {"month": 28, "season": 80, "year": 354}
        elif original_freq == "monthly":
            units2minvalues = {"month": 1, "season": 3, "year": 12}
        else:
            raise NotImplementedError(f"Unkown original freq {original_freq}")
        return units2minvalues[self.units]


@dataclass
class PeriodsConfig:
    reference_period: Period
    periods: list[Period]

    @property
    def all_list(self) -> list[Period]:
        return [
            self.reference_period,
        ] + self.periods
