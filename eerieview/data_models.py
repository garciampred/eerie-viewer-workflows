import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Literal, Tuple

from eerieview.constants import NUM2MONTH

InputLocation = Literal["levante", "cloud", "levante_cmor"]
DecadalProduct = Literal["clim", "trend"]
EERIEProduct = Literal["series"] | DecadalProduct
Period = Tuple[int, int]


@dataclass
class Member:
    model: str
    simulation: str
    version: str
    grid: str
    npieces: ClassVar[int] = 4

    @classmethod
    def from_string(cls, member_str: str) -> "Member":
        pieces = member_str.split(".")
        if len(pieces) != cls.npieces:
            raise RuntimeError(
                f"Member string must have {cls.npieces} sections separated by points"
            )
        return cls(*pieces)

    @property
    def slug(self) -> str:
        return f"{self.model}-{self.simulation}"

    def to_string(self) -> str:
        return ".".join(str(getattr(self, f.name)) for f in fields(self))

    def to_ocean(self) -> "Member":
        raise NotImplementedError

    def to_atmos(self) -> "Member":
        raise NotImplementedError

    def to_daily(self) -> "Member":
        raise NotImplementedError


@dataclass
class EERIEMember(Member):
    # 'ifs-fesom2-sr.hist-1950.v20240304.atmos.gr025.2D_monthly_avg'
    realm: str
    freq: str
    npieces: ClassVar[int] = 6

    def to_ocean(self) -> "EERIEMember":
        return copy.replace(self, realm="ocean")

    def to_atmos(self) -> "EERIEMember":
        return copy.replace(self, realm="atmos")

    def to_daily(self) -> "EERIEMember":
        return copy.replace(self, freq="daily")


@dataclass
class CmorEerieMember(Member):
    # 'ifs-nemo-er.hist-1950.v20250516.gr025.Amon'
    cmor_table: str
    npieces: ClassVar[int] = 5

    def to_ocean(self) -> "CmorEerieMember":
        return copy.replace(self, cmor_table=self.cmor_table.replace("A", "O"))

    def to_atmos(self) -> "CmorEerieMember":
        return copy.replace(self, cmor_table=self.cmor_table.replace("O", "A"))

    def to_daily(self) -> "CmorEerieMember":
        if self.cmor_table == "Omon":
            new_table = "Oday"
        else:
            new_table = "day"
        return copy.replace(self, cmor_table=new_table)


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
