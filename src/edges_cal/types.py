"""Simple type definitions for use internally."""
from astropy import units
from pathlib import Path
from typing import Type, Union

from edges_cal.modelling import Model

PathLike = Union[str, Path]
Modelable = Union[str, Type[Model]]
FreqType = units.Quantity["frequency"]
ImpedanceType = units.Quantity["electrical impedance"]
OhmType = units.Quantity[units.ohm]
TimeType = units.Quantity["time"]
DimlessType = units.Quantity["dimensionless"]
