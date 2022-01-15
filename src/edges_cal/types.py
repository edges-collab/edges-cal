"""Simple type definitions for use internally."""
from pathlib import Path
from typing import Type, Union

from edges_cal.modelling import Model
from astropy import units

PathLike = Union[str, Path]
Modelable = Union[str, Type[Model]]
FreqType = units.Quantity['frequency']
ImpedanceType = units.Quantity['electrical impedance']
OhmType = units.Quantity[units.ohm]