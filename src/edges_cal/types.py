"""Various useful type-hint definitions."""

from astropy import units
from edges_io.types import *  # noqa: F403

LengthType = units.Quantity["length"]
Conductivity = units.Quantity["electrical conductivity"]
InductanceType = units.Quantity["electromagnetic field strength"]
