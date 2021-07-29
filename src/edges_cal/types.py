"""Simple type definitions for use internally."""
from pathlib import Path
from typing import Type, Union

from edges_cal.modelling import Model

PathLike = Union[str, Path]
Modelable = Union[str, Type[Model]]
