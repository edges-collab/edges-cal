"""Calibration of EDGES data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edges-cal")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"

from . import modelling, plot
from .calobs import CalibrationObservation, Calibrator, LoadSpectrum
from .s11 import InternalSwitch, LoadS11, Receiver
from .tools import FrequencyRange

del Path
