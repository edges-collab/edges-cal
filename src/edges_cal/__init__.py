"""Calibration of EDGES data."""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"

from . import plot  # noqa: E402
from .cal_coefficients import (  # noqa: E402
    CalibrationObservation,
    Calibrator,
    LoadSpectrum,
)
from .s11 import InternalSwitch, LoadS11, Receiver  # noqa: E402
from .tools import FrequencyRange  # noqa: E402
