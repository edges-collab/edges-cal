"""Calibration of EDGES data."""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: nocover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"

from .cal_coefficients import (  # noqa: E402
    S1P,
    Calibration,
    CalibrationObservation,
    FrequencyRange,
    LoadS11,
    LoadSpectrum,
)
from .s11_correction import InternalSwitch  # noqa: E402
