# -*- coding: utf-8 -*-
"""Calibration of EDGES data."""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: nocover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from .cal_coefficients import (
    S1P,
    Calibration,
    CalibrationObservation,
    EdgesFrequencyRange,
    FrequencyRange,
    LoadSpectrum,
    SwitchCorrection,
)
