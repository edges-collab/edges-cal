# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from .cal_coefficients import (
    VNA,
    Calibration,
    CalibrationObservation,
    EdgesFrequencyRange,
    FrequencyRange,
    LoadSpectrum,
    SwitchCorrection,
)
