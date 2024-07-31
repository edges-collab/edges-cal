"""Deprecated module that is an alias for noise_waves."""

import warnings

from .noise_waves import *  # noqa: F403

warnings.warn(
    "Do not use receiver_calibration_func, instead use noise_waves",
    category=DeprecationWarning,
    stacklevel=2,
)
