"""Deprecated module that is an alias for calobs."""

import warnings

from .calobs import *  # noqa: F403

warnings.warn(
    "Do not use cal_coefficients, but instead the calobs module.",
    category=DeprecationWarning,
    stacklevel=2,
)
