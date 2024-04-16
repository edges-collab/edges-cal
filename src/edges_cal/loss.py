"""Models for loss through cables."""
from __future__ import annotations

import numpy as np

from . import ee
from . import types as tp


def get_cable_loss_model(
    cable: ee.CoaxialCable | str, cable_length: tp.LengthType
) -> callable:
    """Return a callable loss model for a particular cable.

    The returned function is suitable for passing to a :class:`Load`
    as the loss_model.
    """
    if isinstance(cable, str):
        cable = ee.KNOWN_CABLES[cable]

    def loss_model(freq, s11a):
        sparams = cable.scattering_parameters(freq, line_length=cable_length)
        s11 = s22 = sparams[0][0]
        s12 = s21 = sparams[0][1]

        T = (s11a - s11) / (s12 * s21 - s11 * s22 + s22 * s11a)
        return (
            np.abs(s12 * s21)
            * (1 - np.abs(T) ** 2)
            / ((1 - np.abs(s11a) ** 2) * np.abs(1 - s22 * T) ** 2)
        )

    return loss_model
