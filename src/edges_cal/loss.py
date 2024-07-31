"""Models for loss through cables."""

from __future__ import annotations

from functools import cached_property

import attrs
import numpy as np
from astropy import units as un
from hickleable import hickleable
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from . import ee
from . import modelling as mdl
from . import reflection_coefficient as rc
from . import types as tp
from .tools import FrequencyRange, get_data_path


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


@hickleable()
@attrs.define(slots=False, frozen=True, kw_only=True)
class HotLoadCorrection:
    """
    Corrections for the hot load.

    Measurements required to define the HotLoad temperature, from Monsalve et al.
    (2017), Eq. 8+9.

    Parameters
    ----------
    path
        Path to a file containing measurements of the semi-rigid cable reflection
        parameters. A preceding colon (:) indicates to prefix with DATA_PATH.
        The default file was measured in 2015, but there is also a file included
        that can be used from 2017: ":semi_rigid_s_parameters_2017.txt".
    f_low, f_high
        Lowest/highest frequency to retain from measurements.
    n_terms
        The number of terms used in fitting S-parameters of the cable.
    """

    freq: FrequencyRange = attrs.field()
    raw_s11: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    raw_s12s21: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    raw_s22: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))

    model: mdl.Model = attrs.field(default=mdl.Polynomial(n_terms=21))
    complex_model: type[mdl.ComplexRealImagModel] | type[mdl.ComplexMagPhaseModel] = (
        attrs.field(default=mdl.ComplexMagPhaseModel)
    )
    use_spline: bool = attrs.field(default=False)
    model_method: str = attrs.field(default="lstsq")

    @classmethod
    def from_file(
        cls,
        path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        set_transform_range: bool = True,
        **kwargs,
    ):
        """Instantiate the HotLoadCorrection from file.

        Parameters
        ----------
        path
            Path to the S-parameters file.
        f_low, f_high
            The min/max frequencies to use in the modelling.
        """
        path = get_data_path(path)

        data = np.genfromtxt(path)
        freq = FrequencyRange(data[:, 0] * un.MHz, f_low=f_low, f_high=f_high)

        data = data[freq.mask]

        if data.shape[1] == 7:  # Original file from 2015
            data = data[:, 1::2] + 1j * data[:, 2::2]
        elif data.shape[1] == 6:  # File from 2017
            data = np.array(
                [
                    data[:, 1] + 1j * data[:, 2],
                    data[:, 3],
                    data[:, 4] + 1j * data[:, 5],
                ]
            ).T

        model = kwargs.pop(
            "model",
            mdl.Polynomial(
                n_terms=21,
                transform=mdl.UnitTransform(
                    range=(freq.min.to_value("MHz"), freq.max.to_value("MHz"))
                ),
            ),
        )

        if hasattr(model.xtransform, "range") and set_transform_range:
            model = attrs.evolve(
                model,
                transform=attrs.evolve(
                    model.xtransform,
                    range=(freq.min.to_value("MHz"), freq.max.to_value("MHz")),
                ),
            )

        return cls(
            freq=freq,
            raw_s11=data[:, 0],
            raw_s12s21=data[:, 1],
            raw_s22=data[:, 2],
            model=model,
            **kwargs,
        )

    def _get_model(self, raw_data: np.ndarray, **kwargs):
        model = self.complex_model(self.model, self.model)
        return model.fit(xdata=self.freq.freq, ydata=raw_data, method=self.model_method)

    def _get_splines(self, data):
        if self.complex_model == mdl.ComplexRealImagModel:
            return (
                Spline(self.freq.freq.to_value("MHz"), np.real(data)),
                Spline(self.freq.freq.to_value("MHz"), np.imag(data)),
            )
        return (
            Spline(self.freq.freq.to_value("MHz"), np.abs(data)),
            Spline(self.freq.freq.to_value("MHz"), np.angle(data)),
        )

    def _ev_splines(self, splines):
        rl, im = splines
        if self.complex_model == mdl.ComplexRealImagModel:
            return lambda freq: rl(freq) + 1j * im(freq)
        return lambda freq: rl(freq) * np.exp(1j * im(freq))

    @cached_property
    def s11_model(self):
        """The reflection coefficient."""
        if not self.use_spline:
            return self._get_model(self.raw_s11)
        splines = self._get_splines(self.raw_s11)
        return self._ev_splines(splines)

    @cached_property
    def s12s21_model(self):
        """The transmission coefficient."""
        if not self.use_spline:
            return self._get_model(self.raw_s12s21)
        splines = self._get_splines(self.raw_s12s21)
        return self._ev_splines(splines)

    @cached_property
    def s22_model(self):
        """The reflection coefficient from the other side."""
        if not self.use_spline:
            return self._get_model(self.raw_s22)
        splines = self._get_splines(self.raw_s22)
        return self._ev_splines(splines)

    def power_gain(self, freq: tp.FreqType, hot_load_s11: np.ndarray) -> np.ndarray:
        """
        Calculate the power gain.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies.
        hot_load_s11 : array
            The S11 of the hot load.

        Returns
        -------
        gain : np.ndarray
            The power gain as a function of frequency.
        """
        return self.get_power_gain(
            {
                "s11": self.s11_model(freq.to_value("MHz")),
                "s12s21": self.s12s21_model(freq.to_value("MHz")),
                "s22": self.s22_model(freq.to_value("MHz")),
            },
            hot_load_s11,
        )

    @staticmethod
    def get_power_gain(
        semi_rigid_sparams: dict, hot_load_s11: np.ndarray
    ) -> np.ndarray:
        """Define Eq. 9 from M17.

        Parameters
        ----------
        semi_rigid_sparams : dict
            A dictionary of reflection coefficient measurements as a function of
            frequency for the semi-rigid cable.
        hot_load_s11 : array-like
            The S11 measurement of the hot_load.

        Returns
        -------
        gain : np.ndarray
            The power gain.
        """
        rht = rc.gamma_de_embed(
            semi_rigid_sparams["s11"],
            semi_rigid_sparams["s12s21"],
            semi_rigid_sparams["s22"],
            hot_load_s11,
        )

        return (
            np.abs(semi_rigid_sparams["s12s21"])
            * (1 - np.abs(rht) ** 2)
            / (
                (np.abs(1 - semi_rigid_sparams["s11"] * rht)) ** 2
                * (1 - np.abs(hot_load_s11) ** 2)
            )
        )
