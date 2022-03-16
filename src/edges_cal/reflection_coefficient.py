"""Functions for working with reflection coefficients.

Most of the functions in this module follow the formalism/notation of

    Monsalve et al., 2016, "One-Port Direct/Reverse Method for Characterizing
    VNA Calibration Standards", IEEE Transactions on Microwave Theory and
    Techniques, vol. 64, issue 8, pp. 2631-2639, https://arxiv.org/pdf/1606.02446.pdf

They represent basic relations between physical parameters of circuits, as measured
with internal standards.
"""
from __future__ import annotations

import attr
import numpy as np
import warnings
from astropy import units
from edges_io import h5

from . import modelling as mdl
from . import types as tp
from .tools import unit_converter


def impedance2gamma(
    z: float | np.ndarray,
    z0: float | np.ndarray,
) -> float | np.ndarray:
    """Convert impedance to reflection coefficient.

    See Eq. 19 of Monsalve et al. 2016.

    Parameters
    ----------
    z
        Impedance.
    z0
        Reference impedance.

    Returns
    -------
    gamma
        The reflection coefficient.
    """
    return (z - z0) / (z + z0)


def gamma2impedance(
    gamma: float | np.ndarray,
    z0: float | np.ndarray,
) -> float | np.ndarray:
    """Convert reflection coeffiency to impedance.

    See Eq. 19 of Monsalve et al. 2016.

    Parameters
    ----------
    gamma
        Reflection coefficient.
    z0
        Reference impedance.

    Returns
    -------
    z
        The impedance.
    """
    return z0 * (1 + gamma) / (1 - gamma)


def gamma_de_embed(
    s11: np.typing.ArrayLike,
    s12s21: np.typing.ArrayLike,
    s22: np.typing.ArrayLike,
    gamma_ref: np.typing.ArrayLike,
) -> np.typing.ArrayLike:
    """Obtain the intrinsic reflection coefficient.

    See Eq. 2 of Monsalve et al., 2016.

    Obtains the instrinsic reflection coefficient from the
    one measured at the reference plane, given a set of
    reflection coefficients.

    Parameters
    ----------
    s11
        The S11 parameter of the two-port network for the port facing the calibration
        plane.
    s12s21
        The product of ``S12*S21`` of the two-port
        network.
    s22
        The S22 of the two-port network for the port facing the device under test (DUT)
    gamma_ref
        The reflection coefficient of the device
        under test (DUT) measured at the reference plane.

    Returns
    -------
    gamma
        The intrinsic reflection coefficient of the DUT.

    See Also
    --------
    gamma_embed
        The inverse function to this one.
    """
    return (gamma_ref - s11) / (s22 * (gamma_ref - s11) + s12s21)


def gamma_embed(
    s11: np.typing.ArrayLike,
    s12s21: np.typing.ArrayLike,
    s22: np.typing.ArrayLike,
    gamma: np.typing.ArrayLike,
) -> np.typing.ArrayLike:
    """Obtain the intrinsic reflection coefficient.

    See Eq. 1 of Monsalve et al., 2016.

    Obtains the instrinsic reflection coefficient from the
    one measured at the reference plane, given a set of
    reflection coefficients.

    Parameters
    ----------
    s11
        The S11 parameter of the two-port network for the port facing the calibration
        plane.
    s12s21
        The product of ``S12*S21`` of the two-port
        network.
    s22
        The S22 of the two-port network for the port facing the device under test (DUT)
    gamma
        The intrinsic reflection coefficient of the device
        under test (DUT);.

    Returns
    -------
    gamma_ref
         The reflection coefficient of the DUT
         measured at the reference plane.

    See Also
    --------
    gamma_de_embed
        The inverse function to this one.
    """
    return s11 + (s12s21 * gamma / (1 - s22 * gamma))


def get_sparams(
    gamma_open_intr: np.ndarray | float,
    gamma_short_intr: np.ndarray | float,
    gamma_match_intr: np.ndarray | float,
    gamma_open_meas: np.ndarray,
    gamma_short_meas: np.ndarray,
    gamma_match_meas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtain network S-parameters from OSL standards and intrinsic reflections of DUT.

    See Eq. 3 of Monsalve et al., 2016.

    Parameters
    ----------
    gamma_open_intr
        The intrinsic reflection of the open standard
        (assumed as true) as a function of frequency.
    gamma_shrt_intr
        The intrinsic reflection of the short standard
        (assumed as true) as a function of frequency.
    gamma_load_intr
        The intrinsic reflection of the load standard
        (assumed as true) as a function of frequency.
    gamma_open_meas
        The reflection of the open standard
        measured at port 1 as a function of frequency.
    gamma_shrt_meas
        The reflection of the short standard
        measured at port 1 as a function of frequency.
    gamma_load_meas
        The reflection of the load standard
        measured at port 1 as a function of frequency.

    Returns
    -------
    s11
        The S11 of the network.
    s12s21
        The product `S12*S21` of the network
    s22
        The S22 of the network.
    """
    gamma_open_intr = gamma_open_intr * np.ones_like(gamma_open_meas)
    gamma_short_intr = gamma_short_intr * np.ones_like(gamma_open_meas)
    gamma_match_intr = gamma_match_intr * np.ones_like(gamma_open_meas)

    s11 = np.zeros(len(gamma_open_intr)) + 0j  # 0j added to make array complex
    s12s21 = np.zeros(len(gamma_open_intr)) + 0j
    s22 = np.zeros(len(gamma_open_intr)) + 0j

    for i in range(len(gamma_open_intr)):
        b = np.array([gamma_open_meas[i], gamma_short_meas[i], gamma_match_meas[i]])
        A = np.array(
            [
                [
                    1,
                    complex(gamma_open_intr[i]),
                    complex(gamma_open_intr[i] * gamma_open_meas[i]),
                ],
                [
                    1,
                    complex(gamma_short_intr[i]),
                    complex(gamma_short_intr[i] * gamma_short_meas[i]),
                ],
                [
                    1,
                    complex(gamma_match_intr[i]),
                    complex(gamma_match_intr[i] * gamma_match_meas[i]),
                ],
            ]
        )
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        s11[i] = x[0]
        s12s21[i] = x[1] + x[0] * x[2]
        s22[i] = x[2]

    return s11, s12s21, s22


def de_embed(
    gamma_open_intr: np.ndarray | float,
    gamma_short_intr: np.ndarray | float,
    gamma_match_intr: np.ndarray | float,
    gamma_open_meas: np.ndarray,
    gamma_short_meas: np.ndarray,
    gamma_match_meas: np.ndarray,
    gamma_ref,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Obtain network S-parameters from OSL standards and intrinsic reflections of DUT.

    See Eq. 3 of Monsalve et al., 2016.

    Parameters
    ----------
    gamma_open_intr
        The intrinsic reflection of the open standard
        (assumed as true) as a function of frequency.
    gamma_shrt_intr
        The intrinsic reflection of the short standard
        (assumed as true) as a function of frequency.
    gamma_load_intr
        The intrinsic reflection of the load standard
        (assumed as true) as a function of frequency.
    gamma_open_meas
        The reflection of the open standard
        measured at port 1 as a function of frequency.
    gamma_shrt_meas
        The reflection of the short standard
        measured at port 1 as a function of frequency.
    gamma_load_meas
        The reflection of the load standard
        measured at port 1 as a function of frequency.
    gamma_ref
        The reflection coefficient of the device
        under test (DUT) at the reference plane.

    Returns
    -------
    gamma
        The intrinsic reflection coefficient of the DUT.
    s11
        The S11 of the network.
    s12s21
        The product `S12*S21` of the network
    s22
        The S22 of the network.
    """
    s11, s12s21, s22 = get_sparams(
        gamma_open_intr,
        gamma_short_intr,
        gamma_match_intr,
        gamma_open_meas,
        gamma_short_meas,
        gamma_match_meas,
    )

    gamma = gamma_de_embed(s11, s12s21, s22, gamma_ref)

    return gamma, s11, s12s21, s22


def input_impedance_transmission_line(
    z0: np.ndarray, gamma: np.ndarray, length: float, z_load: np.ndarray
) -> np.ndarray:
    """
    Calculate the impedance of a terminated transmission line.

    Parameters
    ----------
    z0 : array-like
        Complex characteristic impedance
    gamma : array-like
        Propagation constant
    length : float
        Length of transmission line
    z_load : array-like
        Impedance of termination.

    Returns
    -------
    Impedance of the transmission line.
    """
    return (
        z0
        * (z_load + z0 * np.tanh(gamma * length))
        / (z_load * np.tanh(gamma * length) + z0)
    )


@h5.hickleable()
@attr.s(frozen=True, kw_only=True)
class CalkitStandard:
    """Class representing a calkit standard.

    The standard could be open, short or load/match.
    See the Appendix of Monsalve et al. 2016 for details.

    For all parameters, 'offset' refers to the small transmission
    line section of the standard (not an offset in the parameter).

    Parameters
    ----------
    resistance
        The resistance of the standard termination, either assumed or measured.
    offset_impedance
        Impedance of the transmission line, in Ohms.
    offset_delay
        One-way delay of the transmission line, in picoseconds.
    offset_loss
        One-way loss of the transmission line, unitless.
    """

    resistance: float | tp.ImpedanceType = attr.ib(converter=unit_converter(units.ohm))
    offset_impedance: float | tp.ImpedanceType = attr.ib(
        50.0 * units.ohm, converter=unit_converter(units.ohm)
    )
    offset_delay: float | tp.TimeType = attr.ib(
        30.0 * units.picosecond, converter=unit_converter(units.picosecond)
    )
    offset_loss: float | units.Quantity[units.Gohm / units.s] = attr.ib(
        default=2.2 * units.Gohm / units.s,
        converter=unit_converter(units.Gohm / units.s),
    )

    capacitance_model: mdl.Polynomial | None = attr.ib(default=None)
    inductance_model: mdl.Polynomial | None = attr.ib(default=None)

    @capacitance_model.validator
    def _cap_vld(self, att, val):
        if self.name == "open" and val is None:
            raise ValueError("capacitance_model is required for open standard")

    @inductance_model.validator
    def _ind_val(self, att, val):
        if self.name == "short" and val is None:
            raise ValueError("inductance_model is required for short standard")

    @property
    def name(self) -> str:
        """The name of the standard. Inferred from the resistance."""
        if np.isinf(self.resistance):
            return "open"
        elif self.resistance == 0:
            return "short"
        else:
            return "match"

    @classmethod
    def _verify_freq(cls, freq: np.ndarray | units.Quantity):
        if units.get_physical_type(freq) != "frequency":
            raise TypeError(
                "freq must be a frequency quantity! "
                f"Got {units.get_physical_type(freq)}"
            )

    @property
    def intrinsic_gamma(self) -> float:
        """The intrinsic reflection coefficient of the idealized standard."""
        if np.isinf(self.resistance):
            return 1.0  # np.inf / np.inf
        else:
            return impedance2gamma(self.resistance, 50.0 * units.Ohm)

    def termination_impedance(self, freq: tp.FreqType) -> tp.OhmType:
        """The impedance of the termination of the standard.

        See Eq. 22-25 of M16 for open and short standards. The match standard
        uses the input measured resistance as the impedance.
        """
        self._verify_freq(freq)
        freq = freq.to("Hz").value

        if self.name == "open":
            return (-1j / (2 * np.pi * freq * self.capacitance_model(freq))) * units.ohm
        elif self.name == "short":
            return 1j * 2 * np.pi * freq * self.inductance_model(freq) * units.ohm
        else:
            return self.resistance

    def termination_gamma(self, freq: tp.FreqType) -> tp.DimlessType:
        """Reflection coefficient of the termination.

        Eq. 19 of M16.
        """
        return impedance2gamma(self.termination_impedance(freq), 50 * units.ohm)

    def lossy_characteristic_impedance(self, freq: tp.FreqType) -> tp.OhmType:
        """Obtain the lossy characteristic impedance of the transmission line (offset).

        See Eq. 20 of Monsalve et al., 2016
        """
        self._verify_freq(freq)
        return self.offset_impedance + (1 - 1j) * (
            self.offset_loss / (2 * 2 * np.pi * freq)
        ) * np.sqrt(freq.to("GHz").value)

    def gl(self, freq: tp.FreqType) -> np.ndarray:
        """Obtain the product gamma*length.

        gamma is the propagation constant of the transmission line (offset) and l
        is its length. See Eq. 21 of Monsalve et al. 2016.
        """
        self._verify_freq(freq)

        temp = (
            np.sqrt(freq.to("GHz").value)
            * (self.offset_loss * self.offset_delay)
            / (2 * self.offset_impedance)
        )
        return ((2 * np.pi * freq * self.offset_delay) * 1j + (1 + 1j) * temp).to_value(
            ""
        )

    def offset_gamma(self, freq: tp.FreqType) -> tp.DimlessType:
        """Obtain reflection coefficient of the offset.

        Eq. 19 of M16.
        """
        return impedance2gamma(
            self.lossy_characteristic_impedance(freq), 50 * units.ohm
        )

    def reflection_coefficient(self, freq: tp.FreqType) -> tp.DimlessType:
        """Obtain the combined reflection coefficient of the standard.

        See Eq. 18 of M16.
        """
        ex = np.exp(-2 * self.gl(freq))
        r1 = self.offset_gamma(freq)
        gamma_termination = self.termination_gamma(freq)
        return (r1 * (1 - ex - r1 * gamma_termination) + ex * gamma_termination) / (
            1 - r1 * (ex * r1 + gamma_termination * (1 - ex))
        ).value


def CalkitOpen(**kwargs) -> CalkitStandard:  # noqa: N802
    """Factory function for creating Open standards, with resistance=inf.

    See :class:`CalkitStandard` for all parameters available.
    """
    return CalkitStandard(resistance=np.inf * units.ohm, **kwargs)


def CalkitShort(**kwargs) -> CalkitStandard:  # noqa: N802
    """Factor function for creating Short standards, with resistance=0.

    See :class:`CalkitStandard` for all parameters available.
    """
    return CalkitStandard(resistance=0 * units.ohm, **kwargs)


def CalkitMatch(resistance=50.0 * units.ohm, **kwargs) -> CalkitStandard:  # noqa: N802
    """Create a Match standard.

    See :class:`CalkitStandard` for all possible parameters.
    """
    return CalkitStandard(resistance=resistance, **kwargs)


@h5.hickleable()
@attr.s(frozen=True)
class Calkit:
    open: CalkitStandard = attr.ib()
    short: CalkitStandard = attr.ib()
    match: CalkitStandard = attr.ib()

    @open.validator
    def _open_vld(self, att, val):
        assert val.name == "open"

    @short.validator
    def _short_vld(self, att, val):
        assert val.name == "short"

    @match.validator
    def _match_vld(self, att, val):
        assert val.name == "match"

    def clone(self, *, short=None, open=None, match=None):
        """Return a clone with updated parameters for each standard."""
        return attr.evolve(
            self,
            open=attr.evolve(self.open, **(open or {})),
            short=attr.evolve(self.short, **(short or {})),
            match=attr.evolve(self.match, **(match or {})),
        )


AGILENT_85033E = Calkit(
    open=CalkitOpen(
        offset_impedance=50.0 * units.ohm,
        offset_delay=29.243 * units.picosecond,
        offset_loss=2.2 * units.Gohm / units.s,
        capacitance_model=mdl.Polynomial(
            parameters=[49.43e-15, -310.1e-27, 23.17e-36, -0.1597e-45]
        ),
    ),
    short=CalkitShort(
        offset_impedance=50.0 * units.ohm,
        offset_delay=31.785 * units.picosecond,
        offset_loss=2.36 * units.Gohm / units.s,
        inductance_model=mdl.Polynomial(
            parameters=[2.077e-12, -108.5e-24, 2.171e-33, -0.01e-42]
        ),
    ),
    match=CalkitMatch(
        offset_impedance=50.0 * units.ohm,
        offset_delay=38.0 * units.picosecond,
        offset_loss=2.3 * units.Gohm / units.s,
    ),
)


def get_calkit(
    base,
    resistance_of_match: tp.ImpedanceType | None = None,
    open: dict | None = None,
    short: dict | None = None,
    match: dict | None = None,
):
    """Get a calkit based on a provided base calkit, with given updates.

    Parameters
    ----------
    base
        The base calkit to use, eg. AGILENT_85033E
    resistance_of_match
        The resistance of the match, overwrites default from the base.
    open
        Dictionary of parameters to overwrite the open standard.
    short
        Dictionary of parameters to overwrite the short standard.
    match
        Dictionary of parameters to overwrite the match standard.
    """
    match = match or {}
    if resistance_of_match:
        match.update(resistance=resistance_of_match)
    return base.clone(short=short, open=open, match=match)


def agilent_85033E(  # noqa: N802
    f: np.ndarray,
    resistance_of_match: float,
    match_delay: bool = True,
    md_value_ps: float = 38.0,
):
    """Generate open, short and match standards for the Agilent 85033E.

    Note: this function is deprecated. Please use the methods of the Calkit objects
    instead!

    Parameters
    ----------
    f : np.ndarray
        Frequencies in MHz.
    resistance_of_match : float
        Resistance of the match standard, in Ohms.
    match_delay : bool
        Whether to match the delay offset.
    md_value_ps : float
        Some number that does something to the delay matching.

    Returns
    -------
    o, s, m : np.ndarray
        The open, short and match standards.
    """
    warnings.warn(
        "This function is deprecated. Use the methods of your Calkit object directly!",
        category=DeprecationWarning,
    )
    calkit = get_calkit(
        AGILENT_85033E,
        resistance_of_match=resistance_of_match * units.ohm,
        match={
            "offset_delay": md_value_ps * units.picosecond
            if match_delay
            else 0 * units.picosecond
        },
    )

    return (
        calkit.open.reflection_coefficient(f * units.MHz),
        calkit.short.reflection_coefficient(f * units.MHz),
        calkit.match.reflection_coefficient(f * units.MHz),
    )
