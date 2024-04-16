"""Electrical enginerring equations."""

from functools import cached_property

import attrs
import numpy as np
from astropy import constants as cnst
from astropy import units as un
from edges_io import types as tp
from pygsdata.attrs import unit_validator as unv


def skin_depth(freq: tp.FreqType, conductivity: tp.Conducitivity) -> un.Quantity[un.m]:
    """Calculate the skin depth of a conducting material."""
    return np.sqrt(1.0 / (np.pi * freq * cnst.mu0 * conductivity)).to("m")


@attrs.define(frozen=True, slots=False)
class TransmissionLine:
    """A transmission line."""

    freq: tp.FreqType = attrs.field(validator=unv("frequency"))
    resistance = attrs.field(validator=unv(un.ohm / un.m))
    inductance = attrs.field(validator=unv("electromagnetic field strength"))
    conductance = attrs.field(validator=unv("electrical conductivity"))
    capacitance = attrs.field(validator=unv("permittivity"))

    @cached_property
    def angular_freq(self) -> tp.FreqType:
        """The angular frequencies at which to evaluate the transmission line."""
        return 2 * np.pi * 1j * self.freq

    @cached_property
    def characteristic_impedance(self) -> tp.ImpedanceType:
        r"""Calculate the characteristic impedance of a transmission line.

        The characteristic impedance Z 0 {\displaystyle Z_{0}} of a transmission line
        is the ratio of the amplitude of a single voltage wave to its current wave.

        https://en.wikipedia.org/wiki/Transmission_line
        """
        return np.sqrt(
            (self.resistance + self.angular_freq * self.inductance)
            / (self.conductance + self.angular_freq * self.capacitance)
        ).to("ohm")

    @cached_property
    def propagation_constant(self) -> un.Quantity[1 / un.m]:
        """Calculate the propagation constant of a transmission line.

        https://en.wikipedia.org/wiki/Transmission_line#General_case_of_a_line_with_losses
        """
        return np.sqrt(
            (self.resistance + self.angular_freq * self.inductance)
            * (self.conductance + self.angular_freq * self.capacitance)
        ).to("1/m")

    def input_impedance(
        self,
        load_impedance: tp.ImpedanceType,
        line_length: tp.LengthType,
    ):
        """Calculate the "input impedance" of a transmission line.

        https://en.wikipedia.org/wiki/Transmission_line#Input_impedance_of_transmission_line

        Parameters
        ----------
        freq : tp.FreqType
            Frequency of the signal.
        """
        return (
            self.characteristic_impedance
            * (
                load_impedance
                + self.characteristic_impedance
                * np.tanh(self.propagation_constant * line_length)
            )
            / (
                self.characteristic_impedance
                + load_impedance * np.tanh(self.propagation_constant * line_length)
            )
        )

    def reflection_coefficient(
        self,
        load_impedance: tp.ImpedanceType,
    ):
        """Calculate the reflection coefficient of a transmission line.

        This is the reflections coefficient measured at the load end of a transmission
        line.

        https://en.wikipedia.org/wiki/Transmission_line
          #Input_impedance_of_transmission_line
        """
        return (load_impedance - self.characteristic_impedance) / (
            load_impedance + self.characteristic_impedance
        )

    def scattering_parameters(
        self,
        load_impedance: tp.ImpedanceType,
        line_length: tp.LengthType,
    ) -> list[list[np.ndarray]]:
        """Calculate the S11 parameter of a transmission line.

        This is the reflection coefficient of the transmission line in the case
        of matched loads at each termination.

        https://en.wikipedia.org/wiki/Transmission_line#Scattering_parameters
        """
        Zo = self.characteristic_impedance
        Zp = load_impedance

        γ = self.propagation_constant
        γl = γ * line_length

        denom = (Zo**2 + Zp**2) * np.sinh(γl) + 2 * Zo * Zp * np.cosh(γl)

        s11 = s22 = (Zo**2 - Zp**2) * np.sinh(γl) / denom
        s12 = s21 = 2 * Zo * Zp / denom
        return [[s11, s12], [s21, s22]]
