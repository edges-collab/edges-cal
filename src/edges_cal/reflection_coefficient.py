"""Functions for working with reflection coefficients."""
from __future__ import annotations

import numpy as np


def impedance2gamma(
    z: float | np.ndarray,
    z0: float | np.ndarray,
) -> float | np.ndarray:
    """Convert impedance to reflection coefficient.

    Parameters
    ----------
    z : float or array
        Impedance.
    z0 : float or array
        Impedance match(?).

    Returns
    -------
    gamma : float or array
        The reflection coefficient.
    """
    return (z - z0) / (z + z0)


def gamma2impedance(
    gamma: float | np.ndarray,
    z0: float | np.ndarray,
) -> float | np.ndarray:
    """Convert reflection coeffiency to impedance.

    Parameters
    ----------
    gamma : float or array
        Reflection coefficient.
    z0 : float or array
        Matching impedance (?)

    Returns
    -------
    z : float or array
        The impedance.
    """
    return z0 * (1 + gamma) / (1 - gamma)


def gamma_de_embed(s11, s12s21, s22, rp):  # noqa
    return (rp - s11) / (s22 * (rp - s11) + s12s21)


def gamma_shifted(s11, s12s21, s22, r):  # noqa
    return s11 + (s12s21 * r / (1 - s22 * r))


def de_embed(r1a, r2a, r3a, r1m, r2m, r3m, rp):  # noqa
    # This only works with 1D arrays, where each point in the array is
    # a value at a given frequency

    # The output is also a 1D array

    s11 = np.zeros(len(r1a)) + 0j  # 0j added to make array complex
    s12s21 = np.zeros(len(r1a)) + 0j
    s22 = np.zeros(len(r1a)) + 0j

    for i in range(len(r1a)):
        b = np.array([r1m[i], r2m[i], r3m[i]])  # .reshape(-1,1)
        A = np.array(
            [
                [1, r1a[i], r1a[i] * r1m[i]],
                [1, r2a[i], r2a[i] * r2m[i]],
                [1, r3a[i], r3a[i] * r3m[i]],
            ]
        )
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        s11[i] = x[0]
        s12s21[i] = x[1] + x[0] * x[2]
        s22[i] = x[2]

    r = gamma_de_embed(s11, s12s21, s22, rp)

    return r, s11, s12s21, s22


def fiducial_parameters_85033E(  # noqa: N802
    r, match_delay: bool = True, md_value_ps: float = 38.0
):
    """Get fiducial parameter for the Agilent 85033E standard kit."""
    # Parameters of open
    open_off_Zo = 50
    open_off_delay = 29.243e-12
    open_off_loss = 2.2 * 1e9
    open_C0 = 49.43e-15
    open_C1 = -310.1e-27
    open_C2 = 23.17e-36
    open_C3 = -0.1597e-45

    op = np.array(
        [open_off_Zo, open_off_delay, open_off_loss, open_C0, open_C1, open_C2, open_C3]
    )

    # Parameters of short
    short_off_Zo = 50
    short_off_delay = 31.785e-12
    short_off_loss = 2.36 * 1e9
    short_L0 = 2.077e-12
    short_L1 = -108.5e-24
    short_L2 = 2.171e-33
    short_L3 = -0.01e-42

    sp = np.array(
        [
            short_off_Zo,
            short_off_delay,
            short_off_loss,
            short_L0,
            short_L1,
            short_L2,
            short_L3,
        ]
    )

    # Parameters of match
    match_off_Zo = 50

    match_off_delay = 0 if not match_delay else md_value_ps * 1e-12
    match_off_loss = 2.3 * 1e9
    match_R = r

    mp = np.array([match_off_Zo, match_off_delay, match_off_loss, match_R])

    return op, sp, mp


def standard(
    f: np.ndarray, par: [list[float], np.ndarray], kind: str
) -> np.ndarray:  # noqa
    """Compute the standard.

    Parameters
    ----------
    f : array-like
        Frequency in Hz.
    par : array-like
        Parameters of the standard.
    kind : str
        Either 'open', 'short' or 'match'.

    Returns
    -------
    standard : array
        The standard.
    """
    assert kind in [
        "open",
        "short",
        "match",
    ], "kind must be one of 'open', 'short', 'match'"

    offset_zo = par[0]
    offset_delay = par[1]
    offset_loss = par[2]

    if kind in ["open", "short"]:
        poly = par[3] + par[4] * f + par[5] * f ** 2 + par[6] * f ** 3

        if kind == "open":
            impedance_termination = -1j / (2 * np.pi * f * poly)
        elif kind == "short":
            impedance_termination = 1j * 2 * np.pi * f * poly
    else:
        impedance_termination = par[3]

    gamma_termination = impedance2gamma(impedance_termination, 50)

    # Transmission line
    zc = (offset_zo + (offset_loss / (2 * 2 * np.pi * f)) * np.sqrt(f / 1e9)) - 1j * (
        offset_loss / (2 * 2 * np.pi * f)
    ) * np.sqrt(f / 1e9)
    temp = ((offset_loss * offset_delay) / (2 * offset_zo)) * np.sqrt(f / 1e9)
    gl = temp + 1j * ((2 * np.pi * f) * offset_delay + temp)

    # Combined reflection coefficient
    r1 = impedance2gamma(zc, 50)
    ex = np.exp(-2 * gl)
    return (r1 * (1 - ex - r1 * gamma_termination) + ex * gamma_termination) / (
        1 - r1 * (ex * r1 + gamma_termination * (1 - ex))
    )


def agilent_85033E(  # noqa: N802
    f: np.ndarray,
    resistance_of_match: float,
    match_delay: bool = True,
    md_value_ps: float = 38.0,
):
    """Generate open, short and match standards for the Agilent 85033E.

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
    op, sp, mp = fiducial_parameters_85033E(
        resistance_of_match, match_delay=match_delay, md_value_ps=md_value_ps
    )
    o = standard(f, op, "open")
    s = standard(f, sp, "short")
    m = standard(f, mp, "match")

    return o, s, m


def input_impedance_transmission_line(
    z0: np.ndarray, gamma: np.ndarray, length: float, z_load: np.ndarray
) -> np.ndarray:
    """
    Calculate the impedance of a transmission line.

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
