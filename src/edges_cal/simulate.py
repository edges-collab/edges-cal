"""Module with routines for simulating calibration datasets."""
from __future__ import annotations

import numpy as np

from . import receiver_calibration_func as rcf


def simulate_q(
    *,
    load_s11: np.ndarray,
    lna_s11: np.ndarray,
    load_temp: float | np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    t_unc: np.ndarray,
    t_cos: np.ndarray,
    t_sin: np.ndarray,
    t_load: float = 300.0,
    t_load_ns: float = 400.0
) -> np.ndarray:
    """Simulate the observed 3-position switch ratio data, Q.

    Parameters
    ----------
    load_s11 : np.ndarray
        The S11 of the input load (antenna, or calibration source) as a function of
        frequency.
    lna_s11 : np.ndarray
        The S11 of the internal LNA.
    load_temp
        The (calibrated) temperature of the input load.
    scale
        The scale polynomial (C1)
    offset : np.ndarray
        The offset polynomial (C2)
    t_unc : np.ndarray
        The noise-wave parameter T_uncorrelated
    t_cos : np.ndarray
        The noise-wave parameter T_cos
    t_sin : np.ndarray
        The noise-wave parameter T_sin
    t_load : float, optional
        The fiducial internal load temperature, by default 300.0
    t_load_ns : float, optional
        The internal load + noise source temperature, by default 400.0

    Returns
    -------
    q
        The simulated 3-position switch ratio data.
    """
    a, b = rcf.get_linear_coefficients(
        gamma_ant=load_s11,
        gamma_rec=lna_s11,
        sca=scale,
        off=offset,
        t_unc=t_unc,
        t_cos=t_cos,
        t_sin=t_sin,
        t_load=t_load,
    )

    uncal_temp = (load_temp - b) / a
    return (uncal_temp - t_load) / t_load_ns


def simulate_q_from_calobs(
    calobs, load: str, scale_model=None, freq=None
) -> np.ndarray:
    """Simulate the observed 3-position switch ratio, Q, from noise-wave solutions.

    Parameters
    ----------
    calobs : :class:`~edges_cal.cal_coefficients.CalibrationObservation`
        The calibration observation that contains the solutions.
    load : str
        The load to simulate.

    Returns
    -------
    np.ndarray
        The 3-position switch values.
    """
    default_freq = freq is None
    if freq is None:
        freq = calobs.freq.freq

    C1 = scale_model(freq) if scale_model is not None else calobs.C1(freq)
    try:
        lna_s11 = calobs.lna.s11_model(freq)
    except AttributeError:
        lna_s11 = calobs.lna_s11(freq)

    if not default_freq and load == "hot_load":
        gain = calobs.hot_load.hot_load_correction.power_gain(
            freq, calobs.hot_load.reflections
        )
        # temperature
        temp_ave = (
            gain * calobs.hot_load.spectrum.temp_ave
            + (1 - gain) * calobs.hot_load.ambient.temp_ave
        )
    else:
        temp_ave = calobs._loads[load].temp_ave

    return simulate_q(
        load_s11=calobs._loads[load].reflections.s11_model(freq),
        lna_s11=lna_s11,
        load_temp=temp_ave,
        scale=C1,
        offset=calobs.C2(freq),
        t_unc=calobs.Tunc(freq),
        t_cos=calobs.Tcos(freq),
        t_sin=calobs.Tsin(freq),
        t_load=calobs.t_load,
        t_load_ns=calobs.t_load_ns,
    )


def simulate_qant_from_calobs(
    calobs,
    ant_s11: np.ndarray,
    ant_temp: np.ndarray,
    scale_model=None,
    freq=None,
    loss=1,
    t_amb=296,
    bm_corr=1,
) -> np.ndarray:
    """Simulate antenna Q from a calibration observation.

    Parameters
    ----------
    calobs : :class:`~edges_cal.cal_coefficients.CalibrationObservation`
        The calibration observation that contains the solutions.
    ant_s11
        The S11 of the antenna.
    ant_temp
        The true temperature of the beam-weighted sky.

    Returns
    -------
    np.ndarray
        The simulated 3-position switch ratio, Q.
    """
    if freq is None:
        freq = calobs.freq.freq

    scale = scale_model(freq) if scale_model is not None else calobs.C1(freq)

    ant_temp = loss * ant_temp / bm_corr + (1 - loss) * t_amb

    lna_s11 = (
        calobs.lna_s11(freq) if callable(calobs.lna_s11) else calobs.lna.s11_model(freq)
    )
    return simulate_q(
        load_s11=ant_s11,
        lna_s11=lna_s11,
        load_temp=ant_temp,
        scale=scale,
        offset=calobs.C2(freq),
        t_unc=calobs.Tunc(freq),
        t_cos=calobs.Tcos(freq),
        t_sin=calobs.Tsin(freq),
        t_load=calobs.t_load,
        t_load_ns=calobs.t_load_ns,
    )
