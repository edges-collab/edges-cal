# -*- coding: utf-8 -*-
"""Functions dealing with correcting S11's."""

import numpy as np
from edges_io import io
from os import path
from pathlib import Path
from typing import Tuple

from . import reflection_coefficient as rc
from .modelling import ModelFit


def _get_parameters_at_temperature(data_path, temp):
    assert temp in [15, 25, 35], "temp must be in 15, 25, 35"

    par = np.genfromtxt(
        path.join(data_path, "parameters_receiver_{}degC.txt".format(temp))
    )

    # frequency
    # TODO: this has magic numbers in it!
    f = np.arange(50, 200.1, 0.25)
    f_norm = f / 150

    # evaluating S-parameters
    models = {}
    for i, kind in enumerate(["s11", "s12s21", "s22"]):
        for j, mag_or_ang in ["mag", "ang"]:
            models[kind + "_" + mag_or_ang] = np.polyval(par[:, j + 2 * i], f_norm)


def high_band_switch_correction(
    data_path: [str, Path], ant_s11: np.ndarray, sw_temp: float, poly_order: int = 10
) -> Tuple[np.ndarray, dict]:
    """Correct for the switch for high-band receiver.

    Parameters
    ----------
    data_path : str or Path
        The path to a folder containing the file "switch_temperatures.txt".
    ant_s11 : np.ndarray
        The uncorrected antenna S11 as a function of frequency.
    sw_temp : float
        The switch temperature.
    poly_order : int
        The order of the polynomial to fit to the corrected S11.

    Returns
    -------
    corr_ant_s11 : np.ndarray
        The corrected antenna S11.
    switch : dict
        The reflection coefficients of the switch.
    """
    data_path = Path(data_path)

    # frequency
    f = np.arange(50, 200.1, 0.25)
    f_norm = f / 150

    # switch temperatures
    temp_all = np.genfromtxt(data_path / "switch_temperatures.txt")
    temp15, temp25, temp35 = temp_all

    temps = [15, 25] if sw_temp <= temp25 else [25, 35]
    temps = np.array(temps)

    models = []
    for temp in temps:
        models.append(_get_parameters_at_temperature(data_path, temp))

    # inter (extra) polating switch S-parameters to input temperature
    switch = {}
    for ikind, kind in enumerate(["s11", "s12s21", "s22"]):
        switch[kind] = 0
        for imag, mag_or_ang in enumerate(["mag", "ang"]):
            key = kind + "_" + mag_or_ang
            low_model = models[0][key]
            high_model = models[1][key]

            # Iterate over frequency.
            new_model = np.array(
                [
                    np.interp(sw_temp, temps, np.array([lm, hm]))
                    for lm, hm in zip(low_model, high_model)
                ]
            )
            new_poly = np.polyfit(f_norm, new_model, poly_order)
            new_val = np.polyval(new_poly, f_norm)

            if not imag:  # mag part
                switch[kind] = new_val
            else:  # ang part
                switch[kind] *= np.cos(new_val) + 1j * np.sin(new_val)

    # corrected antenna S11
    corr_ant_s11 = rc.gamma_de_embed(
        switch["s11"], switch["s12s21"], switch["s22"], ant_s11
    )

    # returns corrected antenna measurement
    return corr_ant_s11, switch


def _read_data_and_corrections(switching_state: io.SwitchingState):

    # Standards assumed at the switch
    sw = {
        "open": 1 * np.ones_like(switching_state.freq),
        "short": -1 * np.ones_like(switching_state.freq),
        "match": np.zeros_like(switching_state.freq),
    }
    # Correction at the switch
    corrections = {}
    for kind in sw:
        corrections[kind] = rc.de_embed(
            sw["open"],
            sw["short"],
            sw["match"],
            getattr(switching_state, "open").s11,
            getattr(switching_state, "short").s11,
            getattr(switching_state, "match").s11,
            getattr(switching_state, "external%s" % kind).s11,
        )[0]

    return corrections, sw


def get_switch_correction(
    ant_s11: np.ndarray,
    internal_switch: io.SwitchingState,
    f_in: np.ndarray = np.zeros([0, 1]),
    resistance_m: float = 50.166,
    poly_order: [int, Tuple] = 7,
    model_type: str = "fourier",
) -> Tuple[np.ndarray, dict]:
    """
    Compute the switch correction.

    Parameters
    ----------
    ant_s11 : array_like
        Array of S11 measurements as a function of frequency
    internal_switch : :class:`io.SwitchingState` instance
        An internal switching state object.
    f_in : np.ndarray
        The input frequencies
    resistance_m : float
        The resistance of the switch.
    poly_order : int or tuple
        Specifies the order of the fitted polynomial for the S11 measurements.
        If a tuple, must be length 3, specifying the order for the s11, s12, s22
        respectively.
    model_type : str
        The type of model to fit to the S11.

    Returns
    -------
    corr_ant_s11 : np.ndarray
        The corrected antenna S11.
    fits : dict
        Dictionary of fits to the reflection coefficients s11, s12 and s22.
    """
    corrections, sw = _read_data_and_corrections(internal_switch)

    flow = f_in.min()
    fhigh = f_in.max()
    f_center = (fhigh + flow) / 2

    # Computation of S-parameters to the receiver input
    oa, sa, la = rc.agilent_85033E(internal_switch.freq * 1e6, resistance_m, 1)

    xx, s11, s12s21, s22 = rc.de_embed(
        oa,
        sa,
        la,
        corrections["open"],
        corrections["short"],
        corrections["match"],
        corrections["open"],
    )

    # Frequency normalization
    fn = internal_switch.freq / f_center

    fn_in = f_in / f_center if len(f_in) > 10 else fn

    poly_order = (poly_order,) * 3 if not hasattr(poly_order, "__len__") else poly_order
    assert len(poly_order) == 3

    # Polynomial fits
    fits = {}
    for ikind, (kind, val, npoly) in enumerate(
        zip(["s11", "s12s21", "s22"], [s11, s12s21, s22], poly_order)
    ):
        fits[kind] = 0 + 0 * 1j
        for imag in range(2):
            mdl = ModelFit(
                model_type, xdata=fn, ydata=[np.real, np.imag][imag](val), n_terms=npoly
            )
            out = mdl.evaluate(fn_in) * (1j if imag else 1)
            fits[kind] += out

    # Corrected antenna S11
    return rc.gamma_de_embed(fits["s11"], fits["s12s21"], fits["s22"], ant_s11), fits
