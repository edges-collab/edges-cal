# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 21:07:31 2018

@author: Nivedita
"""

from os import path

import numpy as np

from . import io
from . import reflection_coefficient as rc


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


def high_band_switch_correction(data_path, ant_s11, sw_temp, poly_order=10):

    # frequency
    f = np.arange(50, 200.1, 0.25)
    f_norm = f / 150

    # switch temperatures
    temp_all = np.genfromtxt(path.join(data_path, "switch_temperatures.txt"))
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
        "load": np.zeros_like(switching_state.freq),
    }
    # Correction at the switch
    corrections = {}
    for kind in sw:
        corrections[kind] = rc.de_embed(
            sw["open"],
            sw["short"],
            sw["load"],
            getattr(switching_state, "open"),
            getattr(switching_state, "short"),
            getattr(switching_state, "match"),
            getattr(switching_state, "external%s" % kind),
        )[0]

    return corrections, sw


def low_band_switch_correction(
    ant_s11, internal_switch, f_in=np.zeros([0, 1]), resistance_m=50.166, poly_order=7
):
    """
    Compute the low band switch correction

    Parameters
    ----------
    ant_s11 : array_like
        Array of S11 measurements as a function of frequency
    internal_switch : :class:`io.SwitchingState` instance
        An internal switching state object.
    f_in
    resistance_m

    Returns
    -------

    """
    corrections, sw = _read_data_and_corrections(internal_switch)

    flow = f_in.min()
    fhigh = f_in.max()
    f_center = (fhigh + flow) / 2

    f = 1e6 * internal_switch.freq
    # Computation of S-parameters to the receiver input
    resistance_of_match = resistance_m
    md = 1
    oa, sa, la = rc.agilent_85033E(f, resistance_of_match, md)

    xx, s11, s12s21, s22 = rc.de_embed(
        oa,
        sa,
        la,
        corrections["Open"],
        corrections["Short"],
        corrections["Match"],
        corrections["Open"],
    )

    # Frequency normalization
    fn = 1e-6 * f / f_center

    if len(f_in) > 10:
        fn_in = f_in / f_center
    else:
        fn_in = fn

    # Polynomial fits
    fits = {}
    for ikind, (kind, val) in enumerate(
        zip(["s11", "s12s21", "s22"], [s11, s12s21, s22])
    ):
        fits[kind] = 0
        for imag in range(2):
            p = np.polyfit(fn, [np.real, np.imag][imag](val), poly_order)
            out = np.polyval(p, fn_in) * (1j if imag else 1)
            fits[kind] += out

    # Corrected antenna S11
    return rc.gamma_de_embed(fits["s11"], fits["s12s21"], fits["s22"], ant_s11)
