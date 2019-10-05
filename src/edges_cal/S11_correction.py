# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 21:07:31 2018

@author: Nivedita
"""

from os import path

import numpy as np

from . import reflection_coefficient as rc


def high_band_switch_correction(data_path, ant_s11, sw_temp):
    par_15 = np.genfromtxt(path.join(data_path, "parameters_receiver_15degC.txt"))
    par_25 = np.genfromtxt(path.join(data_path, "parameters_receiver_25degC.txt"))
    par_35 = np.genfromtxt(path.join(data_path, "parameters_receiver_35degC.txt"))

    # frequency
    f = np.arange(50, 200.1, 0.25)

    # fit degree
    deg = 10

    # evaluating S-parameters
    s11_mag_15 = np.polyval(par_15[:, 0], f / 150)
    s11_ang_15 = np.polyval(par_15[:, 1], f / 150)
    s12s21_mag_15 = np.polyval(par_15[:, 2], f / 150)
    s12s21_ang_15 = np.polyval(par_15[:, 3], f / 150)
    s22_mag_15 = np.polyval(par_15[:, 4], f / 150)
    s22_ang_15 = np.polyval(par_15[:, 5], f / 150)

    s11_mag_25 = np.polyval(par_25[:, 0], f / 150)
    s11_ang_25 = np.polyval(par_25[:, 1], f / 150)
    s12s21_mag_25 = np.polyval(par_25[:, 2], f / 150)
    s12s21_ang_25 = np.polyval(par_25[:, 3], f / 150)
    s22_mag_25 = np.polyval(par_25[:, 4], f / 150)
    s22_ang_25 = np.polyval(par_25[:, 5], f / 150)

    s11_mag_35 = np.polyval(par_35[:, 0], f / 150)
    s11_ang_35 = np.polyval(par_35[:, 1], f / 150)
    s12s21_mag_35 = np.polyval(par_35[:, 2], f / 150)
    s12s21_ang_35 = np.polyval(par_35[:, 3], f / 150)
    s22_mag_35 = np.polyval(par_35[:, 4], f / 150)
    s22_ang_35 = np.polyval(par_35[:, 5], f / 150)

    # switch temperatures
    temp_all = np.genfromtxt(path.join(data_path, "switch_temperatures.txt"))
    temp15 = temp_all[0]
    temp25 = temp_all[1]
    temp35 = temp_all[2]

    # intermediate array
    new_s11_mag = np.zeros(len(f))
    new_s11_ang = np.zeros(len(f))
    new_s12s21_mag = np.zeros(len(f))
    new_s12s21_ang = np.zeros(len(f))
    new_s22_mag = np.zeros(len(f))
    new_s22_ang = np.zeros(len(f))

    # inter (extra) polating switch S-parameters to input temperature
    if sw_temp <= temp25:
        for i in range(len(f)):
            p = np.polyfit(
                np.array([temp15, temp25]), np.array([s11_mag_15[i], s11_mag_25[i]]), 1
            )
            new_s11_mag[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp15, temp25]), np.array([s11_ang_15[i], s11_ang_25[i]]), 1
            )
            new_s11_ang[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp15, temp25]),
                np.array([s12s21_mag_15[i], s12s21_mag_25[i]]),
                1,
            )
            new_s12s21_mag[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp15, temp25]),
                np.array([s12s21_ang_15[i], s12s21_ang_25[i]]),
                1,
            )
            new_s12s21_ang[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp15, temp25]), np.array([s22_mag_15[i], s22_mag_25[i]]), 1
            )
            new_s22_mag[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp15, temp25]), np.array([s22_ang_15[i], s22_ang_25[i]]), 1
            )
            new_s22_ang[i] = np.polyval(p, sw_temp)

    if sw_temp > temp25:
        for i in range(len(f)):
            p = np.polyfit(
                np.array([temp25, temp35]), np.array([s11_mag_25[i], s11_mag_35[i]]), 1
            )
            new_s11_mag[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp25, temp35]), np.array([s11_ang_25[i], s11_ang_35[i]]), 1
            )
            new_s11_ang[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp25, temp35]),
                np.array([s12s21_mag_25[i], s12s21_mag_35[i]]),
                1,
            )
            new_s12s21_mag[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp25, temp35]),
                np.array([s12s21_ang_25[i], s12s21_ang_35[i]]),
                1,
            )
            new_s12s21_ang[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp25, temp35]), np.array([s22_mag_25[i], s22_mag_35[i]]), 1
            )
            new_s22_mag[i] = np.polyval(p, sw_temp)

            p = np.polyfit(
                np.array([temp25, temp35]), np.array([s22_ang_25[i], s22_ang_35[i]]), 1
            )
            new_s22_ang[i] = np.polyval(p, sw_temp)

    # modeling new corrections in frequency
    p_new_s11_mag = np.polyfit(f / 150, new_s11_mag, deg)
    p_new_s11_ang = np.polyfit(f / 150, new_s11_ang, deg)
    p_new_s12s21_mag = np.polyfit(f / 150, new_s12s21_mag, deg)
    p_new_s12s21_ang = np.polyfit(f / 150, new_s12s21_ang, deg)
    p_new_s22_mag = np.polyfit(f / 150, new_s22_mag, deg)
    p_new_s22_ang = np.polyfit(f / 150, new_s22_ang, deg)

    sw_s11_mag = np.polyval(p_new_s11_mag, f / 150)
    sw_s11_ang = np.polyval(p_new_s11_ang, f / 150)
    sw_s12s21_mag = np.polyval(p_new_s12s21_mag, f / 150)
    sw_s12s21_ang = np.polyval(p_new_s12s21_ang, f / 150)
    sw_s22_mag = np.polyval(p_new_s22_mag, f / 150)
    sw_s22_ang = np.polyval(p_new_s22_ang, f / 150)

    # combine computed complex S-parameters
    sw_s11 = sw_s11_mag * (np.cos(sw_s11_ang) + 1j * np.sin(sw_s11_ang))
    sw_s12s21 = sw_s12s21_mag * (np.cos(sw_s12s21_ang) + 1j * np.sin(sw_s12s21_ang))
    sw_s22 = sw_s22_mag * (np.cos(sw_s22_ang) + 1j * np.sin(sw_s22_ang))

    # corrected antenna S11
    corr_ant_s11 = rc.gamma_de_embed(sw_s11, sw_s12s21, sw_s22, ant_s11)

    # returns corrected antenna measurement
    return corr_ant_s11, sw_s11, sw_s12s21, sw_s22


def low_band_switch_correction(root_path, ant_s11, temp_sw, f_in=np.zeros([0, 1])):
    """
    Takes the S11 of the load measured through the LNA and corrects for it.
    """
    measurements = {15: {}, 25: {}, 35: {}}

    for i, temp in enumerate([15, 25, 35]):
        data_path = path.join(
            root_path,
            "Receiver01_01_08_2018_040_to_200_MHz/{}C/S11/InternalSwitch".format(temp),
        )
        for kind in ["open", "short", "load"]:
            measurements[temp][kind] = {}
            for inp in [False, True]:
                measurements[temp][kind]["inp" if inp else "meas"], fd = rc.s1p_read(
                    path.join(data_path, "kind{}.S1P".format("_input" if inp else ""))
                )

    # # Measurements at 15degC
    # o_sw_m15, fd = rc.s1p_read(path.join(data_path_15, 'open.S1P'))
    # s_sw_m15, fd = rc.s1p_read(path.join(data_path_15 ,'short.S1P'))
    # l_sw_m15, fd = rc.s1p_read(path.join(data_path_15, 'load.S1P'))
    #
    # o_sw_in15, fd = rc.s1p_read(data_path_15 + 'open_input.S1P')
    # s_sw_in15, fd = rc.s1p_read(data_path_15 + 'short_input.S1P')
    # l_sw_in15, fd = rc.s1p_read(data_path_15 + 'load_input.S1P')
    #
    # # Measurements at 25degC
    # o_sw_m25, fd = rc.s1p_read(data_path_25 + 'open.S1P')
    # s_sw_m25, fd = rc.s1p_read(data_path_25 + 'short.S1P')
    # l_sw_m25, fd = rc.s1p_read(data_path_25 + 'load.S1P')
    #
    # o_sw_in25, fd = rc.s1p_read(data_path_25 + 'open_input.S1P')
    # s_sw_in25, fd = rc.s1p_read(data_path_25 + 'short_input.S1P')
    # l_sw_in25, fd = rc.s1p_read(data_path_25 + 'load_input.S1P')
    #
    # # Measurements at 35degC
    # o_sw_m35, fd = rc.s1p_read(data_path_35 + 'open.S1P')
    # s_sw_m35, fd = rc.s1p_read(data_path_35 + 'short.S1P')
    # l_sw_m35, fd = rc.s1p_read(data_path_35 + 'load.S1P')
    #
    # o_sw_in35, fd = rc.s1p_read(data_path_35 + 'open_input.S1P')
    # s_sw_in35, fd = rc.s1p_read(data_path_35 + 'short_input.S1P')
    # l_sw_in35, fd = rc.s1p_read(data_path_35 + 'load_input.S1P')

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(fd))
    l_sw = 0 * np.ones(len(fd))

    corrections = {15: {}, 25: {}, 35: {}}

    # Correction at the switch -- 15degC
    for temp in [15, 25, 35]:
        for kind in ["open", "short", "load"]:
            corrections[temp][kind], xx1, xx2, xx3 = rc.de_embed(
                o_sw,
                l_sw,
                measurements[temp]["open"]["meas"],
                measurements[temp]["short"]["meas"],
                measurements[temp]["load"]["meas"],
                measurements[temp]["open"]["inp"],
            )

    # Correction at the input
    resistance_male = {15: 50.13, 25: 50.12, 35: 50.11}

    s11 = {}
    s22 = {}
    s12 = {}
    for temp in [15, 25, 35]:
        oa, sa, la = rc.agilent_85033E(fd, resistance_male[temp], 1)

        xx, s11[temp], s12[temp], s22[temp] = rc.de_embed(
            oa,
            sa,
            la,
            corrections[temp]["open"],
            corrections[temp]["short"],
            corrections[temp]["load"],
            corrections[temp]["open"],
        )

    # Switch temperatures
    switch_temps = {15: 18.67, 25: 27.16, 35: 35.31}

    if temp_sw < switch_temps[25]:
        temp_low = 15
        temp_high = 25
    else:
        temp_low = 15
        temp_high = 25

    temp_array = np.array([switch_temps[temp_low], switch_temps[temp_high]])

    def get_sxx_eval(sxx, imag):
        res = np.zeros(len(fd))
        for i in range(201):
            s = np.array([sxx[temp_low][i], sxx[temp_high][i]])
            if imag:
                s = np.imag(s)
            else:
                s = np.real(s)
            p = np.polyfit(temp_array, s, 1)
            res[i] = np.polyval(p, temp_sw)
        return res

    # Final smoothing
    fdn = fd / 75e6

    if len(f_in) > 10:
        if f_in[0] > 1e5:
            fn_in = f_in / 75e6
        elif f_in < 300:
            fn_in = f_in / 75
    else:
        fn_in = fdn

    def get_smoothed_fit_part(sxx, imag):
        sxx_eval = get_sxx_eval(sxx, imag=imag)
        p = np.polyfit(fdn, sxx_eval, 7)
        return np.polyval(p, fn_in)

    def get_smoothed_fit(sxx):
        real_fit = get_smoothed_fit_part(sxx, imag=False)
        imag_fit = get_smoothed_fit_part(sxx, imag=True)
        return real_fit + 1j * imag_fit

    fit_s11 = get_smoothed_fit(s11)
    fit_s12s21 = get_smoothed_fit(s12)
    fit_s22 = get_smoothed_fit(s22)

    # Corrected antenna S11
    corr_ant_s11 = rc.gamma_de_embed(fit_s11, fit_s12s21, fit_s22, ant_s11)

    return corr_ant_s11, fit_s11, fit_s12s21, fit_s22


def _read_data_and_corrections(root_dir, branch_dir):
    path_folder = path.join(root_dir, branch_dir)
    kinds = ["Open", "Short", "Match"]

    data = {}
    for kind in kinds:
        data[kind] = {}
        for extern in [False, True]:
            data[kind]["ex" if extern else "sw"], f = rc.s1p_read(
                path.join(
                    path_folder, "{}{}01.VNA".format("External" if extern else "", kind)
                )
            )

    # o_sw_m, f = rc.s1p_read(path_folder + 'Open01.VNA')
    # s_sw_m, f = rc.s1p_read(path_folder + 'Short01.VNA')
    # l_sw_m, f = rc.s1p_read(path_folder + 'Match01.VNA')
    #
    # o_ex, f = rc.s1p_read(path_folder + 'ExternalOpen01.VNA')
    # s_ex, f = rc.s1p_read(path_folder + 'ExternalShort01.VNA')
    # l_ex, f = rc.s1p_read(path_folder + 'ExternalMatch01.VNA')

    # Standards assumed at the switch
    sw = {
        "open": 1 * np.ones_like(f),
        "short": -1 * np.ones_like(f),
        "load": np.zeros_like(f),
    }
    # Correction at the switch
    corrections = {}
    for kind in kinds:
        corrections[kind], xx1, xx2, xx3 = rc.de_embed(
            sw["open"],
            sw["short"],
            sw["load"],
            data["Open"]["sw"],
            data["Short"]["sw"],
            data["Match"]["sw"],
            data[kind]["ex"],
        )

    return data, corrections, sw, xx1, xx2, xx3, f


def low_band_switch_correction_june_2016(
    root_folder, ant_s11, f_in=np.zeros([0, 1]), resistance_m=50.166
):
    """
    Compute the low band switch correction

    Parameters
    ----------
    root_folder : str path
        Path to root folder of the data set.
    ant_s11 : array_like
        Array of S11 measurements as a function of frequency
    f_in
    resistance_m

    Returns
    -------

    """
    data, corrections, sw, xx1, xx2, xx3, f = _read_data_and_corrections(
        root_folder, "Receiver01_2018_01_08_040_to_200_MHz/25C/S11/InternalSwitch/"
    )

    flow = f_in.min()
    fhigh = f_in.max()
    f_center = (fhigh - flow) / 2 + flow

    # Computation of S-parameters to the receiver input
    resistance_of_match = resistance_m  # 50.027 #50.177#50.124#male
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

    # Polynomial fit of S-parameters from "f" to input frequency vector "f_in"
    # ------------------------------------------------------------------------

    # Frequency normalization
    fn = 1e-6 * f / f_center

    if len(f_in) > 10:
        fn_in = f_in / f_center
    else:
        fn_in = fn

    # Real-Imaginary parts
    real_s11 = np.real(s11)
    imag_s11 = np.imag(s11)
    real_s12s21 = np.real(s12s21)
    imag_s12s21 = np.imag(s12s21)
    real_s22 = np.real(s22)
    imag_s22 = np.imag(s22)

    # Polynomial fits
    p = np.polyfit(fn, real_s11, 7)
    fit_real_s11 = np.polyval(p, fn_in)

    p = np.polyfit(fn, imag_s11, 7)
    fit_imag_s11 = np.polyval(p, fn_in)

    p = np.polyfit(fn, real_s12s21, 7)
    fit_real_s12s21 = np.polyval(p, fn_in)

    p = np.polyfit(fn, imag_s12s21, 7)
    fit_imag_s12s21 = np.polyval(p, fn_in)

    p = np.polyfit(fn, real_s22, 7)
    fit_real_s22 = np.polyval(p, fn_in)

    p = np.polyfit(fn, imag_s22, 7)
    fit_imag_s22 = np.polyval(p, fn_in)

    fit_s11 = fit_real_s11 + 1j * fit_imag_s11
    fit_s12s21 = fit_real_s12s21 + 1j * fit_imag_s12s21
    fit_s22 = fit_real_s22 + 1j * fit_imag_s22

    # Corrected antenna S11
    return rc.gamma_de_embed(fit_s11, fit_s12s21, fit_s22, ant_s11)


def low_band_switch_correction_may_2019(
    root_dir, ant_s11, f_in=np.zeros([0, 1]), flow=50, fhigh=100
):
    data, corrections, sw, xx1, xx2, xx3, f = _read_data_and_corrections(
        root_dir, "Receiver02_2018_09_24_040_to_200_MHz/25C/S11/InternalSwitch01/"
    )

    # Computation of S-parameters to the receiver input
    resistance_of_match = 50.027  # 50.027 #50.124#50.225#male
    md = 1
    oa, sa, la = rc.agilent_85033E(f, resistance_of_match, md)

    xx, s11, s12s21, s22 = rc.de_embed(
        oa,
        sa,
        la,
        corrections["open"],
        corrections["short"],
        corrections["load"],
        corrections["open"],
    )

    # Polynomial fit of S-parameters from "f" to input frequency vector "f_in"
    # Frequency normalization

    fn = f / (((fhigh - flow) / 2 + flow) * 10 ** 6)

    if len(f_in) > 10:
        if f_in[0] > 1e5:
            fn_in = f_in / (((fhigh - flow) / 2 + flow) * 10 ** 6)
        elif f_in[-1] < 300:
            fn_in = f_in / (((fhigh - flow) / 2 + flow))
    else:
        fn_in = fn

    # Real-Imaginary parts
    real_s11 = np.real(s11)
    imag_s11 = np.imag(s11)
    real_s12s21 = np.real(s12s21)
    imag_s12s21 = np.imag(s12s21)
    real_s22 = np.real(s22)
    imag_s22 = np.imag(s22)

    # Polynomial fits
    p = np.polyfit(fn, real_s11, 7)
    fit_real_s11 = np.polyval(p, fn_in)

    p = np.polyfit(fn, imag_s11, 7)
    fit_imag_s11 = np.polyval(p, fn_in)

    p = np.polyfit(fn, real_s12s21, 7)
    fit_real_s12s21 = np.polyval(p, fn_in)

    p = np.polyfit(fn, imag_s12s21, 7)
    fit_imag_s12s21 = np.polyval(p, fn_in)

    p = np.polyfit(fn, real_s22, 7)
    fit_real_s22 = np.polyval(p, fn_in)

    p = np.polyfit(fn, imag_s22, 7)
    fit_imag_s22 = np.polyval(p, fn_in)

    fit_s11 = fit_real_s11 + 1j * fit_imag_s11
    fit_s12s21 = fit_real_s12s21 + 1j * fit_imag_s12s21
    fit_s22 = fit_real_s22 + 1j * fit_imag_s22

    # Corrected antenna S11
    return rc.gamma_de_embed(fit_s11, fit_s12s21, fit_s22, ant_s11)
