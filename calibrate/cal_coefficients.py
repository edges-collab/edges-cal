# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2019
Original author: Nivedita Mahesh
Edited by: David Lewis 

"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from calibrate import reflection_coefficient as rc, S11_correction as s11, receiver_calibration_func as rcf

F_CENTER = 75.0


class spectra(object):
    def __init__(self, data_out, path, flow, fhigh, percent, runNum):
        # Initialize file paths and spectra parameters
        self.data_out = data_out
        self.path = path
        self.path_res = os.path.join(path, 'Resistance')
        self.path_spec = os.path.join(path, 'Spectra', 'mat_files')
        self.path_s11 = os.path.join(path, 'S11')
        self.flow = flow  # 50
        self.fhigh = fhigh  # 190
        self.f_center = (flow + (fhigh - flow) / 2)  # 75#
        self.percent = percent  # 5.0
        self.runNum = runNum

    cterms = None
    wterms = None
    Ambient_av_s = None
    Ambient_av_t = None
    amb_temp = None
    amb_ts = None
    Hot_av_s = None
    Hot_av_t = None
    hot_temp = None
    hot_ts = None
    Open_av_s = None
    Open_av_t = None
    open_temp = None
    open_ts = None
    Short_av_s = None
    Short_av_t = None
    short_temp = None
    short_ts = None
    AntSim3_av_s = None
    AntSim3_av_t = None
    antsim_temp = None
    antsim_ts = None
    ff = None
    ilow = None
    ihigh = None
    fe = None
    samb = None
    shot = None
    sopen = None
    sshort = None
    santsim3 = None
    fen = None
    f_s11 = None
    fit_s11_LNA_mag = None
    s11_LNA_mag = None
    fit_s11_LNA_ang = None
    s11_LAN_ang = None
    fit_s11_amb_mag = None
    s11_amb_mag = None
    fit_s11_amb_ang = None
    s11_amb_ang = None
    fit_s11_hot_mag = None
    s11_hot_mag = None
    fit_s11_hot_ang = None
    s11_hot_ang = None
    fit_s11_shorted_mag = None
    s11_shorted_mag = None
    fit_s11_shorted_ang = None
    s11_shorted_ang = None
    fit_s11_shorted_mag = None
    s11_shorted_mag = None
    fit_s11_shorted_ang = None
    s11_shorted_ang = None
    fit_s11_open_mag = None
    s11_open_mag = None
    fit_s11_open_ang = None
    s11_open_ang = None
    fit_s11_antsim3_mag = None
    s11_antsim3_mag = None
    fit_s11_antsim3_ang = None
    s11_antsim3_ang = None
    amb_temp = None
    hot_ts = None
    open_temp = None
    short_temp = None
    antsim_temp = None
    Ant_sim_calibrated = None
    Ambient_calibrated = None
    Hot_calibrated = None
    Open_calibrated = None
    Short_calibrated = None
    scale = None
    off = None
    Tu = None
    TC = None
    TS = None
    Thd = None
    model_s11_LNA_mag = None
    model_s11_LNA_ang = None
    rl = None
    ra = None
    rh = None
    ro = None
    rs = None
    ras = None

    def construct_average(self, kind, percent=5.0):
        spec_files = glob.glob(os.path.join(self.path_spec, kind + '*.mat'))
        res_files = glob.glob(os.path.join(self.path_res, kind + '*.txt'))

        setattr(
            self, kind.lower() + "_ave",
            rcf.AverageCal(spec_files, res_files, percent)
        )

    def save(self):
        for kind in ['samb', 'shot', 'shopen', 'sshort', 'santsim4']:
            np.savetxt(os.path.join(self.data_out, kind), getattr(self, kind))


def explog(x, a, b, c, d, e, f_center=F_CENTER):
    return a * (x / f_center) ** (
            b + c * np.log(x / f_center) + d * np.log(x / f_center) ** 2 + e * np.log(x / f_center) ** 3) + 2.725


def physical5(x, a, b, c, d, e, f_center=F_CENTER):
    return np.log(
        a * (x / f_center) ** (-2.5 + b + c * np.log(x / f_center)) * np.exp(-d * (x / f_center) ** -2) + e * (
                x / f_center) ** -2)  # + 2.725


def physicallin5(x, a, b, c, d, e, f_center=F_CENTER):
    return a * (x / f_center) ** -2.5 + b * (x / f_center) ** -2.5 * np.log(x / f_center) + c * (
            x / f_center) ** -2.5 * (np.log(x / f_center)) ** 2 + d * (x / f_center) ** -4.5 + e * (
                   x / f_center) ** -2


def loglog(x, a, b, c, d, e, f_center=F_CENTER):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


def linlog(x, a, b, c, d, e, f_center=F_CENTER):
    return (x / f_center) ** b * (a + b * np.log(x / f_center) + c * np.log(x / f_center) ** 2 + d * np.log(
        x / f_center) ** 3 + e * np.log(x / f_center) ** 4)


def spec_read(s, percent=5.0):
    # Ambient Load
    s.construct_average("Ambient", percent)
    s.construct_average("HotLoad", 4 * percent)
    s.construct_average("LongCableOpen", percent)
    s.construct_average("LongCableShort", percent)
    s.construct_average("AntSim4", percent)

    s.ff, s.ilow, s.ihigh = rcf.frequency_edges(s.flow, s.fhigh)
    s.fe = s.ff[s.ilow:s.ihigh + 1]
    s.samb = s.ambient_ave.temp_ave[s.ilow:s.ihigh + 1]
    s.shot = s.hotload_ave.temp_ave[s.ilow:s.ihigh + 1]
    s.sopen = s.longcableopen_ave.temp_ave[s.ilow:s.ihigh + 1]
    s.sshort = s.longcableshort_ave.temp_ave[s.ilow:s.ihigh + 1]
    s.santsim3 = s.antsim4_ave.temp_ave[s.ilow:s.ihigh + 1]

    # Spectra modeling
    s.fen = (s.fe - (s.flow + (s.fhigh - s.flow) / 2)) / ((s.fhigh - s.flow) / 2)


def spec_plot(s):
    print('Saving and plotting')
    s.save()

    plt.figure(1, figsize=(12, 8), facecolor='white')
    plt.subplot(4, 1, 1)
    plt.plot(s.fe, s.samb)
    plt.grid()
    plt.ylabel("Ambient (K)")

    plt.subplot(4, 1, 2)
    plt.plot(s.fe, s.shot)
    plt.grid()
    plt.ylabel("Hot  (K)")

    plt.subplot(4, 1, 3)
    plt.plot(s.fe, s.sopen)
    plt.grid()
    plt.ylabel("CableOpen  (K)")

    plt.subplot(4, 1, 4)
    plt.plot(s.fe, s.sshort)
    plt.grid()
    plt.ylabel("CableShort  (K)")
    plt.xlabel('freq(MHz)')
    plt.savefig('Temps', dpi=plt.figure(1).dpi)

    plt.figure(2, facecolor='white', figsize=(12, 2))
    plt.plot(s.fe, s.santsim3)
    plt.grid()
    plt.ylabel("Antsim Spectra (K)")
    plt.xlabel('freq(MHz)')

    plt.savefig('Antsim', dpi=plt.figure(1).dpi)


def s11_model(spec, resistance_f=50.009, resistance_m=50.166):
    print('S11')

    path_LNA = spec.path_s11 + 'ReceiverReading01/'
    path_ambient = spec.path_s11 + 'Ambient/'
    path_hot = spec.path_s11 + 'HotLoad/'
    path_open = spec.path_s11 + 'LongCableOpen/'
    path_shorted = spec.path_s11 + 'LongCableShorted/'
    path_AntSim3 = spec.path_s11 + 'AntSim4/'

    # Receiver reflection coefficient
    print('Reflection Coefficient')

    # Reading measurements
    o, fr0 = rc.s1p_read(path_LNA + 'Open0' + str(spec.runNum) + '.s1p')
    s, fr0 = rc.s1p_read(path_LNA + 'Short0' + str(spec.runNum) + '.s1p')
    l, fr0 = rc.s1p_read(path_LNA + 'Match0' + str(spec.runNum) + '.s1p')
    LNA0, fr0 = rc.s1p_read(path_LNA + 'ReceiverReading0' + str(spec.runNum) + '.s1p')

    # Models of standards
    resistance_of_match = resistance_f
    md = 1
    oa, sa, la = rc.agilent_85033E(fr0, resistance_of_match, md)

    # Correction of measurements
    LNAc, x1, x2, x3 = rc.de_embed(oa, sa, la, o, s, l, LNA0)

    LNA = LNAc[(fr0 / 1e6 >= spec.flow) & (fr0 / 1e6 <= spec.fhigh)]
    fr = fr0[(fr0 / 1e6 >= spec.flow) & (fr0 / 1e6 <= spec.fhigh)]

    # Ambient load before
    print('Ambient Load before')
    o_m, f_a1 = rc.s1p_read(path_ambient + 'Open0' + str(spec.runNum) + '.s1p')
    s_m, f_a1 = rc.s1p_read(path_ambient + 'Short0' + str(spec.runNum) + '.s1p')
    l_m, f_a1 = rc.s1p_read(path_ambient + 'Match0' + str(spec.runNum) + '.s1p')
    a1_m, f_a1 = rc.s1p_read(path_ambient + 'External0' + str(spec.runNum) + '.s1p')

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f_a1))
    s_sw = -1 * np.ones(len(f_a1))
    l_sw = 0 * np.ones(len(f_a1))

    # Correction at switch
    a1_sw_c, x1, x2, x3 = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, a1_m)

    # Correction at receiver input
    a1_c = s11.low_band_switch_correction_june_2016(
        '/data5/edges/data/', a1_sw_c, f_in=f_a1, flow=spec.flow, fhigh=spec.fhigh, resistance_m=resistance_m
    )

    a1 = a1_c[(f_a1 / 1e6 >= spec.flow) & (f_a1 / 1e6 <= spec.fhigh)]

    # Hot load before
    print('Hot load before')
    o_m, f_h1 = rc.s1p_read(path_hot + 'Open0' + str(spec.runNum) + '.s1p')
    s_m, f_h1 = rc.s1p_read(path_hot + 'Short0' + str(spec.runNum) + '.s1p')
    l_m, f_h1 = rc.s1p_read(path_hot + 'Match0' + str(spec.runNum) + '.s1p')
    h1_m, f_h1 = rc.s1p_read(path_hot + 'External0' + str(spec.runNum) + '.s1p')

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f_h1))
    s_sw = -1 * np.ones(len(f_h1))
    l_sw = 0 * np.ones(len(f_h1))

    # Correction at switch
    h1_sw_c, x1, x2, x3 = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, h1_m)

    # Correction at receiver input
    h1_c = s11.low_band_switch_correction_june_2016(
        '/data5/edges/data/', h1_sw_c, f_in=f_h1, flow=spec.flow, fhigh=spec.fhigh, resistance_m=resistance_m
    )

    h1 = h1_c[(f_h1 / 1e6 >= spec.flow) & (f_h1 / 1e6 <= spec.fhigh)]

    # Open Cable before
    print('Open Cable before')
    o_m, f_o1 = rc.s1p_read(path_open + 'Open0' + str(spec.runNum) + '.s1p')
    s_m, f_o1 = rc.s1p_read(path_open + 'Short0' + str(spec.runNum) + '.s1p')
    l_m, f_o1 = rc.s1p_read(path_open + 'Match0' + str(spec.runNum) + '.s1p')
    o1_m, f_o1 = rc.s1p_read(path_open + 'External0' + str(spec.runNum) + '.s1p')

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f_o1))
    s_sw = -1 * np.ones(len(f_o1))
    l_sw = 0 * np.ones(len(f_o1))

    # Correction at switch
    o1_sw_c, x1, x2, x3 = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, o1_m)

    # Correction at receiver input
    o1_c = s11.low_band_switch_correction_june_2016(
        '/data5/edges/data/', o1_sw_c, f_in=f_o1, flow=spec.flow, fhigh=spec.fhigh, resistance_m=resistance_m
    )

    o1 = o1_c[(f_o1 / 1e6 >= spec.flow) & (f_o1 / 1e6 <= spec.fhigh)]

    # Short Cable before
    print('Short cable before')
    o_m, f_s1 = rc.s1p_read(path_shorted + 'Open0' + str(spec.runNum) + '.s1p')
    s_m, f_s1 = rc.s1p_read(path_shorted + 'Short0' + str(spec.runNum) + '.s1p')
    l_m, f_s1 = rc.s1p_read(path_shorted + 'Match0' + str(spec.runNum) + '.s1p')
    s1_m, f_s1 = rc.s1p_read(path_shorted + 'External0' + str(spec.runNum) + '.s1p')

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f_s1))
    s_sw = -1 * np.ones(len(f_s1))
    l_sw = 0 * np.ones(len(f_s1))

    # Correction at switch
    s1_sw_c, x1, x2, x3 = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, s1_m)

    # Correction at receiver input
    s1_c = s11.low_band_switch_correction_june_2016(
        '/data5/edges/data/', s1_sw_c, f_in=f_s1, flow=spec.flow, fhigh=spec.fhigh, resistance_m=resistance_m
    )

    s1 = s1_c[(f_s1 / 1e6 >= spec.flow) & (f_s1 / 1e6 <= spec.fhigh)]

    # AntSim3 before
    print('AntSim3 before')
    o_m, f_as1 = rc.s1p_read(path_AntSim3 + 'Open0' + str(spec.runNum) + '.s1p')
    s_m, f_as1 = rc.s1p_read(path_AntSim3 + 'Short0' + str(spec.runNum) + '.s1p')
    l_m, f_as1 = rc.s1p_read(path_AntSim3 + 'Match0' + str(spec.runNum) + '.s1p')
    as1_m, f_as1 = rc.s1p_read(path_AntSim3 + 'External0' + str(spec.runNum) + '.s1p')

    # Standards assumed at the switch
    o_sw = 1 * np.ones(len(f_s1))
    s_sw = -1 * np.ones(len(f_s1))
    l_sw = 0 * np.ones(len(f_s1))

    # Correction at switch
    as1_sw_c, x1, x2, x3 = rc.de_embed(o_sw, s_sw, l_sw, o_m, s_m, l_m, as1_m)

    # Correction at receiver input
    as1_c = s11.low_band_switch_correction_june_2016(
        '/data5/edges/data/', as1_sw_c, f_in=f_s1, flow=spec.flow, fhigh=spec.fhigh, resistance_m=resistance_m
    )

    as1 = as1_c[(f_as1 / 1e6 >= spec.flow) & (f_as1 / 1e6 <= spec.fhigh)]

    spec.f_s11 = fr  # s11_all[:,0]*1e6#
    s11_LNA = LNA  # s11_all[:,1] + 1j*s11_all[:,2]#
    s11_amb = a1  # s11_all[:,3] + 1j*s11_all[:,4]#
    s11_hot = h1  # s11_all[:,5] + 1j*s11_all[:,6]#
    s11_open = o1  # s11_all[:,7] + 1j*s11_all[:,8]#
    s11_shorted = s1  # s11_all[:,9] + 1j*s11_all[:,10]#
    s11_antsim3 = as1  # s11_all[:,11] + 1j*s11_all[:,12]#

    # Magnitude / Angle
    # LNA
    spec.s11_LNA_mag = np.abs(s11_LNA)
    spec.s11_LNA_ang = np.unwrap(np.angle(s11_LNA))

    # Ambient
    spec.s11_amb_mag = np.abs(s11_amb)
    spec.s11_amb_ang = np.unwrap(np.angle(s11_amb))

    # Hot
    spec.s11_hot_mag = np.abs(s11_hot)
    spec.s11_hot_ang = np.unwrap(np.angle(s11_hot))

    # Open
    spec.s11_open_mag = np.abs(s11_open)
    spec.s11_open_ang = np.unwrap(np.angle(s11_open))

    # Shorted
    spec.s11_shorted_mag = np.abs(s11_shorted)
    spec.s11_shorted_ang = np.unwrap(np.angle(s11_shorted))

    # Shorted
    spec.s11_antsim3_mag = np.abs(s11_antsim3)
    spec.s11_antsim3_ang = np.unwrap(np.angle(s11_antsim3))

    # Modeling S11
    print('Modeling S11')
    f_s11n = (spec.f_s11 / 1e6 - ((spec.fhigh - spec.flow) / 2 + spec.flow)) / ((spec.fhigh - spec.flow) / 2)
    spec.fit_s11_LNA_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_LNA_mag, 37)
    spec.fit_s11_LNA_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_LNA_ang, 37)

    spec.fit_s11_amb_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_amb_mag, 37)
    spec.fit_s11_amb_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_amb_ang, 37)

    spec.fit_s11_hot_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_hot_mag, 37)
    spec.fit_s11_hot_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_hot_ang, 37)

    spec.fit_s11_open_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_open_mag, 105)
    spec.fit_s11_open_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_open_ang, 105)

    spec.fit_s11_shorted_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_shorted_mag, 105)
    spec.fit_s11_shorted_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_shorted_ang, 105)

    spec.fit_s11_antsim3_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_antsim3_mag, 55)
    spec.fit_s11_antsim3_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_antsim3_ang, 55)

    s11_data = np.genfromtxt(spec.path_s11 + 'semi_rigid_s_parameters_WITH_HEADER.txt')

    for i in range(len(s11_data[:, 0])):
        if (s11_data[i, 0] <= spec.flow) and (s11_data[i + 1, 0] > spec.flow):
            index_low = i
        if (s11_data[i - 1, 0] < spec.fhigh) and (s11_data[i, 0] >= spec.fhigh):
            index_high = i
    index_s11 = np.arange(index_low, index_high + 1)

    f_s11n_r = (s11_data[index_s11, 0] - ((spec.fhigh - spec.flow) / 2 + spec.flow)) / ((spec.fhigh - spec.flow) / 2)
    s11_sr = s11_data[index_s11, 1] + 1j * s11_data[index_s11, 2]
    s12s21_sr = s11_data[index_s11, 3] + 1j * s11_data[index_s11, 4]
    s22_sr = s11_data[index_s11, 5] + 1j * s11_data[index_s11, 6]

    # sr-s11
    s11_sr_mag = np.abs(s11_sr)
    s11_sr_ang = np.unwrap(np.angle(s11_sr))

    # sr-s12s21
    s12s21_sr_mag = np.abs(s12s21_sr)
    s12s21_sr_ang = np.unwrap(np.angle(s12s21_sr))

    # sr-s22
    s22_sr_mag = np.abs(s22_sr)
    s22_sr_ang = np.unwrap(np.angle(s22_sr))

    fit_s11_sr_mag = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, s11_sr_mag, 21)
    fit_s11_sr_ang = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, s11_sr_ang, 21)

    fit_s12s21_sr_mag = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, s12s21_sr_mag, 21)
    fit_s12s21_sr_ang = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, s12s21_sr_ang, 21)

    fit_s22_sr_mag = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, s22_sr_mag, 21)
    fit_s22_sr_ang = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, s22_sr_ang, 21)

    model_s11_LNA_mag = rcf.model_evaluate('fourier', spec.fit_s11_LNA_mag[0], spec.fen)
    model_s11_LNA_ang = rcf.model_evaluate('fourier', spec.fit_s11_LNA_ang[0], spec.fen)
    spec.model_s11_LNA_mag = model_s11_LNA_mag
    spec.model_s11_LNA_ang = model_s11_LNA_ang

    model_s11_ambient_mag = rcf.model_evaluate('fourier', spec.fit_s11_amb_mag[0], spec.fen)
    model_s11_ambient_ang = rcf.model_evaluate('fourier', spec.fit_s11_amb_ang[0], spec.fen)

    model_s11_hot_mag = rcf.model_evaluate('fourier', spec.fit_s11_hot_mag[0], spec.fen)
    model_s11_hot_ang = rcf.model_evaluate('fourier', spec.fit_s11_hot_ang[0], spec.fen)

    model_s11_open_mag = rcf.model_evaluate('fourier', spec.fit_s11_open_mag[0], spec.fen)
    model_s11_open_ang = rcf.model_evaluate('fourier', spec.fit_s11_open_ang[0], spec.fen)

    model_s11_shorted_mag = rcf.model_evaluate('fourier', spec.fit_s11_shorted_mag[0], spec.fen)
    model_s11_shorted_ang = rcf.model_evaluate('fourier', spec.fit_s11_shorted_ang[0], spec.fen)

    model_s11_antsim3_mag = rcf.model_evaluate('fourier', spec.fit_s11_antsim3_mag[0], spec.fen)
    model_s11_antsim3_ang = rcf.model_evaluate('fourier', spec.fit_s11_antsim3_ang[0], spec.fen)

    model_s11_sr_mag = rcf.model_evaluate('polynomial', fit_s11_sr_mag[0], spec.fen)
    model_s11_sr_ang = rcf.model_evaluate('polynomial', fit_s11_sr_ang[0], spec.fen)

    model_s12s21_sr_mag = rcf.model_evaluate('polynomial', fit_s12s21_sr_mag[0], spec.fen)
    model_s12s21_sr_ang = rcf.model_evaluate('polynomial', fit_s12s21_sr_ang[0], spec.fen)

    model_s22_sr_mag = rcf.model_evaluate('polynomial', fit_s22_sr_mag[0], spec.fen)
    model_s22_sr_ang = rcf.model_evaluate('polynomial', fit_s22_sr_ang[0], spec.fen)

    # converting back to real/imaginary

    rl = model_s11_LNA_mag * np.cos(model_s11_LNA_ang) + 1j * model_s11_LNA_mag * np.sin(model_s11_LNA_ang)
    spec.rl = rl
    ra = model_s11_ambient_mag * np.cos(model_s11_ambient_ang) + 1j * model_s11_ambient_mag * np.sin(
        model_s11_ambient_ang)
    spec.ra = ra
    rh = model_s11_hot_mag * np.cos(model_s11_hot_ang) + 1j * model_s11_hot_mag * np.sin(model_s11_hot_ang)
    spec.rh = rh
    ro = model_s11_open_mag * np.cos(model_s11_open_ang) + 1j * model_s11_open_mag * np.sin(model_s11_open_ang)
    spec.ro = ro
    rs = model_s11_shorted_mag * np.cos(model_s11_shorted_ang) + 1j * model_s11_shorted_mag * np.sin(
        model_s11_shorted_ang)
    spec.rs = rs
    ras = model_s11_antsim3_mag * np.cos(model_s11_antsim3_ang) + 1j * model_s11_antsim3_mag * np.sin(
        model_s11_antsim3_ang)
    spec.ras = ras

    s11_sr = model_s11_sr_mag * np.cos(model_s11_sr_ang) + 1j * model_s11_sr_mag * np.sin(model_s11_sr_ang)
    s12s21_sr = model_s12s21_sr_mag * np.cos(model_s12s21_sr_ang) + 1j * model_s12s21_sr_mag * np.sin(
        model_s12s21_sr_ang)
    s22_sr = model_s22_sr_mag * np.cos(model_s22_sr_ang) + 1j * model_s22_sr_mag * np.sin(model_s22_sr_ang)

    rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, rh)

    # inverting the direction of the s-parameters,
    # since the port labels have to be inverted to match those of Pozar eqn 10.25
    s11_sr_rev = s22_sr

    # absolute value of S_21
    abs_s21 = np.sqrt(np.abs(s12s21_sr))

    # available power gain
    G = (abs_s21 ** 2) * (1 - np.abs(rht) ** 2) / ((np.abs(1 - s11_sr_rev * rht)) ** 2 * (1 - (np.abs(rh)) ** 2))

    # temperature
    spec.Thd = G * spec.Hot_av_t + (1 - G) * spec.Ambient_av_t


def s11_cal(spec, cterms, wterms):
    print('Calibration coefficients')
    spec.cterms = cterms
    spec.wterms = wterms

    # ==================================================================================
    # This block originally in function s11_model(). It was moved to keep cterm/wterms
    # defined here only.
    temp = np.array([spec.fe, spec.model_s11_LNA_mag, spec.model_s11_LNA_ang])
    output_file = temp.T
    output_file_str = spec.data_out + 'LNA_' + str(spec.flow) + '_' + str(spec.fhigh) + '_25C_' + str(
        spec.cterms) + '-' + str(spec.wterms) + '_fe-350-load.txt'
    np.savetxt(output_file_str, output_file)
    # ==================================================================================

    spec.scale, spec.off, spec.Tu, spec.TC, spec.TS = rcf.calibration_quantities(
        spec.flow, spec.fhigh, spec.fe, spec.samb, spec.shot, spec.sopen, spec.sshort,
        spec.rl, spec.ra, spec.rh, spec.ro, spec.rs, spec.Ambient_av_t, spec.Thd,
        spec.Open_av_t, spec.Short_av_t, spec.cterms, spec.wterms
    )
    spec.Ant_sim_calibrated = rcf.calibrated_antenna_temperature(
        spec.santsim3, spec.ras, spec.rl, spec.scale, spec.off,
        spec.Tu, spec.TC, spec.TS, Tamb_internal=300
    )
    spec.Ambient_calibrated = rcf.calibrated_antenna_temperature(
        spec.samb, spec.ra, spec.rl, spec.scale, spec.off, spec.Tu,
        spec.TC, spec.TS, Tamb_internal=300
    )
    spec.Hot_calibrated = rcf.calibrated_antenna_temperature(
        spec.shot, spec.rh, spec.rl, spec.scale, spec.off, spec.Tu,
        spec.TC, spec.TS, Tamb_internal=300
    )
    spec.Open_calibrated = rcf.calibrated_antenna_temperature(
        spec.sopen, spec.ro, spec.rl, spec.scale, spec.off, spec.Tu,
        spec.TC, spec.TS, Tamb_internal=300
    )
    spec.Short_calibrated = rcf.calibrated_antenna_temperature(
        spec.sshort, spec.rs, spec.rl, spec.scale, spec.off, spec.Tu,
        spec.TC, spec.TS, Tamb_internal=300
    )


def s11_plot(s):
    print('S11 plots')

    plt.figure(8, facecolor='w')
    plt.subplot(4, 1, 1)
    plt.plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_LNA_mag[1]))
    plt.ylabel('S11(mag)')
    plt.title('LNA')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(s.f_s11 / 1e6, (s.fit_s11_LNA_mag[1] - s.s11_LNA_mag), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(mag)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(s.f_s11 / 1e6, s.fit_s11_LNA_ang[1] * 180 / np.pi)
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(' S11(Ang)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(s.f_s11 / 1e6, (s.s11_LNA_ang - s.fit_s11_LNA_ang[1]), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(Ang)')
    # plt.legend()
    plt.grid()

    plt.figure(9, facecolor='w')
    plt.subplot(4, 1, 1)
    plt.plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_amb_mag[1]))
    plt.ylabel('S11(mag)')
    plt.title('Ambient Load')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(s.f_s11 / 1e6, (s.fit_s11_amb_mag[1] - s.s11_amb_mag), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(mag)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(s.f_s11 / 1e6, s.fit_s11_amb_ang[1] * 180 / np.pi)
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(' S11(Ang)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(s.f_s11 / 1e6, (s.s11_amb_ang - s.fit_s11_amb_ang[1]), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(Ang)')
    # plt.legend()
    plt.grid()

    plt.figure(10, facecolor='w')
    plt.subplot(4, 1, 1)
    plt.plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_hot_mag[1]))
    plt.ylabel('S11(mag)')
    plt.title('Hot Load')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(s.f_s11 / 1e6, (s.fit_s11_hot_mag[1] - s.s11_hot_mag), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(mag)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(s.f_s11 / 1e6, s.fit_s11_hot_ang[1] * 180 / np.pi)
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(' S11(Ang)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(s.f_s11 / 1e6, (s.s11_hot_ang - s.fit_s11_hot_ang[1]), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(Ang)')
    # plt.legend()
    plt.grid()

    plt.figure(11, facecolor='w')
    plt.subplot(4, 1, 1)
    plt.plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_shorted_mag[1]))
    plt.ylabel('S11(mag)')
    plt.title('Long Cable Shorted')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(s.f_s11 / 1e6, (s.fit_s11_shorted_mag[1] - s.s11_shorted_mag), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(mag)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(s.f_s11 / 1e6, s.fit_s11_shorted_ang[1] * 180 / np.pi)
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(' S11(Ang)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(s.f_s11 / 1e6, (s.s11_shorted_ang - s.fit_s11_shorted_ang[1]), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(Ang)')
    # plt.legend()
    plt.grid()

    plt.figure(12, facecolor='w')
    plt.subplot(4, 1, 1)
    plt.plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_open_mag[1]))
    plt.ylabel('S11(mag)')
    plt.title('Long Cable - Open')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(s.f_s11 / 1e6, (s.fit_s11_open_mag[1] - s.s11_open_mag), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(mag)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(s.f_s11 / 1e6, s.fit_s11_open_ang[1] * 180 / np.pi)
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(' S11(Ang)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(s.f_s11 / 1e6, (s.s11_open_ang - s.fit_s11_open_ang[1]), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(Ang)')
    # plt.legend()
    plt.grid()

    plt.figure(13, facecolor='w')
    plt.subplot(4, 1, 1)
    plt.plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_antsim3_mag[1]))
    plt.ylabel('S11(mag)')
    plt.title('Antsim4')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(s.f_s11 / 1e6, (s.fit_s11_antsim3_mag[1] - s.s11_antsim3_mag), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(mag)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(s.f_s11 / 1e6, s.fit_s11_antsim3_ang[1] * 180 / np.pi)
    plt.xlabel('Frequency(MHz)')
    plt.ylabel(' S11(Ang)')
    plt.xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(s.f_s11 / 1e6, (s.s11_antsim3_ang - s.fit_s11_antsim3_ang[1]), 'g')
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Delta S11(Ang)')
    # plt.legend()
    plt.grid()

    lim = len(s.fe)  # np.where(fe==100)[0][0]

    # turn temp into variable
    np.savetxt(s.data_out + 'All_cal-params_' + str(s.flow) + '_' + str(s.fhigh) + '_' + str(s.cterms) + '-' + str(
        s.wterms) + '_25C_s11alan.txt', (s.fe, s.scale, s.off, s.Tu, s.TC, s.TS))

    # binning
    bins = 64  # 128
    fact = len(s.Ant_sim_calibrated[:lim]) / bins
    fnew = np.linspace(s.flow, s.fhigh, bins)
    Ant_sim_cal = np.zeros(bins)

    Ambient_cal = np.zeros(bins)
    Hot_cal = np.zeros(bins)
    Open_cal = np.zeros(bins)
    Short_cal = np.zeros(bins)

    for i in range(bins):
        Ant_sim_cal[i] = np.average(s.Ant_sim_calibrated[int(i * fact):int((i + 1) * fact)])
        Ambient_cal[i] = np.average(s.Ambient_calibrated[int(i * fact):int((i + 1) * fact)])
        Hot_cal[i] = np.average(s.Hot_calibrated[int(i * fact):int((i + 1) * fact)])
        Open_cal[i] = np.average(s.Open_calibrated[int(i * fact):int((i + 1) * fact)])
        Short_cal[i] = np.average(s.Short_calibrated[int(i * fact):int((i + 1) * fact)])

    stop = bins
    antsim_rms = np.sqrt(np.mean((Ant_sim_cal[:stop] - np.mean(Ant_sim_cal[:stop])) ** 2))

    ambient_rms = np.sqrt(np.mean((Ambient_cal[:stop] - np.mean(Ambient_cal[:stop])) ** 2))
    hot_rms = np.sqrt(np.mean((Hot_cal[:stop] - np.mean(Hot_cal[:stop])) ** 2))
    open_rms = np.sqrt(np.mean((Open_cal[:stop] - np.mean(Open_cal[:stop])) ** 2))
    short_rms = np.sqrt(np.mean((Short_cal[:stop] - np.mean(Short_cal[:stop])) ** 2))

    plt.figure(14, facecolor='w')
    plt.plot(fnew[:stop], Ant_sim_cal[:stop], 'r', label='Calibrated')
    plt.text(65, np.max(Ant_sim_cal), 'RMS=' + str(np.round(antsim_rms, 3)) + 'K')
    plt.ylim([np.min(Ant_sim_cal), np.max(Ant_sim_cal)])
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Ant_sim3')
    # plt.ylim([296.22,296.41])
    plt.title('Rcv02 -2018')
    plt.ticklabel_format(useOffset=False)
    plt.grid()

    plt.figure(15, facecolor='w')
    plt.plot(fnew[:stop], Ambient_cal[:stop])
    plt.axhline(s.Ambient_av_t, color='r')
    plt.text(70, np.max(Ambient_cal), 'RMS=' + str(np.round(ambient_rms, 3)) + 'K')
    plt.ylim([np.min(Ambient_cal), np.max(Ambient_cal)])

    # plt.xlim([flow,fhigh])
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Ambient[K]')
    plt.ticklabel_format(useOffset=False)
    plt.grid()

    plt.figure(16, facecolor='w')
    plt.plot(fnew[:stop], Hot_cal[:stop])
    # plt.axhline(Hot_av_t,color='r' )
    plt.plot(s.fe, s.Thd, color='r')
    plt.text(70, np.max(Hot_cal), 'RMS=' + str(np.round(hot_rms, 3)) + 'K')
    plt.ylim([np.min(Hot_cal), np.max(Hot_cal)])
    # plt.ylim([400,400.5])
    # plt.xlim([flow,fhigh])
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Hot Load[K]')
    plt.ticklabel_format(useOffset=False)
    plt.grid()

    plt.figure(17, facecolor='w')
    plt.plot(fnew[:stop], Open_cal[:stop])
    plt.axhline(s.Open_av_t, color='r')
    plt.text(70, np.max(Open_cal), 'RMS=' + str(np.round(open_rms, 3)) + 'K')
    plt.ylim([np.min(Open_cal), np.max(Open_cal)])
    # plt.ylim([296,300])
    # plt.xlim([flow,fhigh])
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Long Cable Open[K]')
    plt.ticklabel_format(useOffset=False)
    plt.grid()

    plt.figure(18, facecolor='w')
    plt.plot(fnew[:stop], Short_cal[:stop])
    plt.axhline(s.Short_av_t, color='r')
    plt.text(70, np.max(Short_cal), 'RMS=' + str(np.round(short_rms, 3)) + 'K')
    plt.ylim([np.min(Short_cal), np.max(Short_cal)])
    # plt.xlim([flow,fhigh])
    plt.xlabel('Frequency(MHz)')
    plt.ylabel('Long Cable SHort[K]')
    plt.ticklabel_format(useOffset=False)
    plt.grid()

    plt.figure(3, facecolor='w')
    # plt.subplot(5,1,1)
    plt.plot(s.fe, s.scale)
    plt.xlabel('frequency(MHz)')
    plt.ylabel('Scale(C1)')
    plt.grid()
    plt.ticklabel_format(useOffset=False)

    plt.figure(4, facecolor='w')
    # plt.subplot(5,2,1)
    plt.plot(s.fe, s.off)
    plt.ylabel('Offset(C2[K])')
    plt.xlabel('frequency(MHz)')
    plt.grid()
    plt.ticklabel_format(useOffset=False)

    plt.figure(5, facecolor='w')
    # plt.subplot(5,1,2)
    plt.plot(s.fe, s.Tu)
    plt.ylabel('TU(K)')
    plt.xlabel('frequency(MHz)')
    plt.grid()
    plt.ticklabel_format(useOffset=False)

    plt.figure(6, facecolor='w')
    # plt.subplot(5,2,2)
    plt.plot(s.fe, s.TC)
    plt.ylabel('TC[K]')
    plt.xlabel('frequency(MHz)')
    plt.grid()
    plt.ticklabel_format(useOffset=False)

    plt.figure(7, facecolor='w')
    # plt.subplot(5,1,3)
    plt.plot(s.fe, s.TS)
    plt.ylabel('TS[K]')
    plt.xlabel('frequency(MHz)')
    plt.grid()
    plt.ticklabel_format(useOffset=False)

    np.savetxt(s.data_out + 'fit_s11_LNA_mag.txt', s.fit_s11_LNA_mag[1])
    np.savetxt(s.data_out + 'fit_s11_LNA_ang.txt', s.fit_s11_LNA_ang[1])
    np.savetxt(s.data_out + 'fit_s11_amb_mag.txt', s.fit_s11_amb_mag[1])
    np.savetxt(s.data_out + 'fit_s11_amb_ang.txt', s.fit_s11_amb_ang[1])
    np.savetxt(s.data_out + 'fit_s11_hot_mag.txt', s.fit_s11_hot_mag[1])
    np.savetxt(s.data_out + 'fit_s11_hot_ang.txt', s.fit_s11_hot_ang[1])
    np.savetxt(s.data_out + 'model_s11_shorted_mag.txt', s.fit_s11_shorted_mag[0])
    np.savetxt(s.data_out + 'model_s11_shorted_ang.txt', s.fit_s11_shorted_ang[0])
    np.savetxt(s.data_out + 'model_s11_open_mag.txt', s.fit_s11_open_mag[0])
    np.savetxt(s.data_out + 'model_s11_open_ang.txt', s.fit_s11_open_ang[0])

    plt.figure(19, figsize=(8, 8), facecolor='w')
    plt.subplot(5, 1, 1)
    plt.plot(s.amb_temp[0::120] - 273)
    plt.ylabel('Amb temp [$^o$ C]')
    plt.grid()

    plt.subplot(5, 1, 2)
    plt.plot(s.hot_ts[0::120] - 273)
    plt.ylabel('hot temp [$^o$ C]')
    plt.grid()

    plt.subplot(5, 1, 3)
    plt.plot(s.open_temp[0::120] - 273)
    plt.ylabel('open temp [$^o$ C]')
    plt.grid()

    plt.subplot(5, 1, 4)
    plt.plot(s.short_temp[0::120] - 273)
    plt.ylabel('Short temp [$^o$ C]')
    plt.grid()

    plt.subplot(5, 1, 5)
    plt.plot(s.antsim_temp[0::120] - 273)
    plt.xlabel('Time(Hr)')
    plt.ylabel('Sim2 temp [$^o$ C]')
    plt.grid()
    plt.subplots_adjust(hspace=0.5)

    np.savetxt(s.data_out + 'amb_temp.txt', s.amb_temp)
    np.savetxt(s.data_out + 'hot_temp.txt', s.hot_ts)
    np.savetxt(s.data_out + 'open_temp.txt', s.open_temp)
    np.savetxt(s.data_out + 'short_temp.txt', s.short_temp)
    np.savetxt(s.data_out + 'antsim_temp.txt', s.antsim_temp)

    plt.show()
