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


class s1p:
    _mapping = {
        "open": "Open0",
        "short": "Short0",
        "match": "Match0",
        "external": "External0",
        "receiver": "ReceiverReading0"
    }

    def __init__(self, base_path, run_num, ignore=None):
        self.base_path = base_path
        self.run_num = run_num

        # Read in spectra files
        for k, (spec, f) in self.read(ignore or []):
            setattr(self, k, spec)
            self.f = f

    def read(self, ignore=None):
        ignore = ignore or []
        for k, v in self._mapping.items():
            if k in ignore:
                continue

            yield k, rc.s1p_read(os.path.join(self.base_path, "{}{}.s1p".format(v, self.run_num)))

    @property
    def switchval_open(self):
        if not hasattr(self, "open"):
            return None
        else:
            return np.ones_like(self.f)

    @property
    def switchval_short(self):
        if not hasattr(self, "short"):
            return None
        else:
            return -1 * np.ones_like(self.f)

    @property
    def switchval_match(self):
        if not hasattr(self, "match"):
            return None
        else:
            return np.zeros_like(self.f)

    def get_corrections(self):
        # Correction at switch
        return rc.de_embed(
            self.switchval_open, self.switchval_short, self.switchval_match,
            self.open, self.short, self.match, self.external
        )


class spectra(object):
    _kinds = ['ambient', 'hot_load', 'open', 'short', 'antsim']
    _filemap = ['Ambient', "HotLoad", 'LongCableOpen', "LongCableShort", "AntSim4"]

    def __init__(self, data_out, path, flow, fhigh, percent, runNum):
        # Initialize file paths and spectra parameters
        self.data_out = data_out
        self.path = path
        self.path_res = os.path.join(path, 'Resistance')
        self.path_spec = os.path.join(path, 'Spectra', 'mat_files')
        self.path_s11 = os.path.join(path, 'S11')
        self.flow = flow  # 50
        self.fhigh = fhigh  # 190
        self.f_range = fhigh - flow
        self.f_center = flow + self.f_range / 2.0
        self.percent = percent  # 5.0
        self.runNum = runNum

        self.ff, self.ilow, self.ihigh = rcf.frequency_edges(self.flow, self.fhigh)
        self.fe = self.ff[self.ilow:self.ihigh + 1]
        self.fen = (self.fe - self.f_center) / (self.f_range / 2)

    def construct_average(self, kind, percent=5.0, spec_files=None, res_files=None):
        if spec_files is None:
            spec_files = glob.glob(os.path.join(self.path_spec, self._filemap[self._kinds.index(kind)] + '*.mat'))
        if res_files is None:    
            res_files = glob.glob(os.path.join(self.path_res, self._filemap[self._kinds.index(kind)] + '*.txt'))

        if not spec_files:
            raise FileNotFoundError("No .mat files found for {}".format(kind))
        if not res_files:
            raise FileNotFoundError("No .txt files found for {}".format(kind))
        #print(spec_files)
        #print(res_files)
        setattr(
            self, kind + "_ave",
            rcf.AverageCal(spec_files, res_files, percent)
        )

    def save(self):
        for kind in self._kinds:
            np.savetxt(os.path.join(self.data_out, kind + "_spec.txt"), getattr(self, "s_{}".format(kind)))


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


def spec_read(s, percent=5.0, spec_files=None, res_files=None):
    for kind in s._kinds:
        if spec_files is not None: 
            speclist=[file for file in spec_files if file.lower().find(kind)!=-1]
            if not speclist: speclist=None
        else: speclist=spec_files
        if res_files is not None: 
            reslist=[file for file in res_files if file.lower().find(kind)!=-1]
            if not reslist: reslist=None
        else: reslist=res_files
        print("Reading {}".format(kind))
        #print(speclist)
        #print(reslist)
        s.construct_average(kind, percent * (4 if kind == 'hot_load' else 1), spec_files=speclist, res_files=reslist)
        setattr(
            s, "s_{}".format(kind),
            getattr(s, "{}_ave".format(kind)).ave_spectrum[s.ilow:s.ihigh + 1]
        )
    print("Finished reading")

def spec_plot(s):
    fig, ax = plt.subplots(5, 1, figsize=(12, 8), facecolor='white', sharex=True)

    for i, kind in enumerate(s._kinds):
        ax[i].plot(s.fe, getattr(s, "s_{}".format(kind)))
        ax[i].grid(True)
        ax[i].set_ylabel(s._filemap[i] + " [K]")
        ax[i].set_xlabel("Frequency [MHz]")
    plt.show()
    # return fig


def s11_model(spec, s11_path, resistance_f=50.009, resistance_m=50.166):
    s11_data = np.genfromtxt(os.path.join(spec.path_s11, 'semi_rigid_s_parameters_WITH_HEADER.txt'))

    index_s11 = np.where(np.logical_and(s11_data >= spec.flow, s11_data <= spec.fhigh))[0]
    f_range = spec.fhigh - spec.flow

    f_s11n_r = (s11_data[index_s11, 0] - (f_range / 2 + spec.flow)) / (f_range / 2)
    s11_sr = s11_data[index_s11, 1] + 1j * s11_data[index_s11, 2]
    s12s21_sr = s11_data[index_s11, 3] + 1j * s11_data[index_s11, 4]
    s22_sr = s11_data[index_s11, 5] + 1j * s11_data[index_s11, 6]

    # TODO: the following should be in a loop!
    fit_s11_sr_mag = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, np.abs(s11_sr), 21)
    fit_s11_sr_ang = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, np.unwrap(np.angle(s11_sr)), 21)

    fit_s12s21_sr_mag = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, np.abs(s12s21_sr), 21)
    fit_s12s21_sr_ang = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, np.unwrap(np.angle(s12s21_sr)), 21)

    fit_s22_sr_mag = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, np.abs(s22_sr), 21)
    fit_s22_sr_ang = rcf.fit_polynomial_fourier('polynomial', f_s11n_r, np.unwrap(np.angle(s22_sr)), 21)

    model_s11_sr_mag = rcf.model_evaluate('polynomial', fit_s11_sr_mag[0], spec.fen)
    model_s11_sr_ang = rcf.model_evaluate('polynomial', fit_s11_sr_ang[0], spec.fen)

    model_s12s21_sr_mag = rcf.model_evaluate('polynomial', fit_s12s21_sr_mag[0], spec.fen)
    model_s12s21_sr_ang = rcf.model_evaluate('polynomial', fit_s12s21_sr_ang[0], spec.fen)

    model_s22_sr_mag = rcf.model_evaluate('polynomial', fit_s22_sr_mag[0], spec.fen)
    model_s22_sr_ang = rcf.model_evaluate('polynomial', fit_s22_sr_ang[0], spec.fen)

    # Receiver reflection coefficient
    print('Reflection Coefficient')

    # Reading measurements
    path_LNA = os.path.join(spec.path_s11, 'ReceiverReading01')
    measurements = s1p(path_LNA, spec.runNum, ignore=['external'])

    # Models of standards
    oa, sa, la = rc.agilent_85033E(measurements.f, resistance_f, m=1)

    # Correction of measurements
    LNAc = rc.de_embed(oa, sa, la, measurements.open, measurements.short,
                       measurements.match, measurements.receiver)[0]
    LNA = LNAc[(measurements.f / 1e6 >= spec.flow) & (measurements.f / 1e6 <= spec.fhigh)]
    spec.f_s11 = measurements.f[(measurements.f / 1e6 >= spec.flow) & (measurements.f / 1e6 <= spec.fhigh)]
    spec.s11_LNA_mag = np.abs(LNA)
    spec.s11_LNA_ang = np.unwrap(np.angle(LNA))

    f_s11n = (spec.f_s11 / 1e6 - ((spec.fhigh - spec.flow) / 2 + spec.flow)) / ((spec.fhigh - spec.flow) / 2)

    spec.fit_s11_LNA_mag = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_LNA_mag, 37)
    spec.fit_s11_LNA_ang = rcf.fit_polynomial_fourier('fourier', f_s11n, spec.s11_LNA_ang, 37)

    spec.model_s11_LNA_mag = rcf.model_evaluate('fourier', spec.fit_s11_LNA_mag[0], spec.fen)
    spec.model_s11_LNA_ang = rcf.model_evaluate('fourier', spec.fit_s11_LNA_ang[0], spec.fen)
    spec.r_lna = spec.model_s11_LNA_mag * (np.cos(spec.model_s11_LNA_ang) + 1j * np.sin(spec.model_s11_LNA_ang))

    kind_mapping = {
        'ambient': "Ambient",
        'hot_load': "HotLoad",
        'open': "LongCableOpen",
        'short': "LongCableShorted",
        'antsim': "AntSim4"
    }

    kind_nterms = {
        'ambient': 37,
        'hot_load': 37,
        'open': 105,
        'short': 105,
        'antsim': 55
    }

    models = {}
    corrs = {}
    datas = {}
    for kind, long_kind in kind_mapping.items():
        pth = os.path.join(spec.path_s11, long_kind)
        datas[kind] = s1p(pth, spec.runNum, ignore=['receiver'])
        _corr = s11.low_band_switch_correction_june_2016(
            s11_path, datas[kind].get_corrections()[0], f_in=datas[kind].f,
            flow=spec.flow, fhigh=spec.fhigh, resistance_m=resistance_m
        )
        corrs[kind] = _corr[(datas[kind].f / 1e6 >= spec.flow) & (datas[kind].f / 1e6 <= spec.fhigh)]

        setattr(spec, "s11_{}_mag".format(kind), np.abs(corrs[kind]))
        setattr(spec, "s11_{}_ang".format(kind), np.unwrap(np.angle(corrs[kind])))

        models[kind] = {}
        for t in ['mag', 'ang']:
            setattr(
                spec, "fit_s11_{}_{}".format(kind, t),
                rcf.fit_polynomial_fourier(
                    'fourier', f_s11n,
                    getattr(spec, "s11_{}_{}".format(kind, t)), kind_nterms[kind]
                )
            )
            models[kind][t] = rcf.model_evaluate(
                'fourier', getattr(spec, "fit_s11_{}_{}".format(kind, t))[0], spec.fen
            )

        ra = models[kind]['mag'] * (np.cos(models[kind]['ang']) + 1j * np.sin(models[kind]['ang']))
        setattr(spec, "r_{}".format(kind), ra)

    s11_sr = model_s11_sr_mag * np.cos(model_s11_sr_ang) + 1j * model_s11_sr_mag * np.sin(model_s11_sr_ang)
    s12s21_sr = model_s12s21_sr_mag * np.cos(model_s12s21_sr_ang) + 1j * model_s12s21_sr_mag * np.sin(
        model_s12s21_sr_ang)
    s22_sr = model_s22_sr_mag * np.cos(model_s22_sr_ang) + 1j * model_s22_sr_mag * np.sin(model_s22_sr_ang)

    rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, spec.r_hot_load)

    # inverting the direction of the s-parameters,
    # since the port labels have to be inverted to match those of Pozar eqn 10.25
    s11_sr_rev = s22_sr

    # absolute value of S_21
    abs_s21 = np.sqrt(np.abs(s12s21_sr))

    # available power gain
    G = (abs_s21 ** 2) * (1 - np.abs(rht) ** 2) / (
            (np.abs(1 - s11_sr_rev * rht)) ** 2 * (1 - (np.abs(spec.r_hot_load)) ** 2))

    # temperature
    spec.Thd = G * spec.hot_load_ave.temp_ave + (1 - G) * spec.ambient_ave.temp_ave


def s11_cal(spec, cterms, wterms):
    print('Calibration coefficients')
    spec.cterms = cterms
    spec.wterms = wterms

    # ==================================================================================
    # This block originally in function s11_model(). It was moved to keep cterm/wterms
    # defined here only.
    temp = np.array([spec.fe, spec.model_s11_LNA_mag, spec.model_s11_LNA_ang])
    output_file = temp.T
    output_file_str = os.path.join(
        spec.data_out,
        'LNA_{}_{}_25C_{}-{}__fe-350-load.txt'.format(spec.flow, spec.fhigh, cterms, wterms)
    )
    np.savetxt(output_file_str, output_file)
    # ==================================================================================

    spec.scale, spec.off, spec.Tu, spec.TC, spec.TS = rcf.calibration_quantities(
        spec.flow, spec.fhigh, spec.fe, spec.s_ambient, spec.s_hot_load, spec.s_open,
        spec.s_short, spec.r_lna, spec.r_ambient, spec.r_hot_load, spec.r_open,
        spec.r_short, spec.ambient_ave.temp_ave, spec.Thd, spec.open_ave.temp_ave,
        spec.short_ave.temp_ave, spec.cterms, spec.wterms
    )
    for kind in spec._kinds:
        setattr(
            spec, "{}_calibrated".format(kind),
            rcf.calibrated_antenna_temperature(
                getattr(spec, "s_{}".format(kind)),
                getattr(spec, "r_{}".format(kind)),
                spec.r_lna, spec.scale, spec.off, spec.Tu, spec.TC, spec.TS,
                Tamb_internal=300
            )
        )


def residual_plot(s, kind):
    fig, ax = plt.subplots(4, 1, sharex=True, facecolor='w')
    for axx in ax:
        axx.xaxis.set_ticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], [])
        axx.grid(True)
        axx.set_xlabel("Frequency [MHz]")

    fig.suptitle(kind)

    ax[0].plot(s.f_s11 / 1e6, 20 * np.log10(s.fit_s11_LNA_mag[1]))
    ax[0].set_ylabel('S11(mag)')

    ax[1].plot(s.f_s11 / 1e6, (s.fit_s11_LNA_mag[1] - s.s11_LNA_mag), 'g')
    ax[1].set_ylabel('Delta S11(mag)')

    ax[2].plot(s.f_s11 / 1e6, s.fit_s11_LNA_ang[1] * 180 / np.pi)
    ax[2].set_ylabel(' S11(Ang)')

    ax[3].plot(s.f_s11 / 1e6, (s.s11_LNA_ang - s.fit_s11_LNA_ang[1]), 'g')
    ax[3].set_ylabel('Delta S11(Ang)')

    return fig


def cal_plot(s, kind, bins=64):
    lim = len(s.fe)

    # binning
    fact = len(getattr(s, "{}_calibrated".format(kind))[:lim]) / bins
    fnew = np.linspace(s.flow, s.fhigh, bins)

    cal = np.zeros(bins)

    for i in range(bins):
        cal[i] = np.average(s.antsim_calibrated[int(i * fact):int((i + 1) * fact)])

    stop = bins
    rms = np.sqrt(np.mean((cal[:stop] - np.mean(cal[:stop])) ** 2))

    plt.figure(facecolor='w')
    plt.plot(fnew[:stop], cal[:stop], 'r', label='Calibrated')
    plt.text(65, np.max(cal), 'RMS=' + str(np.round(rms, 3)) + 'K')
    plt.ylim([np.min(cal), np.max(cal)])
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Ant_sim3')
    plt.ticklabel_format(useOffset=False)
    plt.grid()

    return plt.gcf()


def s11_plot(s):
    # TODO: break this up into sub-functions

    figs = []
    for kind in ['short', 'open', 'hot_load', 'antsim', 'lna']:
        figs.append(residual_plot(s, kind))
        if kind == 'antsim':
            figs.append(cal_plot(s, kind))

    # turn temp into variable
    # TODO: why is this here?
    np.savetxt(
        os.path.join(
            s.data_out, 'All_cal-params_{}_{}_{}-{}_25C_s11alan.txt'.format(
                s.flow, s.fhigh, s.cterms, s.wterms)
        ),
        [s.fe, s.scale, s.off, s.Tu, s.TC, s.TS]
    )

    labels = ["Scale (C1)", "Offset (C2) [K]", "TU [K]", "TC [K]", "TS [K]"]
    for kind, label in zip(['scale', 'off', "Tu", "TC", "TS"], labels):
        plt.figure(facecolor='w')
        plt.plot(s.fe, getattr(s, kind))
        plt.xlabel("Frequency [MHz]")
        plt.ylabel(label)
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        figs.append(plt.gcf())

    # Plot temperatures in Celsius
    fig, ax = plt.subplots(5, 1, sharex=True, facecolor='w')
    for i, kind in enumerate(['ambient', 'hot_load', 'open', 'short', 'antsim']):
        ax[i].plot(getattr(s, "{}_ave".format(kind)).thermistor_temp[::120] - 273)
        ax[i].grid(True)
        ax[i].set_ylabel("{} [$^o$ C]".format(kind))

    figs.append(fig)
    return figs


def s11_write(s):
    # TODO: this shouldn't be in a plotting function!
    np.savetxt(s.data_out + 'fit_s11_LNA_mag.txt', s.fit_s11_LNA_mag[1])
    np.savetxt(s.data_out + 'fit_s11_LNA_ang.txt', s.fit_s11_LNA_ang[1])

    for kind in s._kinds:
        for t in ['mag', 'ang']:
            key = 'fit_s11_{}_{}'.format(kind, t)
            indx = 1 if kind in ['ambient', 'hot_load'] else 0
            np.savetxt(
                os.path.join(s.data_out, key + ".txt"), getattr(s, key)[indx]
            )

        key = "{}_ave".format(kind)
        np.savetxt(os.path.join(s.data_out, key + ".txt"), getattr(s, key).thermistor_temp)
