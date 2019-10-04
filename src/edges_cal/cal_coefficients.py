# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2019
Original author: Nivedita Mahesh
Edited by: David Lewis, Steven Murray

"""
import glob
import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property

from . import S11_correction as s11
from . import modelling as mdl
from . import receiver_calibration_func as rcf
from . import reflection_coefficient as rc


class s1p:
    _mapping = {
        "open": "Open0",
        "short": "Short0",
        "match": "Match0",
        "external": "External0",
        "receiver": "ReceiverReading0",
    }

    def __init__(self, base_path, run_num, f_low, f_high, ignore=None):
        self.base_path = base_path
        self.run_num = run_num
        self.freq = Frequencies(f_low, f_high)

        # Read in Spectra files
        for k, (spec, f) in self.read(ignore or []):
            self.f_full = f / 1e6
            setattr(self, k, spec[self.f_mask])

    @property
    def f_mask(self):
        """Mask choosing frequencies within the given bounds"""
        return self.f_full >= self.freq.f_low & self.f_full <= self.freq.f_high

    @property
    def f(self):
        """Frequencies within the given bounds"""
        return self.f_full[self.f_mask]

    @property
    def f_norm(self):
        """Frequencies scaled from -1 to 1"""
        return (
            2 * (self.f - (self.freq.f_range / 2 + self.freq.f_low)) / self.freq.f_range
        )

    def read(self, ignore=None):
        ignore = ignore or []
        for k, v in self._mapping.items():
            if k in ignore:
                continue
            s1pPath = os.path.join(self.base_path, "{}{}.s1p".format(v, self.run_num))
            if os.path.isfile(s1pPath):
                yield k, rc.s1p_read(s1pPath)
            elif k == "receiver":
                s1pPath = os.path.join(
                    self.base_path, "Receiver0{}.s1p".format(self.run_num)
                )
                yield k, rc.s1p_read(s1pPath)

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
            self.switchval_open,
            self.switchval_short,
            self.switchval_match,
            self.open,
            self.short,
            self.match,
            self.external,
        )


class Frequencies:
    def __init__(self, f_low, f_high):
        self.f_low = f_low
        self.f_high = f_high
        self.f_range = f_high - f_low
        self.f_center = f_low + self.f_range / 2.0

        full_freq, self.lower_index, self.upper_index = rcf.frequency_edges(
            self.f_low, self.f_high
        )
        self.f_edges = full_freq[self.mask]
        self.f_edges_recentred = (self.f_edges - self.f_center) / (self.f_range / 2)

    @property
    def mask(self):
        return slice(self.lower_index, self.upper_index)

    def norm(self, freq):
        return 2 * (freq - self.f_center) / self.f_range


class Spectra:
    _kinds = {
        "ambient": "Ambient",
        "hot_load": "HotLoad",
        "open": "LongCableOpen",
        "short": "LongCableShort",
        "antsim": "AntSim",
    }

    def __init__(self, kind, path, f_low, f_high, percent=5):
        # Initialize file paths and Spectra parameters
        self.kind = kind
        self.path = path
        self.path_res = os.path.join(path, "Resistance")
        self.path_spec = os.path.join(path, "Spectra", "mat_files")

        self.percent = percent
        self.freq = Frequencies(f_low=f_low, f_high=f_high)

    @cached_property
    def average_cal(self):
        """
        An :class:`~rcf.AverageCal` instance of the specified input files.
        """
        spec_files = glob.glob(
            os.path.join(self.path_spec, self._kinds[self.kind] + "*.mat")
        )
        res_files = glob.glob(
            os.path.join(self.path_res, self._kinds[self.kind] + "*.txt")
        )

        if not spec_files:
            raise FileNotFoundError("No .mat files found for {}".format(self.kind))
        if not res_files:
            raise FileNotFoundError("No .txt files found for {}".format(self.kind))

        return rcf.AverageCal(spec_files, res_files, self.percent)

    @cached_property
    def averaged_spectrum(self):
        """An averaged temperature spectrum as a function of frequency"""
        return self.average_cal.ave_spectrum[self.freq.mask]

    @cached_property
    def thermistor_temp(self):
        """The thermistor temperature as a function of frequency"""
        return self.average_cal.thermistor_temp[self.freq.mask]

    def save(self, direc):
        np.savetxt(
            os.path.join(direc, self.kind + "_spec.txt"),
            getattr(self, "s_{}".format(self.kind)),
        )

    def plot(self, fig=None, ax=None, **kwargs):
        if fig is None:
            fig, ax = plt.subplots(1, 1, facecolor=kwargs.get("facecolor", "white"))

        ax.plot(self.freq.f_edges, getattr(self, "s_{}".format(self.kind)))
        ax.grid(True)
        ax.set_ylabel(self._kinds[self.kind] + " [K]")
        ax.set_xlabel("Frequency [MHz]")

    @cached_property
    def s1p(self):
        pth = os.path.join(self.path, self._kinds[self.kind])
        if not os.path.isdir(pth) and self.kind == "ambient":
            pth = os.path.join(self.path_s11, self._kinds[self.kind] + "Load")

        return s1p(pth, self.runNum, ignore=["receiver"])

    def get_s11_correction(self, resistance_m=50.166):

        corr = s11.low_band_switch_correction_june_2016(
            self.path_s11,
            self.s1p.get_corrections()[0],
            f_in=self.s1p.f,
            resistance_m=resistance_m,
        )

        return corr

    def get_s11_correction_model(self, nterms=None, corr=None, resistance_m=50.166):
        kind_nterms = {
            "ambient": 37,
            "hot_load": 37,
            "open": 105,
            "short": 105,
            "antsim": 55,
        }
        nterms = nterms or kind_nterms[self.kind]

        if corr is None:
            corr = self.get_s11_correction(resistance_m)

        def getmodel(mag):
            if mag:
                d = np.abs(corr)
            else:
                d = np.unwrap(np.angle(corr))

            fit = mdl.fit_polynomial_fourier("fourier", self.s1p.f_norm, d, nterms)
            return mdl.model_evaluate("fourier", fit[0], self.freq.f_edges_recentred)

        mag = getmodel(True)
        ang = getmodel(False)

        return mag * (np.cos(ang) + 1j * np.sin(ang))

    def plot_residuals(self):
        fig, ax = plt.subplots(4, 1, sharex=True, facecolor="w")
        for axx in ax:
            axx.xaxis.set_ticks(
                [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], []
            )
            axx.grid(True)
            axx.set_xlabel("Frequency [MHz]")

        fig.suptitle(self.kind)

        corr = self.get_s11_correction()
        model = self.get_s11_correction_model(corr=corr)

        ax[0].plot(self.freq.f_edges, 20 * np.log10(np.abs(model)))
        ax[0].set_ylabel("S11(mag)")

        ax[1].plot(self.freq.f_edges, np.abs(model) - np.abs(corr), "g")
        ax[1].set_ylabel("Delta S11(mag)")

        ax[2].plot(self.freq.f_edges, np.unwrap(np.angle(model)) * 180 / np.pi)
        ax[2].set_ylabel(" S11(Ang)")

        ax[3].plot(
            self.freq.f_edges,
            np.unwrap(np.angle(model)) - np.unwrap(np.angle(corr)),
            "g",
        )
        ax[3].set_ylabel("Delta S11(Ang)")

        return fig

    def plot_thermistor_temp(self, step=120, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="w")

        fig.suptitle("Thermistor Temperature")
        ax.plot(self.thermistor_temp[::step] - 273, label=self.kind)
        ax.grid(True)
        ax.set_ylabel("Temperature [$^o$ C]".format(self.kind))
        ax.set_xlabel(self.average_cal)


class SParam:
    _kinds = {"s11": 1, "s12": 3, "s22": 5}

    def __init__(self, kind, path, f_low, f_high):
        self.kind = kind
        self.path = path
        data = np.genfromtxt(
            os.path.join(path, "semi_rigid_s_parameters_WITH_HEADER.txt")
        )
        self.freq = Frequencies(f_low, f_high)

        f = data[:, 0]
        mask = np.where(np.logical_and(f >= f_low, f <= f_high))[0]
        index = self._kinds[self.kind]

        self.f_norm = self.freq.norm(f[mask])
        self.data = data[mask, index] + 1j * data[mask, index + 1]

    def _get_model_part(self, mag=True):
        """
        Compute an evaluated S11 model, having fit to the data.
        Parameters
        ----------
        mag : bool, optional
            Whether to return the magnitude (otherwise, the angle)

        Returns
        -------
        array_like : The model S-parameter
        """
        if mag:
            d = np.abs(self.data)
        else:
            d = np.unwrap(np.angle(self.data))

        mag = mdl.fit_polynomial_fourier("polynomial", self.f_norm, d, 21)

        return mdl.model_evaluate("polynomial", mag[0], self.freq.f_edges_recentred)

    @cached_property
    def model(self):
        mag = self._get_model_part()
        ang = self._get_model_part(False)
        return mag * (np.cos(ang) + 1j * np.sin(ang))


class LNA:
    def __init__(self, path, run_num, f_low, f_high, resistance_f=50.009):
        path = os.path.join(path, "ReceiverReading0" + str(run_num))
        self.s1p = s1p(path, run_num, ignore=["external"])
        self.freq = Frequencies(f_low, f_high)
        self.resistance_f = resistance_f

    @cached_property
    def data(self):
        # Models of standards
        oa, sa, la = rc.agilent_85033E(self.s1p.f, self.resistance_f, m=1)

        # Correction of measurements
        return rc.de_embed(
            oa, sa, la, self.s1p.open, self.s1p.short, self.s1p.match, self.s1p.receiver
        )[0]

    def _get_model_part(self, mag=True):
        if mag:
            d = np.abs(self.data)
        else:
            d = np.unwrap(np.angle(self.data))

        fit = mdl.fit_polynomial_fourier("fourier", self.s1p.f_norm, d, 37)

        return mdl.model_evaluate("fourier", fit[0], self.freq.f_edges_recentred)

    @cached_property
    def model(self):
        """Fourier-based model-fit to the data"""
        mag = self._get_model_part()
        ang = self._get_model_part(False)
        return mag * (np.cos(ang) + 1j * np.sin(ang))

    def residual_plot(self):
        fig, ax = plt.subplots(4, 1, sharex=True, facecolor="w")
        for axx in ax:
            axx.xaxis.set_ticks(
                [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], []
            )
            axx.grid(True)
            axx.set_xlabel("Frequency [MHz]")

        fig.suptitle("LNA")

        ax[0].plot(self.s1p.f, 20 * np.log10(np.abs(self.model)))
        ax[0].set_ylabel("S11(mag)")

        ax[1].plot(self.s1p.f, np.abs(self.model) - np.abs(self.data), "g")
        ax[1].set_ylabel("Delta S11(mag)")

        ax[2].plot(self.s1p.f, np.unwrap(np.angle(self.model)) * 180 / np.pi)
        ax[2].set_ylabel(" S11(Ang)")

        ax[3].plot(
            self.s1p.f,
            np.unwrap(np.angle(self.model)) - np.unwrap(np.angle(self.data)),
            "g",
        )
        ax[3].set_ylabel("Delta S11(Ang)")

        return fig


class CalibrationObservation:
    _sources = ["ambient", "hot_load", "open", "short"]

    def __init__(
        self,
        path,
        f_low,
        f_high,
        run_num,
        percent=5,
        resistance_f=50.009,
        resistance_m=50.166,
        add_antsim=True,
        cterms=5,
        wterms=7,
    ):
        self.path = path

        self.ambient = Spectra("ambient", path, f_low, f_high, percent)
        self.hot_load = Spectra("hot_load", path, f_low, f_high, percent)
        self.open = Spectra("open", path, f_low, f_high, percent)
        self.short = Spectra("short", path, f_low, f_high, percent)
        if add_antsim:
            self.antsim = Spectra("antsim", path, f_low, f_high, percent)

        self.add_antsim = add_antsim

        for s in ["s11", "s12", "s22"]:
            setattr(self, s, SParam(kind=s, path=path, f_low=f_low, f_high=f_high))

        self.lna = LNA(path, run_num, f_low, f_high, resistance_f)
        self.resistance_m = resistance_m

        self.cterms = cterms
        self.wterms = wterms

        # Expose a Frequency object
        self.edges_freq = self.ambient.freq

    @cached_property
    def Thd(self):
        hot_load_correction = self.hot_load.get_s11_correction(
            resistance_m=self.resistance_m
        )
        rht = rc.gamma_de_embed(
            self.s11.model, self.s12.model, self.s22.model, hot_load_correction
        )

        # inverting the direction of the s-parameters,
        # since the port labels have to be inverted to match those of Pozar eqn 10.25
        s11_sr_rev = self.s22.model

        # absolute value of S_21
        abs_s21 = np.sqrt(np.abs(self.s12.model))

        # available power gain
        G = (
            (abs_s21 ** 2)
            * (1 - np.abs(rht) ** 2)
            / (
                (np.abs(1 - s11_sr_rev * rht)) ** 2
                * (1 - (np.abs(hot_load_correction)) ** 2)
            )
        )

        # temperature
        return (
            G * self.hot_load.average_cal.temp_ave
            + (1 - G) * self.ambient.average_cal.temp_ave
        )

    @cached_property
    def s11_correction_models(self):
        return {
            k: getattr(self, k).get_s11_correction(resistance_m=self.resistance_m)
            for k in self._sources
        }

    @cached_property
    def calibration_coefficients(self):
        scale, off, Tu, TC, TS = rcf.calibration_quantities(
            self.edges_freq.f_low,
            self.edges_freq.f_high,
            self.edges_freq.f_edges,
            self.ambient.averaged_spectrum,
            self.hot_load.averaged_spectrum,
            self.open.averaged_spectrum,
            self.short.averaged_spectrum,
            self.lna.model,
            self.s11_correction_models["ambient"],
            self.s11_correction_models["hot_load"],
            self.s11_correction_models["open"],
            self.s11_correction_models["short"],
            self.ambient.average_cal.temp_ave,
            self.Thd,
            self.open.average_cal.temp_ave,
            self.short.average_cal.temp_ave,
            self.cterms,
            self.wterms,
        )
        return scale, off, Tu, TC, TS

    @lru_cache()
    def calibrate(self, kind):
        scale, off, Tu, TC, TS = self.calibration_coefficients

        return rcf.calibrated_antenna_temperature(
            getattr(self, kind).averaged_spectrum,
            self.s11_correction_models[kind],
            self.lna.model,
            scale,
            off,
            Tu,
            TC,
            TS,
            Tamb_internal=300,
        )

    def plot_calibrated_temps(self, kind, bins=64):
        lim = len(self.edges_freq.f_edges)

        # binning
        temp_calibrated = self.calibrate(kind)
        fact = lim / bins
        fnew = np.linspace(self.edges_freq.f_low, self.edges_freq.f_high, bins)

        # TODO: this would probably be better using a convolution kernel
        freq_ave_cal = np.zeros(bins)
        for i in range(bins):
            freq_ave_cal[i] = np.average(
                temp_calibrated[int(i * fact) : int((i + 1) * fact)]
            )

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        plt.figure(facecolor="w")
        plt.plot(fnew, freq_ave_cal, "b", label=f"Calibrated {kind}")

        if kind != "hot_load":
            plt.axhline(getattr(self, kind).average_cal.temp_ave, color="r")
        else:
            plt.plot(self.edges_freq.f_edges, self.Thd, color="r")

        plt.text(
            self.edges_freq.f_low + self.edges_freq.f_range / 6,
            np.max(freq_ave_cal),
            f"RMS={rms:.3f} [K]",
        )
        plt.ylim([np.min(freq_ave_cal), np.max(freq_ave_cal)])
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Temperature [K]")
        plt.ticklabel_format(useOffset=False)
        plt.grid()

        return plt.gcf()

    def write_coefficients(self, direc="."):
        scale, off, Tu, TC, TS = self.calibration_coefficients
        np.savetxt(
            os.path.join(
                direc,
                "All_cal-params_{}_{}_{}-{}_25C_s11alan.txt".format(
                    self.edges_freq.f_low,
                    self.edges_freq.f_high,
                    self.cterms,
                    self.wterms,
                ),
            ),
            [self.edges_freq.f_edges, scale, off, Tu, TC, TS],
        )

    def plot_coefficients(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="w")

        labels = ["Scale (C1)", "Offset (C2) [K]", "TU [K]", "TC [K]", "TS [K]"]
        for soln, label in zip(self.calibration_coefficients, labels):
            plt.plot(self.edges_freq.f_edges, soln)
            plt.xlabel("Frequency [MHz]")
            plt.ylabel(label)
            plt.grid()
            plt.ticklabel_format(useOffset=False)

        return fig

        # plot calibrated temperature in K

    #        for kind in ["ambient", "hot_load", "open", "short", "antsim"]:
    #            figs.append(cal_plot(s, kind))

    # Plot Thermistor temperatures in Celsius

    def write(self, direc="."):
        np.savetxt(os.path.join(direc, "fit_s11_LNA_mag.txt"), np.abs(self.lna.model))
        np.savetxt(
            os.path.join(direc, "fit_s11_LNA_ang.txt"),
            np.unwrap(np.angle(self.lna.model)),
        )

        sources = tuple(self._sources)
        if self.add_antsim:
            sources = sources + ("antsim",)

        for source in sources:
            src = getattr(self, source)
            for part, fnc in zip(
                ["mag", "ang"], [np.abs, lambda x: np.unwrap(np.angle(x))]
            ):
                out = fnc(src.get_s11_correction_model(resistance_m=self.resistance_m))
                key = "fit_s11_{}_{}".format(source, part)
                np.savetxt(os.path.join(direc, key + ".txt"), out)

            key = "{}_thermistor_temp".format(source)
            np.savetxt(os.path.join(direc, key + ".txt"), src.thermistor_temp)
