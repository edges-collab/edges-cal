# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2019
Original author: Nivedita Mahesh
Edited by: David Lewis, Steven Murray

"""
import glob
import os
from functools import lru_cache

import h5py
import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property

from . import S11_correction as s11
from . import modelling as mdl
from . import receiver_calibration_func as rcf
from . import reflection_coefficient as rc


class FrequencyRange:
    """
    Class defining a set of frequencies.

    A given frequency range can be cut on either end.

    Parameters
    ----------
    f : array_like
        An array of frequencies defining a given spectrum.
    f_low : float, optional
        A minimum frequency to keep in the array. Default is min(f).
    f_higih : float, optional
        A minimum frequency to keep in the array. Default is min(f).
    """

    def __init__(self, f, f_low=None, f_high=None):
        self.freq_full = f
        self._f_high = f_high or f.max()
        self._f_low = f_low or f.min()

        if f_low >= f_high:
            raise ValueError("Cannot create frequency range: f_low >= f_high")

    @cached_property
    def mask(self):
        return self.freq_full >= self._f_low * self.freq_full <= self._f_high

    @cached_property
    def freq(self):
        return self.freq_full[self.mask]

    @cached_property
    def range(self):
        return self.freq.max() - self.freq.min()

    @cached_property
    def center(self):
        return self.freq.min() + self.range / 2.0

    @cached_property
    def freq_recentred(self):
        return self.normalize(self.freq)

    def normalize(self, f):
        """
        Normalise a set of frequencies such that -1 aligns with f_low and
        +1 aligns with f_high.

        Parameters
        ----------
        f : array_like
            Frequencies to normalize

        Returns
        -------
        array_like, shape [f,]
            The normalized frequencies.
        """
        return 2 * (f - self.center) / self.range


class EdgesFrequencyRange(FrequencyRange):
    """
    A subclass of :class:`FrequencyRange` specifying the default EDGES frequencies.
    """

    def __init__(self, nchannels=16384 * 2, max_freq=200.0, f_low=None, f_high=None):
        f = self.get_edges_freqs(nchannels, max_freq)
        super().__init__(f, f_low, f_high)

    def get_edges_freqs(self, nchannels=16384 * 2, max_freq=200.0):
        """
        Return the raw EDGES frequency array, in MHz.

        Parameters
        ----------
        nchannels : int, optional
            Number of channels in the EDGES spectrum
        max_freq : float, optional
            Maximum frequency in the spectrum.

        Returns
        -------
        freqs: 1D-array
            full frequency array from 0 to 200 MHz, at raw resolution
        """
        # Full frequency vector
        fstep = max_freq / nchannels
        freqs = np.arange(0, max_freq, fstep)
        return freqs


class VNA:
    # _mapping = {
    #     "open": "Open0",
    #     "short": "Short0",
    #     "match": "Match0",
    #     "external": "External0",
    #     "receiver": "ReceiverReading0",
    # }

    def __init__(self, fname, f_low=None, f_high=None, run_num=None, switchval=None):
        self.fname = fname
        if run_num is None:
            self.run_num = int(self.fname.split(".")[-2][-2:])

        self.load_name = os.path.splitext(os.path.basename(self.fname))[0][:-2]

        f, spec = self._read(self.fname)
        self.freq = FrequencyRange(f, f_low, f_high)
        self.s11 = spec[self.freq.mask]
        self._switchval = switchval

    def _read(self):
        return rc.s1p_read(self.fname)

    @cached_property
    def switchval(self):
        if self._switchval is not None:
            return self._switchval * np.ones_like(self.freq)
        else:
            return None


class SwitchCorrection:
    def __init__(self, base_path, f_low=None, f_high=None, run_num=0):
        self.base_path = base_path
        self.run_num = run_num
        self.open = VNA(
            os.path.join(base_path, "Open{:.2d}".format(run_num)),
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            switchval=1,
        )
        self.short = VNA(
            os.path.join(base_path, "Short{:.2d}".format(run_num)),
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            switchval=-1,
        )
        self.match = VNA(
            os.path.join(base_path, "Match{:.2d}".format(run_num)),
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            switchval=0,
        )

        if not (self.open.freq.freq == self.short.freq.freq == self.match.freq.freq):
            raise ValueError("S11 files do not match")

        # Expose one of the frequency objects
        self.freq = self.open.freq

    @cached_property
    def external(self):
        """
        VNA S11 measurements for the load.
        """
        return VNA(
            os.path.join(self.base_path, "External{:.2d}".format(self.run_num)),
            f_low=self.freq.f_low,
            f_high=self.freq.f_high,
            run_num=self.run_num,
        )

    @cached_property
    def switch_corrections(self):
        # Correction at switch
        return rc.de_embed(
            self.open.switchval,
            self.short.switchval,
            self.match.switchval,
            self.open.s11,
            self.short.s11,
            self.match.s11,
            self.external.s11,
        )

    @lru_cache()
    def get_s11_correction(self, resistance_m=50.166):
        """
        Determine the correction required for the S11 due to the switch.
        """
        return s11.low_band_switch_correction_june_2016(
            self.base_path,
            self.switch_corrections[0],
            f_in=self.freq.freq,
            resistance_m=resistance_m,
        )

    def get_s11_correction_model(
        self, nterms=None, load_name=None, resistance_m=50.166
    ):
        kind_nterms = {
            "ambient": 37,
            "hot_load": 37,
            "open": 105,
            "short": 105,
            "antsim": 55,
            "lna": 37,
        }
        nterms = nterms or (kind_nterms[load_name] if load_name is not None else 105)

        corr = self.get_s11_correction(resistance_m)

        def getmodel(mag):
            # Returns a callable function that will evaluate a model onto a set of
            # un-normalised frequencies.
            if mag:
                d = np.abs(corr)
            else:
                d = np.unwrap(np.angle(corr))

            fit = mdl.fit_polynomial_fourier(
                "fourier", self.freq.freq_recentred, d, nterms
            )[0]
            return lambda x: mdl.model_evaluate("fourier", fit, x)

        mag = getmodel(True)
        ang = getmodel(False)

        def model(f):
            ff = self.freq.normalize(f)
            return mag(ff) * (np.cos(ang(ff)) + 1j * np.sin(ang(ff)))

        return model

    def plot_residuals(self, model, corr=None):
        fig, ax = plt.subplots(4, 1, sharex=True, facecolor="w")
        for axx in ax:
            axx.xaxis.set_ticks(
                [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], []
            )
            axx.grid(True)
            axx.set_xlabel("Frequency [MHz]")

        if corr is None:
            corr = self.get_s11_correction()

        ax[0].plot(self.freq.freq, 20 * np.log10(np.abs(model(self.freq.freq))))
        ax[0].set_ylabel("S11(mag)")

        ax[1].plot(self.freq.freq, np.abs(model(self.freq.freq)) - np.abs(corr), "g")
        ax[1].set_ylabel("Delta S11(mag)")

        ax[2].plot(
            self.freq.freq, np.unwrap(np.angle(model(self.freq.freq))) * 180 / np.pi
        )
        ax[2].set_ylabel(" S11(Ang)")

        ax[3].plot(
            self.freq.freq,
            np.unwrap(np.angle(model(self.freq.freq))) - np.unwrap(np.angle(corr)),
            "g",
        )
        ax[3].set_ylabel("Delta S11(Ang)")

        return fig


class LNA(SwitchCorrection):
    def __init__(
        self, base_path, f_low=None, f_high=None, run_num=0, resistance_f=50.009
    ):
        super().__init__(
            base_path=base_path, f_low=f_low, f_high=f_high, run_num=run_num
        )

    @cached_property
    def external(self):
        """
        VNA S11 measurements for the load.
        """
        return VNA(
            os.path.join(self.base_path, "ReceiverReading{:.2d}".format(self.run_num)),
            f_low=self.freq.f_low,
            f_high=self.freq.f_high,
            run_num=self.run_num,
        )

    def switch_corrections(self, resistance_f=50.009):
        # Models of standards
        oa, sa, la = rc.agilent_85033E(self.freq.freq, resistance_f, m=1)

        # Correction at switch
        return rc.de_embed(
            oa, sa, la, self.open.s11, self.short.s11, self.match.s11, self.external.s11
        )[0]

    @lru_cache()
    def get_s11_correction(self, resistance_f=50.009):
        """
        Determine the correction required for the S11 due to the switch.
        """
        return self.switch_corrections(resistance_f)

    def get_s11_correction_model(self):
        return super().get_s11_correction_model(
            load_name="lna", resistance_m=self.resistance_f
        )


class LoadSpectrum:
    _kinds = {
        "ambient": "Ambient",
        "hot_load": "HotLoad",
        "open": "LongCableOpen",
        "short": "LongCableShort",
        "antsim": "AntSim",
    }

    def __init__(
        self,
        load_name,
        path,
        switch_correction=None,
        f_low=None,
        f_high=None,
        run_num=0,
        percent=5,
    ):
        self.load_name = load_name
        self.path = path
        self.path_s11 = os.path.join(path, "S11", self._kinds[self.load_name])
        self.path_res = os.path.join(path, "Resistance")
        self.path_spec = os.path.join(path, "Spectra", "mat_files")

        if switch_correction is None:
            self.switch_correction = SwitchCorrection(
                self.path_s11, f_low=f_low, f_high=f_high, run_num=run_num
            )
        else:
            self.switch_correction = switch_correction

        self.percent = percent
        self.freq = EdgesFrequencyRange(f_low=f_low, f_high=f_high)

    @cached_property
    def averaged_spectrum(self):
        """
        Normalised power, Q_p = (P_source - P_load)/(P_noise - P_load)
        """
        spec_files = glob.glob(
            os.path.join(self.path_spec, self._kinds[self.load_name] + "*.mat")
        )
        if not spec_files:
            raise FileNotFoundError("No .mat files found for {}".format(self.load_name))
        return self._read_power(spec_files)[self.freq.mask]

    def _read_power(self, spectrum_files):
        """
        Read a MAT file to get the normalised raw power, i.e.

        Q_p = (P_source - P_load)/(P_noise - P_load)

        Returns
        -------
        ndarray : Q_p as a function of frequency.
        """
        for i, fl in enumerate(spectrum_files):
            tai = rcf.load_level1_MAT(fl)
            if i == 0:
                ta = tai
            else:
                ta = np.concatenate((ta, tai), axis=1)

        index_start_spectra = int((self.percent / 100) * len(ta[0, :]))
        ta_sel = ta[:, index_start_spectra:]
        return np.mean(ta_sel, axis=1)

    def _read_thermistor_temp(self, resistance_file):
        if type(resistance_file) == str:
            resistance_file = [resistance_file]

        if len(resistance_file) == 0:
            raise ValueError("Empty list of resistance files")

        resistance = np.genfromtxt(resistance_file[0])
        for fl in resistance_file[1:]:
            resistance = np.concatenate((resistance, np.genfromtxt(fl)), axis=0)

        temp_spectrum = rcf.temperature_thermistor(resistance)
        return temp_spectrum[(self.percent / 100) * len(temp_spectrum)]

    @cached_property
    def thermistor_temp(self):
        res_files = glob.glob(
            os.path.join(self.path_res, self._kinds[self.load_name] + "*.txt")
        )

        if not res_files:
            raise FileNotFoundError("No .txt files found for {}".format(self.load_name))

        return self._read_thermistor_temp(res_files)[self.freq.mask]

    @cached_property
    def temp_ave(self):
        """Average thermistor temperature"""
        return np.mean(self.thermistor_temp)

    def write(self, direc):
        with h5py.File(
            os.path.join(direc, self.load_name + "_averaged_spectrum.h5")
        ) as fl:
            fl["attrs"]["load_name"] = self.load_name
            fl["freq"] = self.freq.freq
            fl["averaged_raw_spectrum"] = self.averaged_spectrum
            fl["temperature"] = self.thermistor_temp

    def plot(self, temp=True, fig=None, ax=None, **kwargs):
        if fig is None:
            fig, ax = plt.subplots(1, 1, facecolor=kwargs.get("facecolor", "white"))

        if temp:
            ax.plot(self.freq.freq, self.thermistor_temp)
            ax.set_ylabel("Temperature [K]")
        else:
            ax.plot(self.freq.freq, self.averaged_spectrum)
            ax.set_ylabel("Measured $Q_P$")

        ax.grid(True)
        ax.set_xlabel("Frequency [MHz]")


class HotLoadCorrection:
    _kinds = {"s11": 1, "s12": 3, "s22": 5}

    def __init__(self, path, f_low=None, f_high=None):
        self.path = path
        data = np.genfromtxt(
            os.path.join(path, "semi_rigid_s_parameters_WITH_HEADER.txt")
        )

        f = data[:, 0]
        self.freq = FrequencyRange(f, f_low, f_high)
        self.data = data[self.freq.mask] + 1j * data[self.freq.mask]

    def _get_model_part(self, kind, mag=True):
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
        d = self.data[:, self._kinds[kind]] + 1j * self.data[:, self._kinds[kind] + 1]
        if mag:
            d = np.abs(d)
        else:
            d = np.unwrap(np.angle(d))

        mag = mdl.fit_polynomial_fourier("polynomial", self.freq.freq_recentred, d, 21)

        def out(f):
            ff = self.freq.normalize(f)
            return mdl.model_evaluate("polynomial", mag[0], ff)

        return out

    def _get_model_kind(self, kind):
        mag = self._get_model_part(kind)
        ang = self._get_model_part(kind, False)

        def out(f):
            a = ang(f)
            return mag(f) * (np.cos(a) + 1j * np.sin(a))

        return out

    @cached_property
    def s11_model(self):
        return self._get_model_kind("s11")

    @cached_property
    def s12_model(self):
        return self._get_model_kind("s12")

    @cached_property
    def s22_model(self):
        return self._get_model_kind("s22")


class CalibrationObservation:
    _sources = ["ambient", "hot_load", "open", "short"]

    def __init__(
        self,
        path,
        f_low=None,
        f_high=None,
        run_num=0,
        resistance_f=50.009,
        resistance_m=50.166,
        percent=5,
        cterms=5,
        wterms=7,
    ):
        self.path = path

        self.ambient = LoadSpectrum(
            "ambient",
            path,
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            percent=percent,
        )
        self.hot_load = LoadSpectrum(
            "hot_load",
            path,
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            percent=percent,
        )
        self.open = LoadSpectrum(
            "open", path, f_low=f_low, f_high=f_high, run_num=run_num, percent=percent
        )
        self.short = LoadSpectrum(
            "short", path, f_low=f_low, f_high=f_high, run_num=run_num, percent=percent
        )

        self.hot_load_correction = HotLoadCorrection(path, f_low, f_high)

        self.lna_s11 = LNA(path, f_low=f_low, f_high=f_high)
        self.resistance_m = resistance_m
        self.resistance_f = resistance_f

        self.cterms = cterms
        self.wterms = wterms

        # Expose a Frequency object
        self.freq = self.ambient.freq

    @cached_property
    def hot_load_corrected_ave_temp(self):
        hot_load_correction = self.switch_correction.get_s11_correction(
            resistance_m=self.resistance_m
        )
        rht = rc.gamma_de_embed(
            self.hot_load_correction.s11_model,
            self.switch_correction.s12_model,
            self.switch_correction.s22_model,
            hot_load_correction,
        )

        # inverting the direction of the s-parameters,
        # since the port labels have to be inverted to match those of Pozar eqn 10.25
        s11_sr_rev = self.hot_load_correction.s11_model

        # absolute value of S_21
        abs_s21 = np.sqrt(np.abs(self.hot_load_correction.s12_model))

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
        return G * self.hot_load.temp_ave + (1 - G) * self.ambient.temp_ave

    @cached_property
    def s11_correction_models(self):
        return {
            k: getattr(self, k).switch_correction.get_s11_correction(self.resistance_m)
            for k in self._sources
        }

    @cached_property
    def calibration_coefficients(self):
        scale, off, Tu, TC, TS = rcf.calibration_quantities(
            self.freq.f_low,
            self.freq.f_high,
            self.freq.freq,
            self.ambient.averaged_spectrum,
            self.hot_load.averaged_spectrum,
            self.open.averaged_spectrum,
            self.short.averaged_spectrum,
            self.lna_s11.get_s11_correction_model(self.resistance_f),
            self.s11_correction_models["ambient"],
            self.s11_correction_models["hot_load"],
            self.s11_correction_models["open"],
            self.s11_correction_models["short"],
            self.ambient.temp_ave,
            self.hot_load_corrected_ave_temp,
            self.open.temp_ave,
            self.short.temp_ave,
            self.cterms,
            self.wterms,
        )
        return scale, off, Tu, TC, TS

    @lru_cache()
    def calibrate(self, load):
        scale, off, Tu, TC, TS = self.calibration_coefficients

        if type(load) == str:
            try:
                load = getattr(self, load)
            except AttributeError:
                raise AttributeError(
                    "load must be a LoadSpectrum or a string (one of {ambient,hot_load,open,short}"
                )
        return rcf.calibrated_antenna_temperature(
            load.averaged_spectrum,
            load.switch_correction.get_s11_correction(self.resistance_m),
            self.lna_s11.get_s11_correction_model(self.resistance_f),
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
