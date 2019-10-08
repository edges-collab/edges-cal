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
from . import io
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
        return np.logical_and(
            self.freq_full >= self._f_low, self.freq_full <= self._f_high
        )

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
    def __init__(self, fname, f_low=None, f_high=None, run_num=None, switchval=None):
        self.fname = fname
        if run_num is None:
            self.run_num = int(self.fname.split(".")[-2][-2:])

        self.load_name = os.path.splitext(os.path.basename(self.fname))[0][:-2]

        spec, f = self._read()

        self.freq = FrequencyRange(f / 1e6, f_low, f_high)
        self.s11 = spec[self.freq.mask]
        self._switchval = switchval

    def _read(self):
        return rc.s1p_read(self.fname)

    @cached_property
    def switchval(self):
        if self._switchval is not None:
            return self._switchval * np.ones_like(self.freq.freq)
        else:
            return None


class SwitchCorrection:
    def __init__(
        self,
        load_name,
        base_path,
        correction_path,
        f_low=None,
        f_high=None,
        run_num=1,
        resistance=50.166,
    ):
        self.load_name = load_name
        self.base_path = base_path
        self.correction_path = correction_path
        self.run_num = run_num
        self.resistance = resistance

        self.open = VNA(
            os.path.join(base_path, "Open{:02d}.s1p".format(run_num)),
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            switchval=1,
        )
        self.short = VNA(
            os.path.join(base_path, "Short{:02d}.s1p".format(run_num)),
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            switchval=-1,
        )
        self.match = VNA(
            os.path.join(base_path, "Match{:02d}.s1p".format(run_num)),
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            switchval=0,
        )

        if not (
            np.all(self.open.freq.freq == self.short.freq.freq)
            and np.all(self.open.freq.freq == self.match.freq.freq)
        ):
            raise ValueError("S11 files do not match")

        # Expose one of the frequency objects
        self.freq = self.open.freq

    @cached_property
    def external(self):
        """
        VNA S11 measurements for the load.
        """
        return VNA(
            os.path.join(self.base_path, "External{:02d}.s1p".format(self.run_num)),
            f_low=self.freq.freq.min(),
            f_high=self.freq.freq.max(),
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

    @cached_property
    def s11_correction(self):
        """
        Determine the correction required for the S11 due to the switch.
        """
        return s11.low_band_switch_correction_june_2016(
            self.correction_path,
            self.switch_corrections[0],
            f_in=self.freq.freq,
            resistance_m=self.resistance,
        )

    @lru_cache()
    def get_s11_correction_model(self, nterms=None):
        kind_nterms = {
            "ambient": 37,
            "hot_load": 37,
            "open": 105,
            "short": 105,
            "antsim": 55,
            "lna": 37,
        }
        nterms = nterms or kind_nterms[self.load_name]

        def getmodel(mag):
            # Returns a callable function that will evaluate a model onto a set of
            # un-normalised frequencies.
            if mag:
                d = np.abs(self.s11_correction)
            else:
                d = np.unwrap(np.angle(self.s11_correction))

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

    def plot_residuals(self):
        fig, ax = plt.subplots(
            4, 1, sharex=True, gridspec_kw={"hspace": 0.05}, facecolor="w"
        )
        for axx in ax:
            axx.xaxis.set_ticks(
                [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], []
            )
            axx.grid(True)
        ax[-1].set_xlabel("Frequency [MHz]")

        corr = self.s11_correction
        model = self.get_s11_correction_model()
        model = model(self.freq.freq)

        ax[0].plot(self.freq.freq, 20 * np.log10(np.abs(model)))
        ax[0].set_ylabel(r"$|S_{11}|$")

        ax[1].plot(self.freq.freq, np.abs(model) - np.abs(corr), "g")
        ax[1].set_ylabel(r"\Delta $S_{11}$")

        ax[2].plot(self.freq.freq, np.unwrap(np.angle(model)) * 180 / np.pi)
        ax[2].set_ylabel(r"$\angle S_{11}$")

        ax[3].plot(
            self.freq.freq, np.unwrap(np.angle(model)) - np.unwrap(np.angle(corr)), "g"
        )
        ax[3].set_ylabel(r"$\Delta \angle S_{11}$")

        fig.suptitle(f"{self.load_name} Reflection Coefficient Models", fontsize=14)
        return fig


class LNA(SwitchCorrection):
    def __init__(
        self,
        base_path,
        correction_path,
        f_low=None,
        f_high=None,
        run_num=1,
        resistance=50.009,
    ):
        super().__init__(
            load_name="lna",
            base_path=os.path.join(base_path, "ReceiverReading{:02d}".format(run_num)),
            correction_path=correction_path,
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            resistance=resistance,
        )

    @cached_property
    def external(self):
        """
        VNA S11 measurements for the load.
        """
        return VNA(
            os.path.join(
                self.base_path, "ReceiverReading{:02d}.s1p".format(self.run_num)
            ),
            f_low=self.freq.freq.min(),
            f_high=self.freq.freq.max(),
            run_num=self.run_num,
        )

    @cached_property
    def switch_corrections(self):
        # Models of standards
        oa, sa, la = rc.agilent_85033E(self.freq.freq, self.resistance, m=1)

        # Correction at switch
        return rc.de_embed(
            oa, sa, la, self.open.s11, self.short.s11, self.match.s11, self.external.s11
        )[0]

    @cached_property
    def s11_correction(self):
        """
        Determine the correction required for the S11 due to the switch.
        """
        return self.switch_corrections

    @lru_cache()
    def get_s11_correction_model(self, nterms=None):
        return super().get_s11_correction_model(nterms=nterms)


class LoadSpectrum:
    _kinds = {
        "ambient": "Ambient",
        "hot_load": "HotLoad",
        "open": "LongCableOpen",
        "short": "LongCableShorted",
        "antsim": "AntSim4",
    }
    _file_prefixes = {
        "ambient": "Ambient",
        "hot_load": "HotLoad",
        "open": "LongCableOpen",
        "short": "LongCableShort",
        "antsim": "AntSim4",
    }

    def __init__(
        self,
        load_name,
        path,
        switch_correction=None,
        correction_path=None,
        f_low=None,
        f_high=None,
        run_num=1,
        percent=5,
        resistance=50.166,
    ):
        self.load_name = load_name
        self.path = path
        self.path_s11 = os.path.join(path, "S11", self._kinds[self.load_name])
        self.path_res = os.path.join(path, "Resistance")
        self.path_spec = os.path.join(path, "Spectra", "mat_files")

        if switch_correction is None:
            self.switch_correction = SwitchCorrection(
                self.load_name,
                self.path_s11,
                correction_path,
                f_low=f_low,
                f_high=f_high,
                run_num=run_num,
                resistance=resistance,
            )
        else:
            self.switch_correction = switch_correction

        self.percent = percent
        self.freq = EdgesFrequencyRange(f_low=f_low, f_high=f_high)

    @cached_property
    def averaged_spectrum(self):
        """
        Normalised uncalibrated temperature,
        T* = T_noise * (P_source - P_load)/(P_noise - P_load) + T_load
        """
        spec_files = glob.glob(
            os.path.join(self.path_spec, self._file_prefixes[self.load_name] + "*.mat")
        )
        if not spec_files:
            raise FileNotFoundError(
                "No .mat files found for {} in {}".format(
                    self.load_name, self.path_spec
                )
            )
        return self._read_power(spec_files)[self.freq.mask]

    def _read_power(self, spectrum_files):
        """
        Read a MAT file to get the corrected raw temperature, i.e.

        T* = (P_source - P_load)/(P_noise - P_load)*T_noise + T_load

        Returns
        -------
        ndarray : T* as a function of frequency.
        """
        for i, fl in enumerate(spectrum_files):
            tai = io.load_level1_MAT(fl)
            if i == 0:
                ta = tai
            else:
                ta = np.concatenate((ta, tai), axis=1)

        index_start_spectra = int((self.percent / 100) * len(ta[0, :]))
        ta_sel = ta[:, index_start_spectra:]
        return np.mean(ta_sel, axis=1)

    def _read_thermistor_temp(self, resistance_file):
        """
        Read a resistance file and return the associated thermistor temperature in K.
        """
        if type(resistance_file) == str:
            resistance_file = [resistance_file]

        if len(resistance_file) == 0:
            raise ValueError("Empty list of resistance files")

        resistance = np.genfromtxt(resistance_file[0])
        for fl in resistance_file[1:]:
            resistance = np.concatenate((resistance, np.genfromtxt(fl)), axis=0)

        temp_spectrum = rcf.temperature_thermistor(resistance)
        return temp_spectrum[int((self.percent / 100) * len(temp_spectrum)) :]

    @cached_property
    def thermistor_temp(self):
        """
        Temperature of the known noise source.
        """
        res_files = glob.glob(
            os.path.join(self.path_res, self._kinds[self.load_name] + "*.txt")
        )

        if not res_files:
            raise FileNotFoundError("No .txt files found for {}".format(self.load_name))

        temp = self._read_thermistor_temp(res_files)

        # Can't just use mask, because the thermistor spectrum has different resolution.
        return temp[int((self.percent / 100) * len(temp)) :]

    @cached_property
    def temp_ave(self):
        """Average thermistor temperature (over time and frequency)"""
        return np.mean(self.thermistor_temp)

    def write(self, direc):
        with h5py.File(
            os.path.join(direc, self.load_name + "_averaged_spectrum.h5")
        ) as fl:
            fl["attrs"]["load_name"] = self.load_name
            fl["freq"] = self.freq.freq
            fl["averaged_raw_spectrum"] = self.averaged_spectrum
            fl["temperature"] = self.thermistor_temp

    def plot(
        self, thermistor=False, fig=None, ax=None, xlabel=True, ylabel=True, **kwargs
    ):
        if fig is None:
            fig, ax = plt.subplots(1, 1, facecolor=kwargs.get("facecolor", "white"))

        if thermistor:
            ax.plot(self.freq.freq, self.thermistor_temp)
            if ylabel:
                ax.set_ylabel("Temperature [K]")
        else:
            ax.plot(self.freq.freq, self.averaged_spectrum)
            if ylabel:
                ax.set_ylabel("$T^*$ [K]")

        ax.grid(True)
        if xlabel:
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
        correction_path,
        f_low=None,
        f_high=None,
        run_num=1,
        resistance_f=50.009,
        resistance_m=50.166,
        percent=5,
        cterms=5,
        wterms=7,
    ):
        self.path = path

        for source in self._sources:
            setattr(
                self,
                source,
                LoadSpectrum(
                    source,
                    path,
                    correction_path=correction_path,
                    f_low=f_low,
                    f_high=f_high,
                    run_num=run_num,
                    percent=percent,
                    resistance=resistance_m,
                ),
            )

        self.hot_load_correction = HotLoadCorrection(
            os.path.join(path, "S11"), f_low, f_high
        )

        self.lna_s11 = LNA(
            os.path.join(path, "S11"),
            correction_path=correction_path,
            f_low=f_low,
            f_high=f_high,
            run_num=run_num,
            resistance=resistance_f,
        )

        self.cterms = cterms
        self.wterms = wterms

        # Expose a Frequency object
        self.freq = self.ambient.freq

    def plot_raw_spectra(self, fig=None, ax=None):
        """
        Plot raw uncalibrated spectra for all calibrator sources.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                len(self._sources), 1, sharex=True, gridspec_kw={"hspace": 0.05}
            )

        for i, source in enumerate(self._sources):
            src = getattr(self, source)
            src.plot(fig=fig, ax=ax[i], xlabel=(i == (len(self._sources) - 1)))
            ax[i].set_title(source)

        return fig

    def plot_s11_models(self):
        figs = {}
        for source in self._sources:
            src = getattr(self, source)
            figs[source] = src.switch_correction.plot_residuals()
        return figs

    @cached_property
    def hot_load_corrected_ave_temp(self):
        hot_load_correction = self.hot_load.switch_correction.get_s11_correction_model()
        hot_load_correction = hot_load_correction(self.freq.freq)

        rht = rc.gamma_de_embed(
            self.hot_load_correction.s11_model(self.freq.freq),
            self.hot_load_correction.s12_model(self.freq.freq),
            self.hot_load_correction.s22_model(self.freq.freq),
            hot_load_correction,
        )

        # inverting the direction of the s-parameters,
        # since the port labels have to be inverted to match those of Pozar eqn 10.25
        s11_sr_rev = self.hot_load_correction.s11_model(self.freq.freq)

        # absolute value of S_21
        abs_s21 = np.sqrt(np.abs(self.hot_load_correction.s12_model(self.freq.freq)))

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
            k: getattr(self, k).switch_correction.get_s11_correction_model()(
                self.freq.freq
            )
            for k in self._sources
        }

    @cached_property
    def calibration_coefficients(self):
        """
        The calibration polynomials, C1, C2, Tunc, Tcos, Tsin, evaluated at `freq.freq`.
        """
        scale, off, Tu, TC, TS = rcf.get_calibration_quantities_iterative(
            self.freq.freq_recentred,
            T_raw={
                k: getattr(getattr(self, k), "averaged_spectrum") for k in self._sources
            },
            gamma_rec=self.lna_s11.get_s11_correction_model()(self.freq.freq),
            gamma_ant=self.s11_correction_models,
            T_ant={
                k: (
                    self.hot_load_corrected_ave_temp
                    if k == "hot_load"
                    else getattr(self, k).temp_ave
                )
                for k in self._sources
            },
            cterms=self.cterms,
            wterms=self.wterms,
        )
        return scale, off, Tu, TC, TS

    def calibrate(self, load):
        """
        Calibrate the temperature of a given load.

        Parameters
        ----------
        load : :class:`LoadSpectrum` instance
            The load to calibrate.

        Returns
        -------
        array : calibrated antenna temperature in K, len(f).
        """
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
            load.switch_correction.get_s11_correction_model()(self.freq.freq),
            self.lna_s11.get_s11_correction_model()(self.freq.freq),
            scale,
            off,
            Tu,
            TC,
            TS,
            Tamb_internal=300,
        )

    def decalibrate(self, temp, s11):
        """
        Decalibrate a temperature spectrum, yielding Q_p

        Parameters
        ----------
        temp : array_like
            A temperature spectrum, with the same length as `freq.freq`.

        Returns
        -------
        array_like : Q_p, the normalised raw power.
        """
        scale, off, Tu, TC, TS = self.calibration_coefficients

        Q_p = rcf.power_ratio(
            freqs=self.freq.freq,
            temp_ant=temp,
            gamma_ant=s11,
            gamma_rec=self.lna_s11.get_s11_correction_model()(self.freq.freq),
            scale=scale,
            offset=off,
            temp_unc=Tu,
            temp_cos=TC,
            temp_sin=TS,
            temp_noise_source=400,
            temp_load=300,
            ref_freq=75.0,
        )

        return 400 * Q_p + 300

    def plot_calibrated_temp(
        self, load, bins=64, fig=None, ax=None, xlabel=True, ylabel=True
    ):
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="w")

        lim = len(self.freq.freq)

        # binning
        temp_calibrated = self.calibrate(load)
        fact = lim / bins
        f_new = np.linspace(self.freq.freq.min(), self.freq.freq.max(), bins)

        # TODO: this would probably be better using a convolution kernel
        freq_ave_cal = np.zeros(bins)
        for i in range(bins):
            freq_ave_cal[i] = np.average(
                temp_calibrated[int(i * fact) : int((i + 1) * fact)]
            )

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        ax.plot(
            f_new, freq_ave_cal, label=f"Calibrated {load.load_name} [RMS = {rms:.3f}]"
        )

        if load.load_name != "hot_load":
            ax.axhline(load.temp_ave, color="C2", label="Average thermistor temp")
        else:
            ax.plot(
                self.freq.freq,
                self.hot_load_corrected_ave_temp,
                color="C2",
                label="Average thermistor temp",
            )

        ax.set_ylim([np.min(freq_ave_cal), np.max(freq_ave_cal)])
        if xlabel:
            ax.set_xlabel("Frequency [MHz]")

        if ylabel:
            ax.set_ylabel("Temperature [K]")

        plt.ticklabel_format(useOffset=False)
        ax.grid()
        ax.legend()

        return plt.gcf()

    def plot_calibrated_temps(self, bins=64):
        fig, ax = plt.subplots(
            len(self._sources),
            1,
            sharex=True,
            gridspec_kw={"hspace": 0.05},
            figsize=(10, 12),
        )

        for i, source in enumerate(self._sources):
            self.plot_calibrated_temp(
                getattr(self, source),
                bins=bins,
                fig=fig,
                ax=ax[i],
                xlabel=i == (len(self._sources) - 1),
            )

        fig.suptitle("Calibrated Temperatures for Calibration Sources", fontsize=15)
        return fig

    def write_coefficients(self, direc="."):
        scale, off, Tu, TC, TS = self.calibration_coefficients
        np.savetxt(
            os.path.join(
                direc,
                "All_cal-params_{}_{}_{}-{}_25C_s11alan.txt".format(
                    self.freq.freq.min(), self.freq.freq.max(), self.cterms, self.wterms
                ),
            ),
            [self.freq.freq, scale, off, Tu, TC, TS],
        )

    def plot_coefficients(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                5, 1, facecolor="w", gridspec_kw={"hspace": 0.05}, figsize=(10, 9)
            )

        labels = [
            "Scale ($C_1$)",
            "Offset ($C_2$) [K]",
            r"$T_{\rm unc}$ [K]",
            r"$T_{\rm cos}$ [K]",
            r"$T_{\rm sin}$ [K]",
        ]
        for i, (soln, label) in enumerate(zip(self.calibration_coefficients, labels)):
            ax[i].plot(self.freq.freq, soln)
            ax[i].set_ylabel(label, fontsize=13)
            ax[i].grid()
            plt.ticklabel_format(useOffset=False)

            if i == 4:
                ax[i].set_xlabel("Frequency [MHz]", fontsize=13)

        fig.suptitle("Calibration Parameters", fontsize=15)
        return fig

    def write(self, direc="."):
        lnas11 = self.lna_s11.get_s11_correction_model()(self.freq.freq)
        np.savetxt(os.path.join(direc, "fit_s11_LNA_mag.txt"), np.abs(lnas11))
        np.savetxt(
            os.path.join(direc, "fit_s11_LNA_ang.txt"), np.unwrap(np.angle(lnas11))
        )

        sources = tuple(self._sources)

        for source in sources:
            src = getattr(self, source)
            for part, fnc in zip(
                ["mag", "ang"], [np.abs, lambda x: np.unwrap(np.angle(x))]
            ):
                out = fnc(
                    src.switch_correction.get_s11_correction_model()(self.freq.freq)
                )
                key = "fit_s11_{}_{}".format(source, part)
                np.savetxt(os.path.join(direc, key + ".txt"), out)

            key = "{}_thermistor_temp".format(source)
            np.savetxt(os.path.join(direc, key + ".txt"), src.thermistor_temp)
