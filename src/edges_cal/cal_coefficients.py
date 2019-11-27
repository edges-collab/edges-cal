# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2019
Original author: Nivedita Mahesh
Edited by: David Lewis, Steven Murray

This is the main module of `cal_coefficients`. It contains wrappers around lower-level
functions in other modules.
"""

import glob
import os
import warnings
from functools import lru_cache

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from cached_property import cached_property

from . import S11_correction as s11
from . import io
from . import modelling as mdl
from . import receiver_calibration_func as rcf
from . import reflection_coefficient as rc
from . import xrfi


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
    f_high : float, optional
        A minimum frequency to keep in the array. Default is min(f).
    """

    def __init__(self, f, f_low=None, f_high=None):
        self.freq_full = f
        self._f_high = f_high or f.max()
        self._f_low = f_low or f.min()

        if self._f_low >= self._f_high:
            raise ValueError("Cannot create frequency range: f_low >= f_high")

    @cached_property
    def min(self):
        """Minimum frequency in the array"""
        return self.freq.min()

    @cached_property
    def max(self):
        """Maximum frequency in the array"""
        return self.freq.max()

    @cached_property
    def mask(self):
        """Mask used to take input frequencies to output frequencies"""
        return np.logical_and(
            self.freq_full >= self._f_low, self.freq_full <= self._f_high
        )

    @cached_property
    def freq(self):
        """The frequency array"""
        return self.freq_full[self.mask]

    @cached_property
    def range(self):
        """Total range (float) of the frequencies"""
        return self.max - self.min

    @cached_property
    def center(self):
        """The center of the frequency array"""
        return self.freq.min() + self.range / 2.0

    @cached_property
    def freq_recentred(self):
        """The frequency array re-centred so that it extends from -1 to 1"""
        return self.normalize(self.freq)

    def normalize(self, f):
        """
        Normalise a set of frequencies such that -1 aligns with `min` and
        +1 aligns with `max`.

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

    @staticmethod
    def get_edges_freqs(nchannels=16384 * 2, max_freq=200.0):
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
    """
    An object representing the measurements of a VNA.

    The measurements are read in via a .s1p file

    Parameters
    ----------
    fname : str
        The path to a valid .s1p file containing VNA measurements.
    f_low : float, optional
        The minimum frequency to keep.
    f_high : float, optional
        The maximum frequency to keep.
    run_num : int, optional
        An integer identifier for the measurement. In general, it is assumed that it forms
        the last two digits of the filename before its extension.
    switchval : int, optional
        The standard value of the switch for the component.
    """

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
        """Read the s1p file"""
        return rc.s1p_read(self.fname)

    @cached_property
    def switchval(self):
        """The standard value of the switch for the component"""
        if self._switchval is not None:
            return self._switchval * np.ones_like(self.freq.freq)
        else:
            return None


class SwitchCorrection:
    def __init__(
        self,
        load_name,
        base_path,
        correction_path=None,
        f_low=None,
        f_high=None,
        run_num=1,
        resistance=50.166,
    ):
        """
        A class representing relevant switch corrections for a load.

        Parameters
        ----------
        load_name : str
            The name of the load. Affects default values for the S11 correction modeling.
        base_path : str
            The path to the directory in which the s1p files reside for the load.
            Three files must exist there -- open, short and match.
        correction_path : str, optional
            The path to S11 switch correction measurements. If not given, defaults to
            `base_path`.
        f_low : float, optional
            Minimum frequency to use. Default is all frequencies.
        f_high : float, optional
            Maximum frequency to use. Default is all frequencies.
        run_num : int, optional
            If multiple VNA measurements are present, they should be in files named
            eg. Open01.s1p, Open02.s1p.... `run_num` specifies which file to read in.
        resistance : float, optional
            The resistance of the switch (in Ohms).
        """
        self.load_name = load_name
        self.base_path = base_path
        self.correction_path = correction_path or base_path
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
        """The corrections at the switch"""
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
        The correction required for the S11 due to the switch.
        """
        return s11.low_band_switch_correction_june_2016(
            self.correction_path,
            self.switch_corrections[0],
            f_in=self.freq.freq,
            resistance_m=self.resistance,
        )

    @lru_cache()
    def get_s11_correction_model(self, nterms=None):
        """
        Generate a callable model for the S11 correction.

        This should closely match :method:`s11_correction`.

        Parameters
        ----------
        nterms : int, optional
            Number of terms used in the fourier-based model. Not necessary if `load_name`
            is specified in the class.

        Returns
        -------
        callable :
            A function of one argument, f, which should be a frequency in the same units
            as `self.freq.freq`.
        """
        kind_nterms = {
            "ambient": 37,
            "hot_load": 37,
            "open": 105,
            "short": 105,
            "antsim": 55,
            "lna": 37,
        }
        nterms = nterms or kind_nterms[self.load_name]

        if not isinstance(nterms, int):
            raise ValueError(
                "nterms must be an integer or the load_name must be in {}".format(
                    kind_nterms.keys()
                )
            )

        def get_model(mag):
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

        mag = get_model(True)
        ang = get_model(False)

        def model(f):
            ff = self.freq.normalize(f)
            return mag(ff) * (np.cos(ang(ff)) + 1j * np.sin(ang(ff)))

        return model

    def plot_residuals(self, nterms=None):
        """
        Make a plot of the residuals of the S11 model (gotten via
        :func:`get_s11_correction_model`) and the correction data.

        Parameters
        ----------
        nterms : int, optional
            Number of terms used in the fourier-based model. Not necessary if `load_name`
            is specified in the class.

        Returns
        -------
        fig :
            Matplotlib Figure handle.
        """
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
    def __init__(self, base_path, resistance=50.009, **kwargs):
        """
        A special case of :class:`SwitchCorrection` for the LNA.
        """
        super().__init__(
            load_name="lna",
            base_path=os.path.join(
                base_path, "ReceiverReading{:02d}".format(kwargs.get("run_num", 1))
            ),
            resistance=resistance,
            **kwargs,
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
        return self.switch_corrections


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
        s11_model_nterms=None,
        rfi_removal="1D2D",
        rfi_kernel_width_time=16,
        rfi_kernel_width_freq=16,
        rfi_threshold=6,
    ):
        """
        A class representing a measured spectrum from some Load.

        Parameters
        ----------
        load_name : str
            Name of the load
        path : str
            Path to the directory containing all relevant measurements. It is assumed
            that in this directory is an `S11`, `Resistance` and `Spectra` directory.
        switch_correction : :class:`SwitchCorrection`, optional
            A `SwitchCorrection` for this particular load. If not given, will be
            constructed automatically.
        correction_path : str, optional
            A path to switch corrections, if different from base path.
        f_low : float, optional
            Minimum frequency to keep.
        f_high : float, optional
            Maximum frequency to keep.
        run_num : int, optional
            Identifier for the measurement files to read.
        percent : float, optional
            Must be between 0 and 100. Number of time-samples in a file to reject
            from the start of the file.
        resistance : float, optional
            Resistance of the switch.
        s11_model_nterms : int, optional
            Number of terms to use in modelling the S11.
        rfi_removal : str, optional, {'1D', '2D', '1D2D'}
            If given, will perform median and mean-filtered xRFI over either the
            2D waterfall, or integrated 1D spectrum. The latter is usually reasonable
            for calibration sources, while the former is good for field data. "1D2D"
            is a hybrid approach in which the variance per-frequency is determined
            from the 2D data, but filtering occurs only over frequency.
        rfi_kernel_width_time : int, optional
            The kernel width for the detrending of data for
            RFI removal in the time dimension (only used if `rfi_removal` is "2D").
        rfi_kernel_width_freq : int, optional
            The kernel width for the detrending of data for
            RFI removal in the frequency dimension.
        rfi_threshold : float, optional
            The threshold (in equivalent standard deviation units) above which to
            flag data as RFI.
        """
        self.load_name = load_name
        self.path = path
        self.path_s11 = os.path.join(path, "S11", self._kinds[self.load_name])
        self.path_res = os.path.join(path, "Resistance")
        self.path_spec = os.path.join(path, "Spectra", "mat_files")
        self.s11_model_nterms = s11_model_nterms
        self.rfi_kernel_width_time = rfi_kernel_width_time
        self.rfi_kernel_width_freq = rfi_kernel_width_freq
        self.rfi_threshold = rfi_threshold

        assert rfi_removal in [
            "1D",
            "2D",
            "1D2D",
            False,
            None,
        ], "rfi_removal must be either '1D', '2D' or False/None"

        self.rfi_removal = rfi_removal

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
        # TODO: should also get weights!
        spec = self._ave_and_var_spec[0]

        if self.rfi_removal == "1D":
            spec = xrfi.remove_rfi(
                spec, threshold=self.rfi_threshold, Kf=self.rfi_kernel_width_freq
            )
        return spec

    @cached_property
    def variance_spectrum(self):
        """Variance of spectrum across time"""
        return self._ave_and_var_spec[1]

    @cached_property
    def _ave_and_var_spec(self):
        """Get the mean and variance of the spectrum"""
        spec = self.get_spectrum()
        mean = np.nanmean(spec, axis=1)
        var = np.nanvar(spec, axis=1)

        if self.rfi_removal == "1D2D":
            varfilt = xrfi.medfilt(var, kernel_size=self.rfi_kernel_width_freq)
            resid = mean - xrfi.medfilt(mean, kernel_size=self.rfi_kernel_width_freq)
            flags = resid > self.rfi_threshold * np.sqrt(varfilt)
            mean[flags] = np.nan
            var[flags] = np.nan

        return mean, var

    def get_spectrum(self, kind="temp"):
        spec = self._read_spectrum()

        if self.rfi_removal == "2D":
            # Need to set nans and zeros to inf so that median/mean detrending can work.
            spec[np.isnan(spec)] = np.inf

            if kind != "temp":
                spec[spec == 0] = np.inf

            spec = xrfi.remove_rfi(
                spec,
                threshold=self.rfi_threshold,
                Kt=self.rfi_kernel_width_time,
                Kf=self.rfi_kernel_width_freq,
            )
        return spec

    def _read_spectrum(self, spectrum_files=None, kind="temp"):
        """
        Read a MAT file to get the corrected raw temperature, i.e.

        T* = <(P_source - P_load)/(P_noise - P_load)>*T_noise + T_load

        Returns
        -------
        ndarray : T* as a function of frequency.
        """
        if spectrum_files is None:
            spectrum_files = glob.glob(
                os.path.join(
                    self.path_spec, self._file_prefixes[self.load_name] + "*.mat"
                )
            )

        if not spectrum_files:
            raise FileNotFoundError(
                "No .mat files found for {} in {}".format(
                    self.load_name, self.path_spec
                )
            )

        for i, fl in enumerate(spectrum_files):
            tai = io.load_level1_MAT(fl, kind=kind)
            if i == 0:
                ta = tai
            else:
                ta = np.concatenate((ta, tai), axis=1)

        index_start_spectra = int((self.percent / 100) * len(ta[0, :]))
        return ta[self.freq.mask, index_start_spectra:]

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
    def s11_model(self):
        """
        The S11 model of the load evaluated at `freq.freq`.
        """
        return self.switch_correction.get_s11_correction_model(
            nterms=self.s11_model_nterms
        )(self.freq.freq)

    @cached_property
    def temp_ave(self):
        """Average thermistor temperature (over time and frequency)"""
        return np.mean(self.thermistor_temp)

    def write(self, direc):
        """
        Write a HDF5 file containing the contents of the LoadSpectrum.

        Parameters
        ----------
        direc : str
            Directory into which to save the file. Filename will be
            <load_name>_averaged_spectrum.h5
        """
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
        """
        Make a plot of the averaged uncalibrated spectrum associated with this load.

        Parameters
        ----------
        thermistor : bool, optional
            Whether to plot the thermistor temperature on the same axis.
        fig : Figure, optional
            Optionally, pass a matplotlib figure handle which will be used to plot.
        ax : Axis, optional
            Optional, pass a matplotlib Axis handle which will be added to.
        xlabel : bool, optional
            Whether to make an x-axis label.
        ylabel : bool, optional
            Whether to plot the y-axis label
        kwargs :
            All other arguments are passed to `plt.subplots()`.
        """
        if fig is None:
            fig, ax = plt.subplots(
                1, 1, facecolor=kwargs.pop("facecolor", "white"), **kwargs
            )

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
    """
    Class representing measurements required to define the HotLoad temperature,
    from Monsalve et al. (2017), Eq. 8+9.
    """

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

    def power_gain(self, freq, hot_load):
        """Define Eq. 9 from M17"""
        hot_load_correction = hot_load.s11_model

        rht = rc.gamma_de_embed(
            self.s11_model(freq),
            self.s12_model(freq),
            self.s22_model(freq),
            hot_load_correction,
        )

        # inverting the direction of the s-parameters,
        # since the port labels have to be inverted to match those of Pozar eqn 10.25
        s11_sr_rev = self.s11_model(freq)

        # absolute value of S_21
        abs_s21 = np.sqrt(np.abs(self.s12_model(freq)))

        # available power gain
        G = (
            (abs_s21 ** 2)
            * (1 - np.abs(rht) ** 2)
            / (
                (np.abs(1 - s11_sr_rev * rht)) ** 2
                * (1 - (np.abs(hot_load_correction)) ** 2)
            )
        )
        return G


class CalibrationObservation:
    _sources = ["ambient", "hot_load", "open", "short"]

    def __init__(
        self,
        path,
        correction_path=None,
        f_low=None,
        f_high=None,
        run_num=1,
        resistance_f=50.009,
        resistance_m=50.166,
        percent=5,
        cterms=5,
        wterms=7,
        rfi_removal="1D2D",
        rfi_kernel_width_time=16,
        rfi_kernel_width_freq=16,
        rfi_threshold=6,
    ):
        """
        An composite object representing a full Calibration Observation.

        This includes spectra of all calibrators, and methods to find the calibration
        parameters. It strictly follows Monsalve et al. (2017) in its formalism.

        Parameters
        ----------
        path : str
            Path to the directory containing all relevant measurements. It is assumed
            that in this directory is an `S11`, `Resistance` and `Spectra` directory.
        correction_path : str, optional
            A path to switch corrections, if different from base path.
        f_low : float, optional
            Minimum frequency to keep.
        f_high : float, optional
            Maximum frequency to keep.
        run_num : int, optional
            Identifier for the measurement files to read.
        resistance_f : float, optional
            Female resistance (Ohms)
        resistance_m : float, optional
            Male resistance (Ohms)
        percent : float, optional
            Percent of time samples to reject from start of spectrum files.
        cterms : int, optional
            Number of terms used in the polynomial model for C1 and C2.
        wterms : int, optional
            Number of terms used in the polynomial models for Tunc, Tcos and Tsin.
        rfi_removal : str, optional, {'1D', '2D', '1D2D'}
            If given, will perform median and mean-filtered xRFI over either the
            2D waterfall, or integrated 1D spectrum. The latter is usually reasonable
            for calibration sources, while the former is good for field data. "1D2D"
            is a hybrid approach in which the variance per-frequency is determined
            from the 2D data, but filtering occurs only over frequency.
        rfi_kernel_width_time : int, optional
            The kernel width for the detrending of data for
            RFI removal in the time dimension (only used if `rfi_removal` is "2D").
        rfi_kernel_width_freq : int, optional
            The kernel width for the detrending of data for
            RFI removal in the frequency dimension.
        rfi_threshold : float, optional
            The threshold (in equivalent standard deviation units) above which to
            flag data as RFI.
        """
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
                    rfi_removal=rfi_removal,
                    rfi_kernel_width_freq=rfi_kernel_width_freq,
                    rfi_kernel_width_time=rfi_kernel_width_time,
                    rfi_threshold=rfi_threshold,
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
        """
        Plot residuals of S11 models for all sources

        Returns
        -------
        dict:
            Each entry has a key of the source name, and the value is a matplotlib figure.
        """
        figs = {}
        for source in self._sources:
            src = getattr(self, source)
            figs[source] = src.switch_correction.plot_residuals()
        return figs

    @cached_property
    def hot_load_corrected_ave_temp(self):
        """The hot-load averaged temperature, as a function of frequency"""
        G = self.hot_load_correction.power_gain(self.freq.freq, self.hot_load)

        # temperature
        return G * self.hot_load.temp_ave + (1 - G) * self.ambient.temp_ave

    @cached_property
    def s11_correction_models(self):
        """Dictionary of S11 correction models, one for each source"""
        return {k: getattr(self, k).s11_model for k in self._sources}

    @cached_property
    def _calibration_coefficients(self):
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
        return (scale, off, Tu, TC, TS)

    @cached_property
    def C1_poly(self):
        """`np.poly1d` object describing the Scaling calibration coefficient C1.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~C1` as a direct function
        on frequency.
        """
        return self._calibration_coefficients[0]

    @cached_property
    def C2_poly(self):
        """`np.poly1d` object describing the offset calibration coefficient C2.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~C2` as a direct function
        on frequency.
        """
        return self._calibration_coefficients[1]

    @cached_property
    def Tunc_poly(self):
        """`np.poly1d` object describing the uncorrelated noise-wave parameter, Tunc.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~Tunc` as a direct function
        on frequency.
        """
        return self._calibration_coefficients[2]

    @cached_property
    def Tcos_poly(self):
        """`np.poly1d` object describing the cosine noise-wave parameter, Tcos.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~Tcos` as a direct function
        on frequency.
        """
        return self._calibration_coefficients[3]

    @cached_property
    def Tsin_poly(self):
        """`np.poly1d` object describing the sine noise-wave parameter, Tsin.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~Tsin` as a direct function
        on frequency.
        """
        return self._calibration_coefficients[4]

    def C1(self, f=None):
        """
        Scaling calibration parameter.
        """
        if f is None:
            fnorm = self.freq.freq_recentred
        else:
            fnorm = self.freq.normalize(f)
        return self.C1_poly(fnorm)

    def C2(self, f=None):
        """
        Offset calibration parameter.
        """
        if f is None:
            fnorm = self.freq.freq_recentred
        else:
            fnorm = self.freq.normalize(f)
        return self.C2_poly(fnorm)

    def Tunc(self, f=None):
        """
        Uncorrelated noise-wave parameter
        """
        if f is None:
            fnorm = self.freq.freq_recentred
        else:
            fnorm = self.freq.normalize(f)
        return self.Tunc_poly(fnorm)

    def Tcos(self, f=None):
        """
        Cosine noise-wave parameter
        """
        if f is None:
            fnorm = self.freq.freq_recentred
        else:
            fnorm = self.freq.normalize(f)
        return self.Tcos_poly(fnorm)

    def Tsin(self, f=None):
        """
        Sine noise-wave parameter
        """
        if f is None:
            fnorm = self.freq.freq_recentred
        else:
            fnorm = self.freq.normalize(f)
        return self.Tsin_poly(fnorm)

    def get_linear_coefficients(self, s11):
        """
        Calibration coefficients a,b such that T = aT* + b (derived from Eq. 7)
        """
        if type(s11) == str:
            try:
                s11 = getattr(self, s11).s11_model
            except AttributeError:
                raise AttributeError(
                    "s11 must be a LoadSpectrum or a string (one of {ambient,hot_load,open,short}"
                )

        return rcf.get_linear_coefficients(
            s11,
            self.lna_s11.get_s11_correction_model()(self.freq.freq),
            self.C1(self.freq.freq),
            self.C2(self.freq.freq),
            self.Tunc(self.freq.freq),
            self.Tcos(self.freq.freq),
            self.Tsin(self.freq.freq),
            T_load=300,
        )

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
        if type(load) == str:
            load = getattr(self, load)

        a, b = self.get_linear_coefficients(load.s11_model)
        return a * load.averaged_spectrum + b

    def decalibrate(self, temp, s11, freq=None):
        """
        Decalibrate a temperature spectrum, yielding uncalibrated T*.

        Parameters
        ----------
        temp : array_like
            A temperature spectrum, with the same length as `freq.freq`.

        Returns
        -------
        array_like : T*, the normalised uncalibrated temperature.
        """
        if freq is None:
            freq = self.freq.freq

        if freq.min() < self.freq.freq.min():
            warnings.warn(
                "The minimum frequency is outside the calibrated range ({} - {} MHz)".format(
                    self.freq.freq.min(), self.freq.freq.max()
                )
            )

        if freq.min() > self.freq.freq.max():
            warnings.warn(
                "The maximum frequency is outside the calibrated range ({} - {} MHz)".format(
                    self.freq.freq.min(), self.freq.freq.max()
                )
            )

        a, b = self.get_linear_coefficients(s11)
        return (temp - b) / a

    def plot_calibrated_temp(
        self, load, bins=2, fig=None, ax=None, xlabel=True, ylabel=True
    ):
        """
        Make a plot of calibrated temperature for a given source.

        Parameters
        ----------
        load : :class:`~LoadSpectrum` instance
            Source to plot.
        bins : int, optional
            Number of bins to smooth over (std of Gaussian kernel)
        fig : Figure, optional
            Optionally provide a matplotlib figure to add to.
        ax : Axis, optional
            Optionally provide a matplotlib Axis to add to.
        xlabel : bool, optional
            Whether to write the x-axis label
        ylabel : bool, optional
            Whether to write the y-axis label

        Returns
        -------
        fig :
            The matplotlib figure that was created.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="w")

        # binning
        temp_calibrated = self.calibrate(load)
        #        f_new = np.linspace(self.freq.freq.min(), self.freq.freq.max(), bins)

        # TODO: this would probably be better using a convolution kernel
        if bins > 0:
            freq_ave_cal = convolve(temp_calibrated, Gaussian1DKernel(stddev=bins))
        else:
            freq_ave_cal = temp_calibrated

        # np.zeros(bins)
        # for i in range(bins):
        #     freq_ave_cal[i] = np.average(
        #         temp_calibrated[int(i * fact) : int((i + 1) * fact)]
        #     )

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        ax.plot(
            self.freq.freq,
            freq_ave_cal,
            label=f"Calibrated {load.load_name} [RMS = {rms:.3f}]",
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
        """
        Plot all calibrated temperatures in a single figure.

        Parameters
        ----------
        bins : int, optional
            Number of bins in the smoothed spectrum

        Returns
        -------
        fig :
            Matplotlib figure that was created.
        """
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
        """
        Save a text file with the derived calibration co-efficients.

        Parameters
        ----------
        direc : str
            Directory in which to write the file. The filename starts with `All_cal-params`
            and includes parameters of the class in the filename.
        """
        np.savetxt(
            os.path.join(
                direc,
                "All_cal-params_{}_{}_{}-{}_25C_s11alan.txt".format(
                    self.freq.freq.min(), self.freq.freq.max(), self.cterms, self.wterms
                ),
            ),
            [
                self.freq.freq,
                self.C1(),
                self.C1(),
                self.Tunc(),
                self.Tcos(),
                self.Tsin(),
            ],
        )

    def plot_coefficients(self, fig=None, ax=None):
        """
        Make a plot of the calibration models, C1, C2, Tunc, Tcos and Tsin.

        Parameters
        ----------
        fig : Figure, optional
            Optionally pass a matplotlib figure to add to.
        ax : Axis, optional
            Optionally pass a matplotlib axis to pass to. Must have 5 axes.
        """
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
        for i, (kind, label) in enumerate(
            zip(["C1", "C2", "Tunc", "Tcos", "Tsin"], labels)
        ):
            ax[i].plot(self.freq.freq, getattr(self, kind)())
            ax[i].set_ylabel(label, fontsize=13)
            ax[i].grid()
            plt.ticklabel_format(useOffset=False)

            if i == 4:
                ax[i].set_xlabel("Frequency [MHz]", fontsize=13)

        fig.suptitle("Calibration Parameters", fontsize=15)
        return fig

    def write(self, direc="."):
        # TODO: this is a bad function
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
                out = fnc(src.s11_model())
                key = "fit_s11_{}_{}".format(source, part)
                np.savetxt(os.path.join(direc, key + ".txt"), out)

            key = "{}_thermistor_temp".format(source)
            np.savetxt(os.path.join(direc, key + ".txt"), src.thermistor_temp)
