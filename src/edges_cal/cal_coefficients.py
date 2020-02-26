# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2019
Original author: Nivedita Mahesh
Edited by: David Lewis, Steven Murray

This is the main module of `cal_coefficients`. It contains wrappers around lower-level
functions in other modules.
"""

import os
import warnings
from functools import lru_cache
from hashlib import md5

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from edges_io import io
from edges_io.logging import logger

from . import S11_correction as s11
from . import modelling as mdl
from . import receiver_calibration_func as rcf
from . import reflection_coefficient as rc
from . import xrfi
from .cached_property import cached_property


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
        return np.arange(0, max_freq, fstep)


class VNA:
    """
    An object representing the measurements of a VNA.

    The measurements are read in via a .s1p file

    Parameters
    ----------
    path : str
        The root to a valid .s1p file containing VNA measurements.
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

    def __init__(self, s1p, f_low=None, f_high=None, switchval=None):
        if type(s1p) == str:
            self.s1p = io.S1P(s1p)
        elif type(s1p) == io.S1P:
            self.s1p = s1p
        else:
            raise ValueError("s1p must be a path to an s1p file, or an io.S1P object")

        self.load_name = self.s1p.kind
        self.run_num = self.s1p.run_num

        spec = self.s1p.s11
        f = self.s1p.freq

        self.freq = FrequencyRange(f, f_low, f_high)
        self.s11 = spec[self.freq.mask]
        self._switchval = switchval

    @cached_property
    def switchval(self):
        """The standard value of the switch for the component"""
        if self._switchval is not None:
            return self._switchval * np.ones_like(self.freq.freq)
        else:
            return None


class SwitchCorrection:
    default_nterms = {
        "ambient": 37,
        "hot_load": 37,
        "open": 105,
        "short": 105,
        "AntSim2": 55,
        "AntSim3": 55,
        "AntSim4": 55,
        "lna": 37,
    }

    def __init__(
        self,
        load_s11: io._S11SubDir,
        internal_switch,
        f_low=None,
        f_high=None,
        resistance=50.166,
        n_terms=None,
    ):
        """
        A class representing relevant switch corrections for a load.

        Parameters
        ----------
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
        self.load_s11 = load_s11
        self.base_path = self.load_s11.path

        try:
            self.load_name = getattr(self.load_s11, "load_name")
        except AttributeError:
            self.load_name = None

        self.run_num = self.load_s11.run_num
        self.resistance = resistance

        self.internal_switch = internal_switch
        switchvals = {"open": 1, "short": -1, "match": 0}

        for name in self.load_s11.STANDARD_NAMES:
            setattr(
                self,
                name.lower(),
                VNA(
                    s1p=getattr(self.load_s11, name.lower()),
                    f_low=f_low,
                    f_high=f_high,
                    switchval=switchvals.get(name.lower(), None),
                ),
            )

        # Expose one of the frequency objects
        self.freq = self.open.freq
        self._nterms = n_terms

    @cached_property
    def n_terms(self):
        """Number of terms to use (by default) in modelling the S11"""
        return self._nterms or self.default_nterms.get(self.load_name, None)

    @classmethod
    def from_path(
        cls,
        load_name,
        path,
        run_num_load=None,
        run_num_switch=None,
        repeat_num=None,
        **kwargs,
    ):
        antsim = load_name.startswith("ant_sim")

        if not antsim:
            s11_load_dir = io.LoadS11(
                os.path.join(path, load_name), run_num=run_num_load
            )
        else:
            s11_load_dir = io.AntSimS11(
                os.path.join(path, load_name), run_num=run_num_load
            )

        internal_switch = io.SwitchingState(
            os.path.join(path, "SwitchingState{:>02}".format(repeat_num)),
            run_num=run_num_switch,
        )
        return cls(s11_load_dir, internal_switch=internal_switch, **kwargs)

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
        return s11.low_band_switch_correction(
            self.switch_corrections[0],
            self.internal_switch,
            f_in=self.freq.freq,
            resistance_m=self.resistance,
        )[0]

    @lru_cache()
    def get_s11_correction_model(self, n_terms=None):
        """
        Generate a callable model for the S11 correction.

        This should closely match :method:`s11_correction`.

        Parameters
        ----------
        n_terms : int, optional
            Number of terms used in the fourier-based model. Not necessary if `load_name`
            is specified in the class.

        Returns
        -------
        callable :
            A function of one argument, f, which should be a frequency in the same units
            as `self.freq.freq`.
        """

        n_terms = n_terms or self.n_terms

        if not isinstance(n_terms, int):
            raise ValueError(
                "n_terms must be an integer, got {} with load {}".format(
                    n_terms, self.load_name
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
                "fourier", self.freq.freq_recentred, d, n_terms
            )[0]
            return lambda x: mdl.model_evaluate("fourier", fit, x)

        mag = get_model(True)
        ang = get_model(False)

        def model(f):
            ff = self.freq.normalize(f)
            return mag(ff) * (np.cos(ang(ff)) + 1j * np.sin(ang(ff)))

        return model

    @cached_property
    def s11_model(self):
        return self.get_s11_correction_model()

    def plot_residuals(self):
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
        model = self.s11_model
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
    def __init__(self, load_s11: io.ReceiverReading, resistance=50.009, **kwargs):
        """
        A special case of :class:`SwitchCorrection` for the LNA.
        """
        super().__init__(load_s11=load_s11, resistance=resistance, **kwargs)
        self.load_name = "lna"
        self.repeat_num = self.load_s11.repeat_num

    @classmethod
    def from_path(cls, path, run_num_load=None, run_num_switch=None, **kwargs):
        load_s11 = io.ReceiverReading(direc=path, run_num=run_num_load, fix=False)
        internal_switch = io.SwitchingState(
            os.path.dirname(os.path.normpath(path)), run_num=run_num_switch
        )
        return cls(load_s11, internal_switch=internal_switch, **kwargs)

    @cached_property
    def external(self):
        """
        VNA S11 measurements for the load.
        """
        return VNA(
            self.load_s11.receiverreading,
            f_low=self.freq.freq.min(),
            f_high=self.freq.freq.max(),
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
    def __init__(
        self,
        spec_obj: io.Spectrum,
        resistance_obj: io.Resistance,
        switch_correction=None,
        f_low=None,
        f_high=None,
        ignore_times_percent=5,
        rfi_removal="1D2D",
        rfi_kernel_width_time=16,
        rfi_kernel_width_freq=16,
        rfi_threshold=6,
        cache_dir=None,
    ):
        """
        A class representing a measured spectrum from some Load.

        Parameters
        ----------
        switch_correction : :class:`SwitchCorrection`, optional
            A `SwitchCorrection` for this particular load. If not given, will be
            constructed automatically.
        f_low : float, optional
            Minimum frequency to keep.
        f_high : float, optional
            Maximum frequency to keep.
        ignore_times_percent : float, optional
            Must be between 0 and 100. Number of time-samples in a file to reject
            from the start of the file.
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
        cache_dir : str path, optional
            An alternative directory in which to load/save cached reduced files. By
            default, the same as the path to the .mat files. If you don't have
            write permission there, it may be useful to use an alternative path.
        """
        self.spec_obj = spec_obj
        self.resistance_obj = resistance_obj

        self.load_name = self.spec_obj.load_name
        assert (
            self.spec_obj.load_name == self.resistance_obj.load_name
        ), "spec and resistance load_name must be the same"

        self.spec_files = self.spec_obj.path
        self.resistance_file = self.resistance_obj.path

        self.run_num = self.spec_obj.run_num

        self.cache_dir = cache_dir or os.path.curdir

        self.rfi_kernel_width_time = rfi_kernel_width_time
        self.rfi_kernel_width_freq = rfi_kernel_width_freq
        self.rfi_threshold = rfi_threshold

        assert rfi_removal in [
            "1D",
            "2D",
            "1D2D",
            False,
            None,
        ], "rfi_removal must be either '1D', '2D', '1D2D, or False/None"

        self.rfi_removal = rfi_removal

        self.switch_correction = switch_correction

        self.ignore_times_percent = ignore_times_percent
        self.freq = EdgesFrequencyRange(f_low=f_low, f_high=f_high)

    @classmethod
    def from_load_name(cls, load_name, direc, run_num=None, filetype=None, **kwargs):
        """Instantiate the class using a simple form, passing the load_name and direc"""
        spec = io.Spectrum.from_load(
            load=load_name,
            direc=os.path.join(direc, "Spectra"),
            run_num=run_num,
            filetype=filetype,
        )
        res = io.Resistance.from_load(
            load=load_name,
            direc=os.path.join(direc, "Resistance"),
            run_num=run_num,
            filetype=filetype,
        )
        return cls(spec_obj=spec, resistance_obj=res, **kwargs)

    @cached_property
    def averaged_Q(self):
        """
        Ratio of powers, Q = (P_source - P_load)/(P_noise - P_load).
        Average over time.
        """
        # TODO: should also get weights!
        spec = self._ave_and_var_spec[0]["Qratio"]

        if self.rfi_removal == "1D":
            spec = xrfi.remove_rfi(
                spec, threshold=self.rfi_threshold, Kf=self.rfi_kernel_width_freq
            )
        return spec

    @cached_property
    def variance_Q(self):
        """Variance of Q across time (see averaged_Q)"""
        return self._ave_and_var_spec[1]["Qratio"]

    @property
    def averaged_spectrum(self):
        """T* = T_noise * Q  + T_load"""
        return self.averaged_Q * 400 + 300

    @property
    def variance_spectrum(self):
        """Variance of uncalibrated spectrum across time (see averaged_spectrum)"""
        return self.variance_Q * 400 ** 2

    @cached_property
    def averaged_p0(self):
        return self._ave_and_var_spec[0]["p0"]

    @cached_property
    def averaged_p1(self):
        return self._ave_and_var_spec[0]["p1"]

    @cached_property
    def averaged_p2(self):
        return self._ave_and_var_spec[0]["p2"]

    @cached_property
    def variance_p0(self):
        return self._ave_and_var_spec[1]["p0"]

    @cached_property
    def variance_p1(self):
        return self._ave_and_var_spec[1]["p1"]

    @cached_property
    def variance_p2(self):
        return self._ave_and_var_spec[1]["p2"]

    def _get_integrated_filename(self):
        """Determine the relevant unique filename for the reduced data (averaged over time)
        for this instance"""
        params = (
            self.rfi_threshold,
            self.rfi_kernel_width_time,
            self.rfi_kernel_width_freq,
            self.rfi_removal,
            self.ignore_times_percent,
            self.freq.min,
            self.freq.max,
            self.spec_files,
        )
        hsh = md5(str(params).encode()).hexdigest()

        return os.path.join(self.cache_dir, self.load_name + "_" + hsh + ".h5")

    @cached_property
    def _ave_and_var_spec(self):
        """Get the mean and variance of the spectra"""
        fname = self._get_integrated_filename()
        kinds = ["p0", "p1", "p2", "Qratio"]
        if os.path.exists(fname):
            logger.info(
                "Reading in previously-created integrated {} spectra...".format(
                    self.load_name
                )
            )
            means = {}
            vars = {}
            with h5py.File(fname, "r") as fl:
                for kind in kinds:
                    means[kind] = fl[kind + "_mean"][...]
                    vars[kind] = fl[kind + "_var"][...]
            return means, vars

        logger.info("Reducing {} spectra...".format(self.load_name))
        spectra = self.get_spectra()

        means = {}
        vars = {}
        for key, spec in spectra.items():
            # Weird thing where there are zeros in the spectra.
            spec[spec == 0] = np.nan

            mean = np.nanmean(spec, axis=1)
            var = np.nanvar(spec, axis=1)

            if self.rfi_removal == "1D2D":
                nsample = np.sum(~np.isnan(spec), axis=1)
                varfilt = xrfi.medfilt(
                    var, kernel_size=2 * self.rfi_kernel_width_freq + 1
                )
                resid = mean - xrfi.medfilt(
                    mean, kernel_size=2 * self.rfi_kernel_width_freq + 1
                )
                flags = np.logical_or(
                    resid > self.rfi_threshold * np.sqrt(varfilt / nsample),
                    var - varfilt
                    > self.rfi_threshold * np.sqrt(2 * varfilt ** 2 / (nsample - 1)),
                )

                mean[flags] = np.nan
                var[flags] = np.nan

            means[key] = mean
            vars[key] = var

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        with h5py.File(fname, "w") as fl:
            logger.info("Saving reduced spectra to cache at {}".format(fname))
            for kind in kinds:
                fl[kind + "_mean"] = means[kind]
                fl[kind + "_var"] = vars[kind]

        return means, vars

    def get_spectra(self):
        spec = self._read_spectrum()

        if self.rfi_removal == "2D":
            for key, val in spec.items():
                # Need to set nans and zeros to inf so that median/mean detrending can work.
                val[np.isnan(val)] = np.inf

                if key != "Qratio":
                    val[val == 0] = np.inf

                val = xrfi.remove_rfi(
                    val,
                    threshold=self.rfi_threshold,
                    Kt=self.rfi_kernel_width_time,
                    Kf=self.rfi_kernel_width_freq,
                )
                spec[key] = val
        return spec

    def _read_spectrum(self):
        """
        Read the contents of the spectrum files into memory, removing a starting
        percentage of times, and masking out certain frequencies.

        Returns
        -------
        dict : a dictionary of the contents of the file. Usually p0, p1, p2 (un-normalised
               powers of source, load, and load+noise respectively), and ant_temp (the
               uncalibrated, but normalised antenna temperature).
        """
        out = self.spec_obj.read()

        for key, val in out.items():
            index_start_spectra = int(
                (self.ignore_times_percent / 100) * len(val[0, :])
            )
            out[key] = val[self.freq.mask, index_start_spectra:]
        return out

    @cached_property
    def thermistor_temp(self):
        """
        Read a resistance file and return the associated thermistor temperature in K.
        """
        resistance = self.resistance_obj.read()
        temp_spectrum = rcf.temperature_thermistor(resistance)
        return temp_spectrum[
            int((self.ignore_times_percent / 100) * len(temp_spectrum)) :
        ]

    @cached_property
    def temp_ave(self):
        """Average thermistor temperature (over time and frequency)"""
        return np.mean(self.thermistor_temp)

    def write(self, path=None):
        """
        Write a HDF5 file containing the contents of the LoadSpectrum.

        Parameters
        ----------
        path : str
            Directory into which to save the file, or full path to file.
            If a directory, filename will be <load_name>_averaged_spectrum.h5.
            Default is current directory.
        """
        direc = path or os.path.curdir

        # Allow to pass in a directory name *or* full path.
        if os.path.isdir(path):
            path = os.path.join(direc, self.load_name + "_averaged_spectrum.h5")

        with h5py.File(path) as fl:
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

    _kinds = {"s11": 0, "s12": 1, "s22": 2}

    def __init__(self, path, f_low=None, f_high=None):
        self.path = path
        data = np.genfromtxt(
            os.path.join(path, "semi_rigid_s_parameters_WITH_HEADER.txt")
        )

        f = data[:, 0]
        self.freq = FrequencyRange(f, f_low, f_high)
        self.data = data[self.freq.mask, 1::2] + 1j * data[self.freq.mask, 2::2]

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
        d = self.data[:, self._kinds[kind]]
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

    def power_gain(self, freq, hot_load_s11: SwitchCorrection):
        assert isinstance(
            hot_load_s11, SwitchCorrection
        ), "hot_load_s11 must be a switch correction"
        assert (
            hot_load_s11.load_name == "hot_load"
        ), "hot_load_s11 must be a hot_load s11"

        return self.get_power_gain(
            {
                "s11": self.s11_model(freq),
                "s12s21": self.s12_model(freq),
                "s22": self.s22_model(freq),
            },
            hot_load_s11.s11_model(freq),
        )

    @staticmethod
    def get_power_gain(semi_rigid_sparams, hot_load_s11):
        """Define Eq. 9 from M17"""
        rht = rc.gamma_de_embed(
            semi_rigid_sparams["s11"],
            semi_rigid_sparams["s12s21"],
            semi_rigid_sparams["s22"],
            hot_load_s11,
        )

        return (
            np.abs(semi_rigid_sparams["s12s21"])
            * (1 - np.abs(rht) ** 2)
            / (
                (np.abs(1 - semi_rigid_sparams["s11"] * rht)) ** 2
                * (1 - np.abs(hot_load_s11) ** 2)
            )
        )


class Load:
    """Wrapper class containing all relevant information for a given load."""

    def __init__(
        self,
        spectrum: LoadSpectrum,
        reflections: SwitchCorrection,
        hot_load_correction: [HotLoadCorrection, None] = None,
        ambient: [LoadSpectrum, None] = None,
    ):
        assert isinstance(spectrum, LoadSpectrum), "spectrum must be a LoadSpectrum"
        assert isinstance(
            reflections, SwitchCorrection
        ), "spectrum must be a SwitchCorrection"
        assert spectrum.load_name == reflections.load_name

        self.spectrum = spectrum
        self.reflections = reflections
        self.load_name = spectrum.load_name

        if self.load_name == "hot_load":
            self._correction = hot_load_correction
            self._ambient = ambient

    @classmethod
    def from_path(
        cls,
        spec_path: str,
        reflection_path: str,
        load_name: str,
        run_num_spec: [int, None] = None,
        filetype: [str, None] = None,
        run_num_load: [int, None] = None,
        f_low: [float, None] = None,
        f_high: [float, None] = None,
        run_num_switch: [int, None] = None,
        reflection_kwargs: [dict, None] = None,
        spec_kwargs: [dict, None] = None,
        repeat_num: [int, None] = None,
    ):
        if not spec_kwargs:
            spec_kwargs = {}
        if not reflection_kwargs:
            reflection_kwargs = {}

        spec = LoadSpectrum.from_load_name(
            load_name,
            spec_path,
            run_num=run_num_spec,
            filetype=filetype,
            f_low=f_low,
            f_high=f_high,
            **spec_kwargs,
        )

        refl = SwitchCorrection.from_path(
            load_name,
            reflection_path,
            run_num_load=run_num_load,
            run_num_switch=run_num_switch,
            f_low=f_low,
            f_high=f_high,
            repeat_num=repeat_num,
            **reflection_kwargs,
        )

        return cls(spec, refl)

    @property
    def s11_model(self):
        return self.reflections.s11_model

    @cached_property
    def temp_ave(self):
        if self.load_name != "hot_load":
            return self.spectrum.temp_ave
        else:
            G = self._correction.power_gain(self.freq.freq, self.reflections)

            # temperature
            return G * self.spectrum.temp_ave + (1 - G) * self._ambient.temp_ave

    @property
    def averaged_Q(self):
        return self.spectrum.averaged_Q

    @property
    def averaged_spectrum(self):
        return self.spectrum.averaged_spectrum

    @property
    def freq(self):
        return self.spectrum.freq


class CalibrationObservation:
    _sources = ("ambient", "hot_load", "open", "short")

    def __init__(
        self,
        path,
        semi_rigid_path=None,
        ambient_temp=25,
        f_low=None,
        f_high=None,
        run_num=None,
        repeat_num=None,
        resistance_f=50.009,
        resistance_m=50.166,
        ignore_times_percent=5,
        cterms=5,
        wterms=7,
        rfi_removal="1D2D",
        rfi_kernel_width_time=16,
        rfi_kernel_width_freq=16,
        rfi_threshold=6,
        cache_dir=None,
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
            A root to switch corrections, if different from base root.
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
        ignore_times_percent : float, optional
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
        self.io = io.CalibrationObservation(
            path,
            ambient_temp=ambient_temp,
            run_num=run_num,
            repeat_num=repeat_num,
            fix=False,
        )

        self.path = self.io.path

        hot_load_correction = HotLoadCorrection(
            semi_rigid_path
            or os.path.join(
                io.utils.get_parent_dir(self.path, 2), "SemiRigidCableMeasurements"
            ),
            f_low,
            f_high,
        )

        self._loads = {}
        for source in self._sources:
            load = LoadSpectrum(
                spec_obj=getattr(self.io.spectra, source),
                resistance_obj=getattr(self.io.resistance, source),
                f_low=f_low,
                f_high=f_high,
                ignore_times_percent=ignore_times_percent,
                rfi_removal=rfi_removal,
                rfi_kernel_width_freq=rfi_kernel_width_freq,
                rfi_kernel_width_time=rfi_kernel_width_time,
                rfi_threshold=rfi_threshold,
                cache_dir=cache_dir,
            )

            refl = SwitchCorrection(
                getattr(self.io.s11, source),
                self.io.s11.switching_state,
                f_low=f_low,
                f_high=f_high,
                resistance=resistance_m,
            )
            if source == "hot_load":
                self._loads[source] = Load(
                    load,
                    refl,
                    hot_load_correction=hot_load_correction,
                    ambient=self._loads["ambient"].spectrum,
                )
            else:
                self._loads[source] = Load(load, refl)

        self.short = self._loads["short"]
        self.open = self._loads["open"]
        self.hot_load = self._loads["hot_load"]
        self.ambient = self._loads["ambient"]

        self.lna = LNA(
            self.io.s11.receiver_reading,
            internal_switch=self.io.s11.switching_state,
            f_low=f_low,
            f_high=f_high,
            resistance=resistance_f,
        )

        self.cterms = cterms
        self.wterms = wterms

        # Expose a Frequency object
        self.freq = self.ambient.freq

    def new_load(
        self,
        load_name,
        run_num_spec=None,
        run_num_load=None,
        reflection_kwargs=None,
        spec_kwargs=None,
    ):
        """Create a new load with the given load name, from files inside the current observation"""
        if reflection_kwargs is None:
            reflection_kwargs = {}
        if spec_kwargs is None:
            spec_kwargs = {}

        # Fill up kwargs with keywords from this instance
        if "resistance" not in reflection_kwargs:
            reflection_kwargs["resistance"] = self.open.reflections.resistance
        for key in [
            "ignore_times_percent",
            "rfi_removal",
            "rfi_kernel_width_freq",
            "rfi_kernel_width_time",
            "rfi_threshold",
            "cache_dir",
        ]:
            if key not in spec_kwargs:
                spec_kwargs[key] = getattr(self.open.spectrum, key)

        return Load.from_path(
            spec_path=self.io.path,
            load_name=load_name,
            reflection_path=self.io.s11.path,
            f_low=self.freq.min,
            f_high=self.freq.max,
            run_num_switch=self.io.s11.switching_state.run_num,
            run_num_load=run_num_load,
            run_num_spec=run_num_spec,
            repeat_num=self.io.s11.switching_state.repeat_num,
            reflection_kwargs=reflection_kwargs,
            spec_kwargs=spec_kwargs,
        )

    def plot_raw_spectra(self, fig=None, ax=None):
        """
        Plot raw uncalibrated spectra for all calibrator sources.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(
                len(self._sources), 1, sharex=True, gridspec_kw={"hspace": 0.05}
            )

        for i, (name, load) in enumerate(self._loads.items()):
            load.spectrum.plot(
                fig=fig, ax=ax[i], xlabel=(i == (len(self._sources) - 1))
            )
            ax[i].set_title(name)

        return fig

    def plot_s11_models(self):
        """
        Plot residuals of S11 models for all sources

        Returns
        -------
        dict:
            Each entry has a key of the source name, and the value is a matplotlib figure.
        """
        return {
            name: source.reflections.plot_residuals()
            for name, source in self._loads.items()
        }

    @cached_property
    def s11_correction_models(self):
        """Dictionary of S11 correction models, one for each source"""
        return {
            name: source.s11_model(self.freq.freq)
            for name, source in self._loads.items()
        }

    @cached_property
    def _calibration_coefficients(self):
        """
        The calibration polynomials, C1, C2, Tunc, Tcos, Tsin, evaluated at `freq.freq`.
        """
        scale, off, Tu, TC, TS = rcf.get_calibration_quantities_iterative(
            self.freq.freq_recentred,
            T_raw={k: source.averaged_spectrum for k, source in self._loads.items()},
            gamma_rec=self.lna.s11_model(self.freq.freq),
            gamma_ant=self.s11_correction_models,
            T_ant={k: source.temp_ave for k, source in self._loads.items()},
            cterms=self.cterms,
            wterms=self.wterms,
        )
        return scale, off, Tu, TC, TS

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
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.C1_poly(fnorm)

    def C2(self, f=None):
        """
        Offset calibration parameter.
        """
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.C2_poly(fnorm)

    def Tunc(self, f=None):
        """
        Uncorrelated noise-wave parameter
        """
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.Tunc_poly(fnorm)

    def Tcos(self, f=None):
        """
        Cosine noise-wave parameter
        """
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.Tcos_poly(fnorm)

    def Tsin(self, f=None):
        """
        Sine noise-wave parameter
        """
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.Tsin_poly(fnorm)

    def get_linear_coefficients(self, load: [Load, str]):
        """
        Calibration coefficients a,b such that T = aT* + b (derived from Eq. 7)
        """
        load = self._load_str_to_load(load)
        return rcf.get_linear_coefficients(
            load.s11_model(self.freq.freq),
            self.lna.s11_model(self.freq.freq),
            self.C1(self.freq.freq),
            self.C2(self.freq.freq),
            self.Tunc(self.freq.freq),
            self.Tcos(self.freq.freq),
            self.Tsin(self.freq.freq),
            T_load=300,
        )

    def calibrate(self, load: [Load, str]):
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
        load = self._load_str_to_load(load)
        a, b = self.get_linear_coefficients(load)
        return a * load.averaged_spectrum + b

    def _load_str_to_load(self, load: [Load, str]):
        if type(load) == str:
            try:
                load = self._loads[load]
            except AttributeError:
                raise AttributeError(
                    "load must be a Load object or a string (one of {ambient,hot_load,open,short}"
                )
        else:
            assert isinstance(
                load, Load
            ), "load must be a Load instance, got the {} {}".format(load, type(Load))
        return load

    def decalibrate(self, temp: np.ndarray, load: [Load, str], freq: np.ndarray = None):
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

        a, b = self.get_linear_coefficients(load)
        return (temp - b) / a

    def plot_calibrated_temp(
        self,
        load: [Load, str],
        bins: int = 2,
        fig=None,
        ax=None,
        xlabel=True,
        ylabel=True,
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
        load = self._load_str_to_load(load)

        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, facecolor="w")

        # binning
        temp_calibrated = self.calibrate(load)

        if bins > 0:
            freq_ave_cal = convolve(
                temp_calibrated, Gaussian1DKernel(stddev=bins), boundary="extend"
            )
        else:
            freq_ave_cal = temp_calibrated

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        ax.plot(
            self.freq.freq,
            freq_ave_cal,
            label=f"Calibrated {load.spectrum.load_name} [RMS = {rms:.3f}]",
        )

        if load.load_name != "hot_load":
            ax.axhline(load.temp_ave, color="C2", label="Average thermistor temp")
        else:
            ax.plot(
                self.freq.freq,
                load.temp_ave,
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

    def get_load_residuals(self):
        """Get residuals of the calibrated temperature for a each load."""
        out = {}
        for source in self._sources:
            load = self._load_str_to_load(source)
            cal = self.calibrate(load)
            true = load.temp_ave
            out[source] = cal - true
        return out

    def get_rms(self, smooth: int = 4):
        """Return a dict of RMS values for each source."""
        resids = self.get_load_residuals()
        out = {}
        for name, res in resids.items():
            if smooth > 1:
                res = convolve(res, Gaussian1DKernel(stddev=smooth), boundary="extend")
            out[name] = np.sqrt(np.nanmean(res ** 2))
        return out

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
                source,
                bins=bins,
                fig=fig,
                ax=ax[i],
                xlabel=i == (len(self._sources) - 1),
            )

        fig.suptitle("Calibrated Temperatures for Calibration Sources", fontsize=15)
        return fig

    def write_coefficients(self, path: [str, None] = None):
        """
        Save a text file with the derived calibration co-efficients.

        Parameters
        ----------
        path : str
            Directory in which to write the file. The filename starts with `All_cal-params`
            and includes parameters of the class in the filename. By default, current
            directory.
        """
        path = path or os.path.curdir

        if os.path.isdir(path):
            path = os.path.join(
                path,
                "calibration_parameters_fmin{}_fmax{}_C{}_W{}.txt".format(
                    self.freq.freq.min(), self.freq.freq.max(), self.cterms, self.wterms
                ),
            )

        np.savetxt(
            path,
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

    def invalidate_cache(self):
        """Invalidate all cached attributes so they must be recalculated."""
        if not hasattr(self, "_cached_"):
            return

        for cache in self._cached_:
            del self.__dict__[cache]

    def update(self, **kwargs):
        """Update the class in-place, invalidating the cache as well."""
        self.invalidate_cache()
        for k, v in kwargs.items():
            setattr(self, k, v)
