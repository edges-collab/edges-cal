"""
The main user-facing module of ``edges-cal``.

This module contains wrappers around lower-level functions in other modules, providing
a one-stop interface for everything related to calibration.
"""
from __future__ import annotations

import attr
import h5py
import numpy as np
import tempfile
import warnings
import yaml
from abc import ABCMeta, abstractmethod
from astropy.convolution import Gaussian1DKernel, convolve
from copy import copy
from edges_io import io
from edges_io.logging import logger
from functools import lru_cache
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from typing import Any, Callable, Literal

from . import modelling as mdl
from . import receiver_calibration_func as rcf
from . import reflection_coefficient as rc
from . import s11_correction as s11
from . import tools
from . import types as tp
from . import xrfi
from .cached_property import cached_property, safe_property
from .tools import FrequencyRange, get_data_path


def _s1p_converter(s1p: tp.PathLike | io.S1P) -> io.S1P:
    try:
        s1p = Path(s1p)
        return io.S1P(s1p)
    except TypeError:
        if isinstance(s1p, io.S1P):
            return s1p
        else:
            raise TypeError("s1p must be a path to an s1p file, or an io.S1P object")


@attr.s(frozen=True)
class S1P:
    """
    An object representing the measurements of a VNA.

    The measurements are read in via a .s1p file

    Parameters
    ----------
    s1p
        The path to a valid .s1p file containing VNA measurements, or an S1P
        object of such a type.
    f_low, f_high
        The minimum/maximum frequency to keep.
    switchval
        The standard value of the switch for the component.
    """

    s1p: tp.PathLike | io.S1P = attr.ib(converter=_s1p_converter)
    f_low: float | None = attr.ib(
        default=None, converter=attr.converters.optional(float), kw_only=True
    )
    f_high: float | None = attr.ib(
        default=None, converter=attr.converters.optional(float), kw_only=True
    )
    _switchval: int | None = attr.ib(
        default=None, kw_only=True, converter=attr.converters.optional(int)
    )

    @property
    def load_name(self) -> str:
        """The name of the load whose S11 this is."""
        return self.s1p.kind

    @property
    def repeat_num(self) -> int:
        """The repeat number of this S11 measurement."""
        return int(self.s1p.repeat_num)

    @cached_property
    def freq(self) -> FrequencyRange:
        """The frequencies of the S11 measurement."""
        kwargs = {}
        if self.f_low is not None:
            kwargs["f_low"] = self.f_low
        if self.f_high is not None:
            kwargs["f_high"] = self.f_high

        return FrequencyRange(self.s1p.freq, **kwargs)

    @cached_property
    def s11(self) -> np.ndarray:
        """The S11 measurement."""
        return self.s1p.s11[self.freq.mask]

    @cached_property
    def switchval(self):
        """The standard value of the switch for the component."""
        if self._switchval is not None:
            return self._switchval * np.ones(self.freq.n)
        else:
            return None


# For backwards compatibility
VNA = S1P


@attr.s(kw_only=True, frozen=True)
class _S11Base(metaclass=ABCMeta):
    """
    A class containing the S11 measurements (and corrections) for a load/interface.

    Parameters
    ----------
    load_s11 : :class:`io._S11SubDir`
        An instance of the basic ``io`` S11 folder.
    f_low : float
        Minimum frequency to use. Default is all frequencies.
    f_high : float
        Maximum frequency to use. Default is all frequencies.
    resistance : float
        The resistance of the switch (in Ohms).
    n_terms : int
        The number of terms to use in fitting a model to the S11 (used to both
        smooth and interpolate the data). Must be odd.
    """

    default_nterms = {
        "ambient": 37,
        "hot_load": 37,
        "open": 105,
        "short": 105,
        "AntSim2": 55,
        "AntSim3": 55,
        "AntSim4": 55,
        "AntSim1": 55,
        "lna": 37,
    }
    _switchvals = {"open": 1, "short": -1, "match": 0}

    load_s11: io._S11SubDir | io.ReceiverReading = attr.ib()
    f_low: float | None = attr.ib(default=None)
    f_high: float | None = attr.ib(default=None)
    _n_terms: int | None = attr.ib(
        default=None, converter=attr.converters.optional(int)
    )
    model_type: tp.Modelable = attr.ib(default="fourier")

    @property
    def base_path(self) -> Path:
        """The path to the S11 measurements."""
        return self.load_s11.path

    @cached_property
    def load_name(self) -> str | None:
        try:
            return getattr(self.load_s11, "load_name")
        except AttributeError:
            return None

    @property
    def run_num(self) -> int:
        """The run number of the S11 measurement."""
        return self.load_s11.run_num

    @cached_property
    def _standards(self) -> dict[str, S1P]:
        return {
            name.lower(): S1P(
                s1p=self.load_s11.children[name.lower()],
                f_low=self.f_low,
                f_high=self.f_high,
                switchval=self._switchvals.get(name.lower()),
            )
            for name in self.load_s11.STANDARD_NAMES
        }

    def __getattr__(self, item):
        if item in self._standards:
            return self._standards[item]

        raise AttributeError(f"{item} does not exist in {self.__class__.__name__}!")

    @property
    def freq(self) -> FrequencyRange:
        """The frequencies at which the internal standards were measured."""
        return self.open.freq

    @cached_property
    def n_terms(self):
        """Number of terms to use (by default) in modelling the S11.

        Raises
        ------
        ValueError
            If n_terms is even.
        """
        res = self._n_terms or self.default_nterms.get(self.load_name, None)
        if not (isinstance(res, int) and res % 2):
            raise ValueError(
                f"n_terms must be odd for S11 models. For {self.load_name} got "
                f"n_terms={res}."
            )
        return res

    @classmethod
    @abstractmethod
    def from_path(cls, **kwargs):
        pass  # pragma: no cover

    @cached_property
    @abstractmethod
    def measured_load_s11_raw(self):
        pass  # pragma: no cover

    @cached_property
    def corrected_load_s11(self) -> np.ndarray:
        """The measured S11 of the load, corrected for internal switch."""
        return self.measured_load_s11_raw

    @lru_cache
    def get_corrected_s11_model(
        self,
        n_terms: int | None = None,
        model_type: tp.Modelable | None = None,
    ):
        """Generate a callable model for the S11 correction.

        This should closely match :method:`s11_correction`.

        Parameters
        ----------
        n_terms : int
            Number of terms used in the fourier-based model. Not necessary if
            `load_name` is specified in the class.

        Returns
        -------
        callable :
            A function of one argument, f, which should be a frequency in the same units
            as `self.freq.freq`.

        Raises
        ------
        ValueError
            If n_terms is not an integer, or not odd.
        """
        n_terms = n_terms or self.n_terms
        model_type = mdl.get_mdl(model_type or self.model_type)
        model = model_type(
            n_terms=n_terms,
            transform=mdl.UnitTransform(range=[self.freq.min, self.freq.max]),
        )
        emodel = model.at(x=self.freq.freq)

        cmodel = mdl.ComplexMagPhaseModel(mag=emodel, phs=emodel)

        s11_correction = self.corrected_load_s11

        return cmodel.fit(ydata=s11_correction)

    @cached_property
    def s11_model(self) -> callable:
        """The S11 model."""
        return self.get_corrected_s11_model()

    def plot_residuals(
        self,
        fig=None,
        ax=None,
        color_abs="C0",
        color_diff="g",
        label=None,
        title=None,
        decade_ticks=True,
        ylabels=True,
    ) -> plt.Figure:
        """
        Make a plot of the residuals of the S11 model and the correction data.

        Residuals obtained  via :func:`get_corrected_s11_model`

        Returns
        -------
        fig :
            Matplotlib Figure handle.
        """
        if fig is None or ax is None or len(ax) != 4:
            fig, ax = plt.subplots(
                4, 1, sharex=True, gridspec_kw={"hspace": 0.05}, facecolor="w"
            )

        if decade_ticks:
            for axx in ax:
                axx.xaxis.set_ticks(
                    [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
                    minor=[],
                )
                axx.grid(True)
        ax[-1].set_xlabel("Frequency [MHz]")

        corr = self.corrected_load_s11
        model = self.s11_model(self.freq.freq)

        ax[0].plot(
            self.freq.freq, 20 * np.log10(np.abs(model)), color=color_abs, label=label
        )
        if ylabels:
            ax[0].set_ylabel(r"$|S_{11}|$")

        ax[1].plot(self.freq.freq, np.abs(model) - np.abs(corr), color_diff)
        if ylabels:
            ax[1].set_ylabel(r"$\Delta  |S_{11}|$")

        ax[2].plot(
            self.freq.freq, np.unwrap(np.angle(model)) * 180 / np.pi, color=color_abs
        )
        if ylabels:
            ax[2].set_ylabel(r"$\angle S_{11}$")

        ax[3].plot(
            self.freq.freq,
            np.unwrap(np.angle(model)) - np.unwrap(np.angle(corr)),
            color_diff,
        )
        if ylabels:
            ax[3].set_ylabel(r"$\Delta \angle S_{11}$")

        if title is None:
            title = f"{self.load_name} Reflection Coefficient Models"

        if title:
            fig.suptitle(f"{self.load_name} Reflection Coefficient Models", fontsize=14)
        if label:
            ax[0].legend()

        return fig


@attr.s(kw_only=True, frozen=True)
class LoadS11(_S11Base):
    """S11 for a lab calibration load.

    Parameters
    ----------
    internal_switch : :class:`s11.InternalSwitch`
        The internal switch state corresponding to the load.

    Other Parameters
    ----------------
    Passed through to :class:`_S11Base`.
    """

    internal_switch: s11.InternalSwitch = attr.ib()

    @classmethod
    def from_path(
        cls,
        load_name: str,
        path: tp.PathLike,
        run_num_load: int = 1,
        run_num_switch: int = 1,
        repeat_num_load: int = attr.NOTHING,
        repeat_num_switch: int = attr.NOTHING,
        resistance: float = 50.166,
        model_internal_switch: mdl.Model = attr.NOTHING,
        **kwargs,
    ):
        """
        Create a new object from a given path and load name.

        Parameters
        ----------
        load_name : str
            The name of the load to create.
        path : str or Path
            The path to the overall calibration observation.
        run_num_load : int
            The run to use (default is last run available).
        run_num_switch : int
            The run to use for the switch S11 (default is last run available).
        kwargs
            All other arguments are passed through to the constructor of
            :class:`LoadS11`.

        Returns
        -------
        s11 : :class:`LoadS11`
            The S11 of the load.
        """
        antsim = load_name.startswith("AntSim")
        path = Path(path)

        if not antsim:
            load_name = io.LOAD_ALIASES[load_name]

        s11_load_dir = (io.AntSimS11 if antsim else io.LoadS11)(
            path / "S11" / f"{load_name}{run_num_load:02}", repeat_num=repeat_num_load
        )

        internal_switch = s11.InternalSwitch(
            data=io.SwitchingState(
                path / "S11" / f"SwitchingState{run_num_switch:02}",
                repeat_num=repeat_num_switch,
            ),
            resistance=resistance,
            model=model_internal_switch,
        )
        return cls(load_s11=s11_load_dir, internal_switch=internal_switch, **kwargs)

    @cached_property
    def measured_load_s11_raw(self):
        """The measured S11 of the load, calculated from raw internal standards."""
        return rc.de_embed(
            self.open.switchval,
            self.short.switchval,
            self.match.switchval,
            self.open.s11,
            self.short.s11,
            self.match.s11,
            self.external.s11,
        )[0]

    @cached_property
    def corrected_load_s11(self) -> np.ndarray:
        """The measured S11 of the load, corrected for the internal switch."""
        return rc.gamma_de_embed(
            self.internal_switch.s11_model(self.freq.freq),
            self.internal_switch.s12_model(self.freq.freq),
            self.internal_switch.s22_model(self.freq.freq),
            self.measured_load_s11_raw,
        )


@attr.s
class LNA(_S11Base):
    """A special case of :class:`SwitchCorrection` for the LNA.

    Parameters
    ----------
    load_s11
        The Receiver Reading S11 measurements.
    resistance
        The resistance of the receiver.
    kwargs
        All other arguments passed to :class:`SwitchCorrection`.
    """

    load_s11: io.ReceiverReading = attr.ib()
    resistance: float = attr.ib(default=50.009, kw_only=True)

    @cached_property
    def load_name(self) -> str:
        """The name of the load."""
        return "lna"

    @property
    def repeat_num(self) -> int:
        """The repeat num used for the LNA measurement."""
        return self.load_s11.repeat_num

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        repeat_num: int | None = attr.NOTHING,
        run_num: int = 1,
        **kwargs,
    ):
        """
        Create an instance from a given path.

        Parameters
        ----------
        path : str or Path
            Path to overall Calibration Observation.
        run_num_load : int
            The run to use for the LNA (default latest available).
        run_num_switch : int
            The run to use for the switching state (default lastest available).
        kwargs
            All other arguments passed through to :class:`SwitchCorrection`.

        Returns
        -------
        lna : :class:`LNA`
            The LNA object.
        """
        path = Path(path)
        load_s11 = io.ReceiverReading(
            path=path / "S11" / f"ReceiverReading{run_num:02}",
            repeat_num=repeat_num,
            fix=False,
        )

        return cls(load_s11=load_s11, **kwargs)

    @cached_property
    def external(self):
        """VNA S11 measurements for the load."""
        return S1P(
            self.load_s11.children["receiverreading"],
            f_low=self.freq.freq.min(),
            f_high=self.freq.freq.max(),
        )

    @cached_property
    def measured_load_s11_raw(self):
        """Measured S11 of of the LNA."""
        # Models of standards
        oa, sa, la = rc.agilent_85033E(
            self.freq.freq, self.resistance, match_delay=True
        )

        # Correction at switch
        return rc.de_embed(
            oa, sa, la, self.open.s11, self.short.s11, self.match.s11, self.external.s11
        )[0]


@attr.s(kw_only=True, frozen=True)
class LoadSpectrum:
    """A class representing a measured spectrum from some Load.

    Parameters
    ----------
    spec_obj : :class:`io.Spectrum`
        The base Spectrum object defining the on-disk spectra.
    resistance_obj : :class:`io.Resistance`
        The base Resistance object defining the on-disk resistance measurements.
    switch_correction : :class:`SwitchCorrection`
        A `SwitchCorrection` for this particular load. If not given, will be
        constructed automatically.
    f_low : float
        Minimum frequency to keep.
    f_high : float
        Maximum frequency to keep.
    ignore_times_percent : float
        Must be between 0 and 100. Number of time-samples in a file to reject
        from the start of the file.
    rfi_removal : str
        Either '1D', '2D' or '1D2D'. If given, will perform median and mean-filtered
        xRFI over either the
        2D waterfall, or integrated 1D spectrum. The latter is usually reasonable
        for calibration sources, while the former is good for field data. "1D2D"
        is a hybrid approach in which the variance per-frequency is determined
        from the 2D data, but filtering occurs only over frequency.
    rfi_kernel_width_time : int
        The kernel width for the detrending of data for
        RFI removal in the time dimension (only used if `rfi_removal` is "2D").
    rfi_kernel_width_freq : int
        The kernel width for the detrending of data for
        RFI removal in the frequency dimension.
    rfi_threshold : float
        The threshold (in equivalent standard deviation units) above which to
        flag data as RFI.
    cache_dir : str or Path
        An alternative directory in which to load/save cached reduced files.
    t_load
        Fiducial guess for the temperature of the internal load.
    t_load_ns
        Fiducial guess for the temperature of the internal load + noise source.
    freq_bin_size
        The size of the frequency bins, in units of their raw size (i.e. default of
        one is to not bin in frequency).
    """

    spec_obj: tuple[io.Spectrum] = attr.ib(converter=tuple)
    resistance_obj: io.Resistance = attr.ib()
    switch_correction: LoadS11 | None = attr.ib(default=None)
    f_low: float = attr.ib(default=40.0)
    f_high: float | None = attr.ib(default=None)
    ignore_times_percent: float = attr.ib(default=5.0)
    rfi_removal: Literal["1D", "2D", "1D2D", False, None] = attr.ib(default="1D2D")
    rfi_kernel_width_time: int = attr.ib(default=16, converter=int)
    rfi_kernel_width_freq: int = attr.ib(default=16, converter=int)
    rfi_threshold: float = attr.ib(default=6.0, converter=float)
    cache_dir: str | Path = attr.ib(default=".", converter=Path)
    t_load: float = attr.ib(default=300.0)
    t_load_ns: float = attr.ib(default=400.0)
    freq_bin_size: int = attr.ib(default=1)

    @property
    def load_name(self) -> str:
        """The name of the load."""
        return self.spec_obj[0].load_name

    @resistance_obj.validator
    def _resistance_validator(self, att, val):
        assert (
            self.spec_obj[0].load_name == val.load_name
        ), "spec and resistance load_name must be the same"

    @property
    def spec_files(self) -> tuple[Path]:
        """The tuple of files that are combined into this spectrum."""
        return tuple(spec_obj.path for spec_obj in self.spec_obj)

    @property
    def resistance_file(self) -> Path:
        """The path to the file holding resistance measurements."""
        return self.resistance_obj.path

    @property
    def run_num(self) -> int:
        """The run number used."""
        return self.spec_obj[0].run_num

    @cached_property
    def freq(self) -> FrequencyRange:
        """Frequencies of observation."""
        f_high = self.f_high or attr.NOTHING
        return FrequencyRange.from_edges(
            f_low=self.f_low, f_high=f_high, bin_size=self.freq_bin_size
        )

    @classmethod
    def from_load_name(
        cls,
        load_name: str,
        direc: str | Path,
        run_num: int | None = None,
        filetype: str | None = None,
        **kwargs,
    ):
        """Instantiate the class from a given load name and directory.

        Parameters
        ----------
        load_name : str
            The load name (one of 'ambient', 'hot_load', 'open' or 'short').
        direc : str or Path
            The top-level calibration observation directory.
        run_num : int
            The run number to use for the spectra.
        filetype : str
            The filetype to look for (acq or h5).
        kwargs :
            All other arguments to :class:`LoadSpectrum`.

        Returns
        -------
        :class:`LoadSpectrum`.
        """
        direc = Path(direc)

        spec = io.Spectrum.from_load(
            load=load_name, direc=direc / "Spectra", run_num=run_num, filetype=filetype
        )
        res = io.Resistance.from_load(
            load=load_name,
            direc=direc / "Resistance",
            run_num=run_num,
            filetype=filetype,
        )
        return cls(spec_obj=spec, resistance_obj=res, **kwargs)

    @cached_property
    def averaged_Q(self) -> np.ndarray:
        """Ratio of powers averaged over time.

        Notes
        -----
        The formula is

        .. math:: Q = (P_source - P_load)/(P_noise - P_load)
        """
        spec = self._ave_and_var_spec[0]["Q"]

        if self.rfi_removal == "1D":
            flags, _ = xrfi.xrfi_medfilt(
                spec, threshold=self.rfi_threshold, kf=self.rfi_kernel_width_freq
            )
            spec[flags] = np.nan
        return spec

    @property
    def variance_Q(self) -> np.ndarray:
        """Variance of Q across time (see averaged_Q)."""
        return self._ave_and_var_spec[1]["Q"]

    @property
    def averaged_spectrum(self) -> np.ndarray:
        """T* = T_noise * Q  + T_load."""
        return self.averaged_Q * self.t_load_ns + self.t_load

    @property
    def variance_spectrum(self) -> np.ndarray:
        """Variance of uncalibrated spectrum across time (see averaged_spectrum)."""
        return self.variance_Q * self.t_load_ns ** 2

    @property
    def ancillary(self) -> list[dict]:
        """Ancillary measurement data."""
        return [d.data["meta"] for d in self.spec_obj]

    @property
    def averaged_p0(self) -> np.ndarray:
        """Power of the load, averaged over time."""
        return self._ave_and_var_spec[0]["p0"]

    @property
    def averaged_p1(self) -> np.ndarray:
        """Power of the noise-source, averaged over time."""
        return self._ave_and_var_spec[0]["p1"]

    @property
    def averaged_p2(self) -> np.ndarray:
        """Power of the load plus noise-source, averaged over time."""
        return self._ave_and_var_spec[0]["p2"]

    @property
    def variance_p0(self) -> np.ndarray:
        """Variance of the load, averaged over time."""
        return self._ave_and_var_spec[1]["p0"]

    @property
    def variance_p1(self) -> np.ndarray:
        """Variance of the noise-source, averaged over time."""
        return self._ave_and_var_spec[1]["p1"]

    @property
    def variance_p2(self) -> np.ndarray:
        """Variance of the load plus noise-source, averaged over time."""
        return self._ave_and_var_spec[1]["p2"]

    @property
    def n_integrations(self) -> int:
        """The number of integrations recorded for the spectrum (after ignoring)."""
        return self._ave_and_var_spec[2]

    def _get_integrated_filename(self):
        """Determine a unique filename for the reduced data of this instance."""
        return self.cache_dir / f"{self.load_name}_{hash(self)}.h5"

    @cached_property
    def _ave_and_var_spec(self) -> tuple[dict, dict, int]:
        """Get the mean and variance of the spectra."""
        fname = self._get_integrated_filename()

        kinds = ["p0", "p1", "p2", "Q"]
        if fname.exists():
            logger.info(
                f"Reading in previously-created integrated {self.load_name} spectra..."
            )
            means = {}
            variances = {}
            n_integrations = 0
            with h5py.File(fname, "r") as fl:
                for kind in kinds:
                    means[kind] = fl[kind + "_mean"][...]
                    variances[kind] = fl[kind + "_var"][...]
                    n_integrations = fl.attrs.get("n_integrations", 0)
            return means, variances, n_integrations

        logger.info(f"Reducing {self.load_name} spectra...")
        spectra = self.get_spectra()

        means = {}
        variances = {}

        for key, spec in spectra.items():
            # Weird thing where there are zeros in the spectra.
            spec[spec == 0] = np.nan

            spec = tools.bin_array(spec.T, size=self.freq_bin_size).T

            mean = np.nanmean(spec, axis=1)
            var = np.nanvar(spec, axis=1)
            n_intg = spec.shape[1]

            if self.rfi_removal == "1D2D":
                nsample = np.sum(~np.isnan(spec), axis=1)

                width = max(1, self.rfi_kernel_width_freq // self.freq_bin_size)

                varfilt = xrfi.flagged_filter(var, size=2 * width + 1)
                resid = mean - xrfi.flagged_filter(mean, size=2 * width + 1)
                flags = np.logical_or(
                    resid > self.rfi_threshold * np.sqrt(varfilt / nsample),
                    var - varfilt
                    > self.rfi_threshold * np.sqrt(2 * varfilt ** 2 / (nsample - 1)),
                )

                mean[flags] = np.nan
                var[flags] = np.nan

            means[key] = mean
            variances[key] = var

        if not self.cache_dir.exists():
            self.cache_dir.mkdir()

        with h5py.File(fname, "w") as fl:
            logger.info(f"Saving reduced spectra to cache at {fname}")
            for kind in kinds:
                fl[kind + "_mean"] = means[kind]
                fl[kind + "_var"] = variances[kind]
            fl.attrs["n_integrations"] = n_intg

        return means, variances, n_intg

    def get_spectra(self) -> dict:
        """Read all spectra and remove RFI.

        Returns
        -------
        dict :
            A dictionary with keys being different powers (p1, p2, p3, Q), and values
            being ndarrays.
        """
        spec = self._read_spectrum()

        if self.rfi_removal == "2D":
            for key, val in spec.items():
                # Need to set nans and zeros to inf so that median/mean detrending
                # can work.
                val[np.isnan(val)] = np.inf

                if key != "Q":
                    val[val == 0] = np.inf

                flags, _ = xrfi.xrfi_medfilt(
                    val,
                    threshold=self.rfi_threshold,
                    kt=self.rfi_kernel_width_time,
                    kf=self.rfi_kernel_width_freq,
                )
                val[flags] = np.nan
                spec[key] = val
        return spec

    def _read_spectrum(self) -> dict:
        """
        Read the contents of the spectrum files into memory.

        Removes a starting percentage of times, and masks out certain frequencies.

        Returns
        -------
        dict :
            A dictionary of the contents of the file. Usually p0, p1, p2 (un-normalised
            powers of source, load, and load+noise respectively), and ant_temp (the
            uncalibrated, but normalised antenna temperature).
        """
        data = [spec_obj.data for spec_obj in self.spec_obj]

        n_times = sum(len(d["time_ancillary"]["times"]) for d in data)
        nfreq = np.sum(self.freq.mask)
        out = {
            "p0": np.empty((nfreq, n_times)),
            "p1": np.empty((nfreq, n_times)),
            "p2": np.empty((nfreq, n_times)),
            "Q": np.empty((nfreq, n_times)),
        }

        index_start_spectra = int((self.ignore_times_percent / 100) * n_times)
        for key, val in out.items():
            nn = 0
            for d in data:
                n = len(d["time_ancillary"]["times"])
                val[:, nn : (nn + n)] = d["spectra"][key][self.freq.mask]
                nn += n

            out[key] = val[:, index_start_spectra:]

        return out

    @cached_property
    def thermistor(self) -> np.ndarray:
        """The thermistor readings."""
        ary = self.resistance_obj.read()[0]

        return ary[int((self.ignore_times_percent / 100) * len(ary)) :]

    @cached_property
    def thermistor_temp(self):
        """The associated thermistor temperature in K."""
        return rcf.temperature_thermistor(self.thermistor["load_resistance"])

    @cached_property
    def temp_ave(self):
        """Average thermistor temperature (over time and frequency)."""
        return np.nanmean(self.thermistor_temp)

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
        path = Path(path or ".")

        # Allow to pass in a directory name *or* full path.
        if path.is_dir():
            path /= f"{self.load_name}_averaged_spectrum.h5"

        with h5py.File(path, "w") as fl:
            fl.attrs["load_name"] = self.load_name
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
        thermistor : bool
            Whether to plot the thermistor temperature on the same axis.
        fig : Figure
            Optionally, pass a matplotlib figure handle which will be used to plot.
        ax : Axis
            Optional, pass a matplotlib Axis handle which will be added to.
        xlabel : bool
            Whether to make an x-axis label.
        ylabel : bool
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


@attr.s(kw_only=True)
class HotLoadCorrection:
    """
    Corrections for the hot load.

    Measurements required to define the HotLoad temperature, from Monsalve et al.
    (2017), Eq. 8+9.

    Parameters
    ----------
    path
        Path to a file containing measurements of the semi-rigid cable reflection
        parameters. A preceding colon (:) indicates to prefix with DATA_PATH.
        The default file was measured in 2015, but there is also a file included
        that can be used from 2017: ":semi_rigid_s_parameters_2017.txt".
    f_low, f_high
        Lowest/highest frequency to retain from measurements.
    n_terms
        The number of terms used in fitting S-parameters of the cable.
    """

    _kinds = {"s11": 0, "s12": 1, "s22": 2}
    path: str | Path = attr.ib(
        default=":semi_rigid_s_parameters_WITH_HEADER.txt", converter=get_data_path
    )
    f_low: float | None = attr.ib(default=None)
    f_high: float | None = attr.ib(default=None)
    n_terms: int = attr.ib(default=21, converter=int)

    @cached_property
    def _data(self) -> np.ndarray:
        return np.genfromtxt(self.path)

    @cached_property
    def freq(self) -> FrequencyRange:
        """Frequencies of observation."""
        kwargs = {}
        if self.f_low is not None:
            kwargs["f_low"] = self.f_low
        if self.f_high is not None:
            kwargs["f_high"] = self.f_high

        return FrequencyRange(self._data[:, 0], **kwargs)

    @cached_property
    def data(self) -> np.ndarray:
        """The actual data."""
        if self._data.shape[1] == 7:  # Original file from 2015
            return (
                self._data[self.freq.mask, 1::2] + 1j * self._data[self.freq.mask, 2::2]
            )
        elif self._data.shape[1] == 6:  # File from 2017
            return np.array(
                [
                    self._data[self.freq.mask, 1] + 1j * self._data[self.freq.mask, 2],
                    self._data[self.freq.mask, 3],
                    self._data[self.freq.mask, 4] + 1j * self._data[self.freq.mask, 5],
                ]
            ).T
        else:
            raise OSError("Semi-Rigid Cable file has wrong data format.")

    def _get_model_kind(self, kind):
        model = mdl.Polynomial(
            n_terms=self.n_terms,
            transform=mdl.UnitTransform(range=(self.freq.min, self.freq.max)),
        )
        model = mdl.ComplexMagPhaseModel(mag=model, phs=model)
        return model.fit(xdata=self.freq.freq, ydata=self.data[:, self._kinds[kind]])

    @cached_property
    def s11_model(self):
        """The reflection coefficient."""
        return self._get_model_kind("s11")

    @cached_property
    def s12_model(self):
        """The transmission coefficient."""
        return self._get_model_kind("s12")

    @cached_property
    def s22_model(self):
        """The reflection coefficient from the other side."""
        return self._get_model_kind("s22")

    def power_gain(self, freq: np.ndarray, hot_load_s11: LoadS11) -> np.ndarray:
        """
        Calculate the power gain.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies.
        hot_load_s11 : :class:`LoadS11`
            The S11 of the hot load.

        Returns
        -------
        gain : np.ndarray
            The power gain as a function of frequency.
        """
        assert isinstance(
            hot_load_s11, LoadS11
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
    def get_power_gain(
        semi_rigid_sparams: dict, hot_load_s11: np.ndarray
    ) -> np.ndarray:
        """Define Eq. 9 from M17.

        Parameters
        ----------
        semi_rigid_sparams : dict
            A dictionary of reflection coefficient measurements as a function of
            frequency for the semi-rigid cable.
        hot_load_s11 : array-like
            The S11 measurement of the hot_load.

        Returns
        -------
        gain : np.ndarray
            The power gain.
        """
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


@attr.s
class Load:
    """Wrapper class containing all relevant information for a given load.

    Parameters
    ----------
    spectrum : :class:`LoadSpectrum`
        The spectrum for this particular load.
    reflections : :class:`SwitchCorrection`
        The S11 measurements for this particular load.
    hot_load_correction : :class:`HotLoadCorrection`
        If this is a hot load, provide a hot load correction.
    ambient : :class:`LoadSpectrum`
        If this is a hot load, need to provide an ambient spectrum to correct it.
    """

    spectrum: LoadSpectrum = attr.ib()
    reflections: LoadS11 = attr.ib()
    hot_load_correction: HotLoadCorrection | None = attr.ib(default=None)
    ambient: LoadSpectrum | None = attr.ib(default=None)

    @reflections.validator
    def _rfl_vld(self, att, val):
        if val.load_name != self.spectrum.load_name:
            raise ValueError("LoadS11 and LoadSpectrum must have the same name!")

    @property
    def load_name(self) -> str:
        """The name of the load."""
        return self.spectrum.load_name

    @property
    def t_load(self) -> float:
        """Assumed temperature of the load."""
        return self.spectrum.t_load

    @property
    def t_load_ns(self) -> float:
        """Assumed temperature of the load + noise source."""
        return self.spectrum.t_load_ns

    @hot_load_correction.validator
    def _hlc_validator(self, att, val):
        if self.load_name == "hot_load" and val is None:
            raise ValueError(
                "You must provide a hot_load_correction to construct the hot_load Load"
            )

    @ambient.validator
    def _amb_validator(self, att, val):
        if self.load_name == "hot_load" and val is None:
            raise ValueError("You must provide ambient to construct the hot_load Load")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        load_name: str,
        f_low: float | attr.NOTHING = attr.NOTHING,
        f_high: float | attr.NOTHING = attr.NOTHING,
        reflection_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
    ):
        """
        Define a full :class:`Load` from a path and name.

        Parameters
        ----------
        path : str or Path
            Path to the top-level calibration observation.
        load_name : str
            Name of a load to define.
        f_low, f_high : float
            Min/max frequencies to keep in measurements.
        reflection_kwargs : dict
            Extra arguments to pass through to :class:`SwitchCorrection`.
        spec_kwargs : dict
            Extra arguments to pass through to :class:`LoadSpectrum`.

        Returns
        -------
        load : :class:`Load`
            The load object, containing all info about spectra and S11's for that load.
        """
        if not spec_kwargs:
            spec_kwargs = {}
        if not reflection_kwargs:
            reflection_kwargs = {}

        spec = LoadSpectrum.from_load_name(
            load_name,
            path,
            f_low=f_low,
            f_high=f_high,
            **spec_kwargs,
        )

        refl = LoadS11.from_path(
            load_name,
            path,
            f_low=f_low,
            f_high=f_high,
            **reflection_kwargs,
        )

        if load_name == "hot_load":
            hlc = HotLoadCorrection()
            amb = LoadSpectrum.from_load_name(
                "ambient", path, f_low=f_low, f_high=f_high, **spec_kwargs
            )
            return cls(spec, refl, hot_load_correction=hlc, ambient=amb)
        else:
            return cls(spec, refl)

    @property
    def s11_model(self):
        """The S11 model."""
        return self.reflections.s11_model

    @cached_property
    def temp_ave(self) -> np.ndarray:
        """The average temperature of the thermistor (over frequency and time)."""
        if self.load_name != "hot_load":
            return self.spectrum.temp_ave

        gain = self.hot_load_correction.power_gain(self.freq.freq, self.reflections)
        # temperature
        return gain * self.spectrum.temp_ave + (1 - gain) * self.ambient.temp_ave

    @property
    def averaged_Q(self) -> np.ndarray:
        """Averaged power ratio."""
        return self.spectrum.averaged_Q

    @property
    def averaged_spectrum(self) -> np.ndarray:
        """Averaged uncalibrated temperature."""
        return self.spectrum.averaged_spectrum

    @property
    def freq(self) -> FrequencyRange:
        """A :class:`FrequencyRange` object corresponding to this measurement."""
        return self.spectrum.freq


@attr.s
class CalibrationObservation:
    """
    A composite object representing a full Calibration Observation.

    This includes spectra of all calibrators, and methods to find the calibration
    parameters. It strictly follows Monsalve et al. (2017) in its formalism.
    While by default the class uses the calibrator sources ("ambient", "hot_load",
    "open", "short"), it can be modified to take other sources by setting
    ``CalibrationObservation._sources`` to a new tuple of strings.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing all relevant measurements. It is assumed
        that in this directory is an `S11`, `Resistance` and `Spectra` directory.
    semi_rigid_path : str or Path, optional
        Path to a file containing S11 measurements for the semi rigid cable. Used to
        correct the hot load S11. Found automatically if not given.
    ambient_temp : int
        Ambient temperature (C) at which measurements were taken.
    f_low : float
        Minimum frequency to keep for all loads (and their S11's). If for some
        reason different frequency bounds are desired per-load, one can pass in
        full load objects through ``load_spectra``.
    f_high : float
        Maximum frequency to keep for all loads (and their S11's). If for some
        reason different frequency bounds are desired per-load, one can pass in
        full load objects through ``load_spectra``.
    run_num : int or dict
        Which run number to use for the calibrators. Default is to use the last run
        for each. Passing an int will attempt to use that run for each source. Pass
        a dict mapping sources to numbers to use different combinations.
    repeat_num : int or dict
        Which repeat number to use for the calibrators. Default is to use the last
        repeat for each. Passing an int will attempt to use that repeat for each
        source. Pass a dict mapping sources to numbers to use different
        combinations.
    resistance_f : float
        Female resistance (Ohms). Used for the LNA S11.
    cterms : int
        The number of terms to use for the polynomial fits to the calibration
        functions.
    wterms : int
        The number of terms to use for the polynomial fits to the noise-wave
        calibration functions.
    load_kwargs : dict
        Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
        objects. See its documentation for relevant parameters. Parameters specified
        here are used for _all_ calibrator sources.
    s11_kwargs : dict
        Keyword arguments used to instantiate the calibrator :class:`LoadS11`
        objects. See its documentation for relevant parameters. Parameters specified
        here are used for _all_ calibrator sources.
    load_spectra : dict
        A dictionary mapping load names of calibration sources (eg. ambient, short)
        to either :class:`LoadSpectrum` instances or dictionaries of keywords to
        instantiate those objects. Useful for individually specifying
        properties of each load separately. Values in these dictionaries (if
        supplied) over-ride those given in ``load_kwargs`` (but values in
        ``load_kwargs`` are still used if not over-ridden).
    load_s11s : dict
        A dictionary mapping load names of calibration sources (eg. ambient, short)
        to :class:`LoadS11` instances or dictionaries of keywords to instantiate
        those objects. Useful for individually specifying properties of each load
        separately. Values in these dictionaries (if  supplied) over-ride those
        given in ``s11_kwargs`` (but values in ``s11_kwargs`` are still used if not
        over-ridden).
    compile_from_def : bool
        Whether to attempt compiling a virtual observation from a
        ``definition.yaml`` inside the observation directory. This is the default
        behaviour, but can be turned off to enforce that the current directory
        should be used directly.
    include_previous : bool
        Whether to include the previous observation by default to supplement this
        one if required files are missing.
    freq_bin_size
        The size of each frequency bin (of the spectra) in units of the raw size.

    Examples
    --------
    This will setup an observation with all default options applied:

    >>> path = '/CalibrationObservations/Receiver01_25C_2019_11_26_040_to_200MHz'
    >>> calobs = CalibrationObservation(path)

    To specify some options for constructing the various calibrator load spectra:

    >>> calobs = CalibrationObservation(
    >>>     path,
    >>>    load_kwargs={"cache_dir":".", "ignore_times_percent": 50}
    >>> )

    But if we typically wanted 50% of times ignored, but in one special case we'd
    like 80%:

    >>> calobs = CalibrationObservation(
    >>>     path,
    >>>     load_kwargs={"cache_dir":".", "ignore_times_percent": 50},
    >>>     load_spectra={"short": {"ignore_times_percent": 80}}
    >>> )

    """

    _sources = ("ambient", "hot_load", "open", "short")

    _path: tp.PathLike = attr.ib()
    semi_rigid_path: tp.PathLike = attr.ib(
        default=":semi_rigid_s_parameters_WITH_HEADER.txt", kw_only=True
    )
    f_low: float = attr.ib(default=40.0, kw_only=True, converter=float)
    f_high: float | None = attr.ib(
        default=None, kw_only=True, converter=attr.converters.optional(float)
    )
    run_num: int | dict = attr.ib(default=attr.Factory(dict), kw_only=True)
    repeat_num: int | dict = attr.ib(default=attr.Factory(dict), kw_only=True)
    resistance_f: float | None = attr.ib(default=None, kw_only=True)
    cterms: int = attr.ib(default=5, kw_only=True)
    wterms: int = attr.ib(default=7, kw_only=True)
    load_kwargs: dict = attr.ib(default=attr.Factory(dict), kw_only=True)
    s11_kwargs: dict = attr.ib(default=attr.Factory(dict), kw_only=True)
    load_spectra: dict[str, LoadSpectrum | dict] = attr.ib(
        default=attr.Factory(dict), kw_only=True
    )
    load_s11s: dict = attr.ib(default=attr.Factory(dict), kw_only=True)
    compile_from_def: bool = attr.ib(default=True, kw_only=True)
    include_previous: bool = attr.ib(default=False, kw_only=True)
    internal_switch_kwargs: dict[str, Any] | None = attr.ib(
        default=attr.Factory(dict), kw_only=True
    )
    freq_bin_size: int = attr.ib(default=1, kw_only=True)

    @f_high.validator
    def _fh_vld(self, att, val):
        if val is not None and val < self.f_low:
            raise ValueError("f_high must be larger than f_low!")

    @load_spectra.validator
    def _ls_vld(self, att, val):
        if not isinstance(val, dict):
            raise TypeError("load_spectra must be a dict")
        for k, v in val.items():
            if not isinstance(v, (dict, LoadSpectrum)):
                raise TypeError("values in load_spectra must be dicts or LoadSpectrum")

    @load_spectra.validator
    def _load_spec_vld(self, att, val):
        if any(name not in self._sources for name in val):
            raise ValueError(
                f"Can't specify load_spectra with names: {list(val.keys())}"
            )

    @load_s11s.validator
    def _load_s11s_vld(self, att, val):
        if any(name not in self._sources and name not in ["lna"] for name in val):
            raise ValueError(f"Can't specify load_s11s with names: {list(val.keys())}")

    @cached_property
    def io(self) -> io.CalibrationObservation:
        """The underlying io-based data."""
        if self.compile_from_def:
            return io.CalibrationObservation.from_def(
                self._path,
                run_num=self.run_num,
                repeat_num=self.repeat_num,
                include_previous=self.include_previous,
            )
        else:
            return io.CalibrationObservation(
                self._path,
                run_num=self.run_num,
                repeat_num=self.repeat_num,
            )

    @property
    def compiled_from_def(self) -> bool:
        """Alias for compile_from_def."""
        return self.compile_from_def

    @property
    def previous_included(self) -> bool:
        """Alias for include previous."""
        return self.include_previous

    @safe_property
    def path(self) -> Path:
        """The actual path to the observation (even if virtual)."""
        return self.io.path

    @cached_property
    def internal_switch(self) -> s11.InternalSwitch:
        """The internal switch measurements."""
        kwargs = {
            **self.internal_switch_kwargs,
            **{
                "resistance": self.io.definition["measurements"]["resistance_m"][
                    self.io.s11.switching_state.run_num
                ]
            },
        }
        return s11.InternalSwitch(data=self.io.s11.switching_state, **kwargs)

    @cached_property
    def hot_load_correction(self) -> HotLoadCorrection:
        """The hot load correction used for the hot load."""
        return HotLoadCorrection(
            path=self.semi_rigid_path, f_low=self.f_low, f_high=self.f_high
        )

    @cached_property
    def _loads(self) -> dict[str, Load]:
        try:
            loads = {}
            for source in self._sources:
                load = self.load_spectra.get(source, {})

                if isinstance(load, dict):
                    load = LoadSpectrum(
                        spec_obj=getattr(self.io.spectra, source),
                        resistance_obj=getattr(self.io.resistance, source),
                        f_low=self.f_low,
                        f_high=self.f_high,
                        freq_bin_size=self.freq_bin_size,
                        **{**self.load_kwargs, **load},
                    )

                # Ensure that we finally have a LoadSpectrum
                if not isinstance(load, LoadSpectrum):
                    raise TypeError(
                        "load_spectra must be a dict of LoadSpectrum or dicts."
                    )

                refl = self.load_s11s.get(source, {})

                if isinstance(refl, dict):
                    refl = LoadS11(
                        load_s11=getattr(self.io.s11, source),
                        internal_switch=self.internal_switch,
                        f_low=self.f_low,
                        f_high=self.f_high,
                        **{**self.s11_kwargs, **refl},
                    )

                if source == "hot_load":
                    loads[source] = Load(
                        load,
                        refl,
                        hot_load_correction=self.hot_load_correction,
                        ambient=loads["ambient"].spectrum,
                    )
                else:
                    loads[source] = Load(load, refl)

            # We must use the most restricted frequency range available from all
            # available sources as well as the LNA.
            fmin = max(
                sum(
                    (
                        [load.spectrum.freq._f_low, load.reflections.freq._f_low]
                        for load in loads.values()
                    ),
                    [],
                )
                + [self.lna.freq._f_low]
            )

            fmax = min(
                sum(
                    (
                        [load.spectrum.freq._f_high, load.reflections.freq._f_high]
                        for load in loads.values()
                    ),
                    [],
                )
                + [self.lna.freq._f_high]
            )

            if fmax <= fmin:
                raise ValueError(
                    "The inputs loads and S11s have non-overlapping frequency ranges!"
                )

            # Now make everything actually consistent in its frequency range.
            new_loads = {}
            for name, load in loads.items():
                new_loads[name] = attr.evolve(
                    load, spectrum=attr.evolve(load.spectrum, f_low=fmin, f_high=fmax)
                )
        except AttributeError as e:
            raise RuntimeError(str(e))

        return new_loads

    def __getattr__(self, name):
        """Get loads as attributes."""
        if name in self._sources:
            return self._loads[name]

        raise AttributeError(f"{name} does not exist in CalibrationObservation!")

    @cached_property
    def lna(self) -> LNA:
        """The LNA measurements."""
        refl = self.load_s11s.get("lna", {})

        return LNA(
            load_s11=self.io.s11.receiver_reading,
            f_low=self.f_low,
            f_high=self.f_high,
            resistance=self.resistance_f
            or self.io.definition["measurements"]["resistance_f"][
                self.io.s11.receiver_reading.run_num
            ],
            **{**self.s11_kwargs, **refl},
        )

    @cached_property
    def t_load(self) -> float:
        """Assumed temperature of the load."""
        return self._loads[list(self._loads.keys())[0]].t_load

    @cached_property
    def t_load_ns(self) -> float:
        """Assumed temperature of the load + noise source."""
        return self._loads[list(self._loads.keys())[0]].t_load_ns

    @cached_property
    def freq(self) -> FrequencyRange:
        """The frequencies at which spectra were measured."""
        return self._loads[list(self._loads.keys())[0]].spectrum.freq

    @safe_property
    def load_names(self) -> tuple[str]:
        """Names of the loads."""
        return tuple(self._loads.keys())

    def new_load(
        self,
        load_name: str,
        run_num: int = 1,
        reflection_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
    ):
        """Create a new load with the given load name.

        Uses files inside the current observation.

        Parameters
        ----------
        load_name : str
            The name of the load
        run_num_spec : dict or int
            Run number to use for the spectrum.
        run_num_load : dict or int
            Run number to use for the load's S11.
        reflection_kwargs : dict
            Keyword arguments to construct the :class:`SwitchCorrection`.
        spec_kwargs : dict
            Keyword arguments to construct the :class:`LoadSpectrum`.
        """
        reflection_kwargs = reflection_kwargs or {}
        spec_kwargs = spec_kwargs or {}

        # Fill up kwargs with keywords from this instance
        if "resistance" not in reflection_kwargs:
            reflection_kwargs[
                "resistance"
            ] = self.open.reflections.internal_switch.resistance

        for key in [
            "ignore_times_percent",
            "rfi_removal",
            "rfi_kernel_width_freq",
            "rfi_kernel_width_time",
            "rfi_threshold",
            "cache_dir",
            "t_load",
            "t_load_ns",
        ]:
            if key not in spec_kwargs:
                spec_kwargs[key] = getattr(self.open.spectrum, key)

        reflection_kwargs["run_num_load"] = run_num
        reflection_kwargs["repeat_num_switch"] = self.io.s11.switching_state.repeat_num
        reflection_kwargs["run_num_switch"] = self.io.s11.switching_state.run_num
        spec_kwargs["run_num"] = run_num
        spec_kwargs["freq_bin_size"] = self.freq.bin_size

        return Load.from_path(
            path=self.io.path,
            load_name=load_name,
            f_low=self.freq._f_low,
            f_high=self.freq._f_high,
            reflection_kwargs=reflection_kwargs,
            spec_kwargs=spec_kwargs,
        )

    def plot_raw_spectra(self, fig=None, ax=None) -> plt.Figure:
        """
        Plot raw uncalibrated spectra for all calibrator sources.

        Parameters
        ----------
        fig : :class:`plt.Figure`
            A matplotlib figure on which to make the plot. By default creates a new one.
        ax : :class:`plt.Axes`
            A matplotlib Axes on which to make the plot. By default creates a new one.

        Returns
        -------
        fig : :class:`plt.Figure`
            The figure on which the plot was made.
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

    def plot_s11_models(self, **kwargs):
        """
        Plot residuals of S11 models for all sources.

        Returns
        -------
        dict:
            Each entry has a key of the source name, and the value is a matplotlib fig.
        """
        out = {
            name: source.reflections.plot_residuals(**kwargs)
            for name, source in self._loads.items()
        }
        out.update({"lna": self.lna.plot_residuals(**kwargs)})
        return out

    @cached_property
    def s11_correction_models(self):
        """Dictionary of S11 correction models, one for each source."""
        try:
            return dict(self._injected_source_s11s)
        except (TypeError, AttributeError):
            return {
                name: source.s11_model(self.freq.freq)
                for name, source in self._loads.items()
            }

    @cached_property
    def source_thermistor_temps(self) -> dict[str, float | np.ndarray]:
        """Dictionary of input source thermistor temperatures."""
        if (
            hasattr(self, "_injected_source_temps")
            and self._injected_source_temps is not None
        ):
            return self._injected_source_temps

        return {k: source.temp_ave for k, source in self._loads.items()}

    @cached_property
    def _calibration_coefficients(self):
        """The calibration polynomials, evaluated at `freq.freq`."""
        if (
            hasattr(self, "_injected_averaged_spectra")
            and self._injected_averaged_spectra is not None
        ):
            ave_spec = self._injected_averaged_spectra
        else:
            ave_spec = {
                k: source.averaged_spectrum for k, source in self._loads.items()
            }
        scale, off, Tu, TC, TS = rcf.get_calibration_quantities_iterative(
            self.freq.freq_recentred,
            temp_raw=ave_spec,
            gamma_rec=self.lna_s11,
            gamma_ant=self.s11_correction_models,
            temp_ant=self.source_thermistor_temps,
            cterms=self.cterms,
            wterms=self.wterms,
            temp_amb_internal=self.t_load,
        )
        return scale, off, Tu, TC, TS

    @cached_property
    def C1_poly(self):  # noqa: N802
        """`np.poly1d` object describing the Scaling calibration coefficient C1.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~C1` as a direct
        function on frequency.
        """
        return self._calibration_coefficients[0]

    @cached_property
    def C2_poly(self):  # noqa: N802
        """`np.poly1d` object describing the offset calibration coefficient C2.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~C2` as a direct
        function on frequency.
        """
        return self._calibration_coefficients[1]

    @cached_property
    def Tunc_poly(self):  # noqa: N802
        """`np.poly1d` object describing the uncorrelated noise-wave parameter, Tunc.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~Tunc` as a direct
        function on frequency.
        """
        return self._calibration_coefficients[2]

    @cached_property
    def Tcos_poly(self):  # noqa: N802
        """`np.poly1d` object describing the cosine noise-wave parameter, Tcos.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~Tcos` as a direct
        function on frequency.
        """
        return self._calibration_coefficients[3]

    @cached_property
    def Tsin_poly(self):  # noqa: N802
        """`np.poly1d` object describing the sine noise-wave parameter, Tsin.

        The polynomial is defined to act on normalized frequencies such that `freq.min`
        and `freq.max` map to -1 and 1 respectively. Use :func:`~Tsin` as a direct
        function on frequency.
        """
        return self._calibration_coefficients[4]

    def C1(self, f: float | np.ndarray | None = None):  # noqa: N802
        """
        Scaling calibration parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate C1. By default, the frequencies of this
            instance.
        """
        if hasattr(self, "_injected_c1") and self._injected_c1 is not None:
            return np.array(self._injected_c1)
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.C1_poly(fnorm)

    def C2(self, f: float | np.ndarray | None = None):  # noqa: N802
        """
        Offset calibration parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate C2. By default, the frequencies of this
            instance.
        """
        if hasattr(self, "_injected_c2") and self._injected_c2 is not None:
            return np.array(self._injected_c2)
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.C2_poly(fnorm)

    def Tunc(self, f: float | np.ndarray | None = None):  # noqa: N802
        """
        Uncorrelated noise-wave parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate Tunc. By default, the frequencies of
            thisinstance.
        """
        if hasattr(self, "_injected_t_unc") and self._injected_t_unc is not None:
            return np.array(self._injected_t_unc)
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.Tunc_poly(fnorm)

    def Tcos(self, f: float | np.ndarray | None = None):  # noqa: N802
        """
        Cosine noise-wave parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate Tcos. By default, the frequencies of
            this instance.
        """
        if hasattr(self, "_injected_t_cos") and self._injected_t_cos is not None:
            return np.array(self._injected_t_cos)
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.Tcos_poly(fnorm)

    def Tsin(self, f: float | np.ndarray | None = None):  # noqa: N802
        """
        Sine noise-wave parameter.

        Parameters
        ----------
        f : array-like
            The frequencies at which to evaluate Tsin. By default, the frequencies of
            this instance.
        """
        if hasattr(self, "_injected_t_sin") and self._injected_t_sin is not None:
            return np.array(self._injected_t_sin)
        fnorm = self.freq.freq_recentred if f is None else self.freq.normalize(f)
        return self.Tsin_poly(fnorm)

    @cached_property
    def lna_s11(self):
        """The corrected S11 of the LNA evaluated at the data frequencies."""
        if hasattr(self, "_injected_lna_s11") and self._injected_lna_s11 is not None:
            return self._injected_lna_s11
        else:
            return self.lna.s11_model(self.freq.freq)

    def get_linear_coefficients(self, load: Load | str):
        """
        Calibration coefficients a,b such that T = aT* + b (derived from Eq. 7).

        Parameters
        ----------
        load : str or :class:`Load`
            The load for which to get the linear coefficients.
        """
        if isinstance(load, str):
            load_s11 = self.s11_correction_models[load]
        elif load.load_name in self.s11_correction_models:
            load_s11 = self.s11_correction_models[load.load_name]
        else:
            load_s11 = load.s11_model(self.freq.freq)

        return rcf.get_linear_coefficients(
            load_s11,
            self.lna_s11,
            self.C1(self.freq.freq),
            self.C2(self.freq.freq),
            self.Tunc(self.freq.freq),
            self.Tcos(self.freq.freq),
            self.Tsin(self.freq.freq),
            t_load=self.t_load,
        )

    def calibrate(self, load: Load | str, q=None, temp=None):
        """
        Calibrate the temperature of a given load.

        Parameters
        ----------
        load : :class:`Load` or str
            The load to calibrate.

        Returns
        -------
        array : calibrated antenna temperature in K, len(f).
        """
        load = self._load_str_to_load(load)
        a, b = self.get_linear_coefficients(load)

        if q is not None:
            temp = self.t_load_ns * q + self.t_load
        elif temp is None:
            temp = load.averaged_spectrum

        return a * temp + b

    def _load_str_to_load(self, load: Load | str):
        if isinstance(load, str):
            try:
                load = self._loads[load]
            except AttributeError:
                raise AttributeError(
                    "load must be a Load object or a string (one of "
                    "{ambient,hot_load,open,short})"
                )
        else:
            assert isinstance(
                load, Load
            ), f"load must be a Load instance, got the {load} {type(Load)}"
        return load

    def decalibrate(self, temp: np.ndarray, load: Load | str, freq: np.ndarray = None):
        """
        Decalibrate a temperature spectrum, yielding uncalibrated T*.

        Parameters
        ----------
        temp : array_like
            A temperature spectrum, with the same length as `freq.freq`.
        load : str or :class:`Load`
            The load to calibrate.
        freq : array-like
            The frequencies at which to decalibrate. By default, the frequencies of the
            instance.

        Returns
        -------
        array_like : T*, the normalised uncalibrated temperature.
        """
        if freq is None:
            freq = self.freq.freq

        if freq.min() < self.freq.freq.min():
            warnings.warn(
                "The minimum frequency is outside the calibrated range "
                f"({self.freq.freq.min()} - {self.freq.freq.max()} MHz)"
            )

        if freq.min() > self.freq.freq.max():
            warnings.warn("The maximum frequency is outside the calibrated range ")

        a, b = self.get_linear_coefficients(load)
        return (temp - b) / a

    def get_K(
        self, freq: np.ndarray | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the source-S11-dependent factors of Monsalve (2017) Eq. 7."""
        if freq is None:
            freq = self.freq.freq
            gamma_ants = self.s11_correction_models
        else:
            gamma_ants = {
                name: source.s11_model(freq) for name, source in self._loads.items()
            }

        lna_s11 = self.lna.s11_model(freq)
        return {
            name: rcf.get_K(gamma_rec=lna_s11, gamma_ant=gamma_ant)
            for name, gamma_ant in gamma_ants.items()
        }

    def plot_calibrated_temp(
        self,
        load: Load | str,
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
        bins : int
            Number of bins to smooth over (std of Gaussian kernel)
        fig : Figure
            Optionally provide a matplotlib figure to add to.
        ax : Axis
            Optionally provide a matplotlib Axis to add to.
        xlabel : bool
            Whether to write the x-axis label
        ylabel : bool
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
        freq_ave_cal[np.isinf(freq_ave_cal)] = np.nan

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        ax.plot(
            self.freq.freq,
            freq_ave_cal,
            label=f"Calibrated {load.spectrum.load_name} [RMS = {rms:.3f}]",
        )

        temp_ave = self.source_thermistor_temps.get(load.load_name, load.temp_ave)

        if not hasattr(temp_ave, "__len__"):
            ax.axhline(temp_ave, color="C2", label="Average thermistor temp")
        else:
            ax.plot(
                self.freq.freq,
                temp_ave,
                color="C2",
                label="Average thermistor temp",
            )

        ax.set_ylim([np.nanmin(freq_ave_cal), np.nanmax(freq_ave_cal)])
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
            true = self.source_thermistor_temps[source]
            out[source] = cal - true
        return out

    def get_rms(self, smooth: int = 4):
        """Return a dict of RMS values for each source.

        Parameters
        ----------
        smooth : int
            The number of bins over which to smooth residuals before taking the RMS.
        """
        resids = self.get_load_residuals()
        out = {}
        for name, res in resids.items():
            if smooth > 1:
                res = convolve(res, Gaussian1DKernel(stddev=smooth), boundary="extend")
            out[name] = np.sqrt(np.nanmean(res ** 2))
        return out

    def plot_calibrated_temps(self, bins=64, fig=None, ax=None):
        """
        Plot all calibrated temperatures in a single figure.

        Parameters
        ----------
        bins : int
            Number of bins in the smoothed spectrum

        Returns
        -------
        fig :
            Matplotlib figure that was created.
        """
        if fig is None or ax is None or len(ax) != len(self._sources):
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

    def write_coefficients(self, path: str | None = None):
        """
        Save a text file with the derived calibration co-efficients.

        Parameters
        ----------
        path : str
            Directory in which to write the file. The filename starts with
            `All_cal-params` and includes parameters of the class in the filename.
            By default, current directory.
        """
        path = Path(path or ".")

        if path.is_dir():
            path /= (
                f"calibration_parameters_fmin{self.freq.freq.min()}_"
                f"fmax{self.freq.freq.max()}_C{self.cterms}_W{self.wterms}.txt"
            )

        np.savetxt(
            path,
            [
                self.freq.freq,
                self.C1(),
                self.C2(),
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
        fig : Figure
            Optionally pass a matplotlib figure to add to.
        ax : Axis
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

    def clone(self, **kwargs):
        """Clone the instance, updating some parameters.

        Parameters
        ----------
        kwargs :
            All parameters to be updated.
        """
        return attr.evolve(self, **kwargs)

    def write(self, filename: str | Path):
        """
        Write all information required to calibrate a new spectrum to file.

        Parameters
        ----------
        filename : path
            The filename to write to.
        """
        with h5py.File(filename, "w") as fl:
            # Write attributes
            fl.attrs["path"] = str(self.io.original_path)
            fl.attrs["cterms"] = self.cterms
            fl.attrs["wterms"] = self.wterms
            fl.attrs["switch_path"] = str(self.internal_switch.data.path)
            fl.attrs["switch_repeat_num"] = self.internal_switch.data.repeat_num
            fl.attrs["switch_resistance"] = self.internal_switch.resistance
            fl.attrs["switch_nterms"] = self.internal_switch.n_terms[0]
            fl.attrs["switch_model"] = str(self.internal_switch.model)
            fl.attrs["t_load"] = self.open.spectrum.t_load
            fl.attrs["t_load_ns"] = self.open.spectrum.t_load_ns

            fl["C1"] = self.C1_poly.coefficients
            fl["C2"] = self.C2_poly.coefficients
            fl["Tunc"] = self.Tunc_poly.coefficients
            fl["Tcos"] = self.Tcos_poly.coefficients
            fl["Tsin"] = self.Tsin_poly.coefficients
            fl["frequencies"] = self.freq.freq
            fl["lna_s11_real"] = self.lna.s11_model(self.freq.freq).real
            fl["lna_s11_imag"] = self.lna.s11_model(self.freq.freq).imag

            fl["internal_switch_s11_real"] = np.real(
                self.internal_switch.s11_model(self.freq.freq)
            )
            fl["internal_switch_s11_imag"] = np.imag(
                self.internal_switch.s11_model(self.freq.freq)
            )
            fl["internal_switch_s12_real"] = np.real(
                self.internal_switch.s12_model(self.freq.freq)
            )
            fl["internal_switch_s12_imag"] = np.imag(
                self.internal_switch.s12_model(self.freq.freq)
            )
            fl["internal_switch_s22_real"] = np.real(
                self.internal_switch.s22_model(self.freq.freq)
            )
            fl["internal_switch_s22_imag"] = np.imag(
                self.internal_switch.s22_model(self.freq.freq)
            )

            load_grp = fl.create_group("loads")

            for name, load in self._loads.items():
                grp = load_grp.create_group(name)
                grp.attrs["s11_model"] = yaml.dump(load.s11_model)
                grp["averaged_Q"] = load.spectrum.averaged_Q
                grp["variance_Q"] = load.spectrum.variance_Q
                grp["temp_ave"] = load.temp_ave
                grp.attrs["n_integrations"] = load.spectrum.n_integrations

    def to_calfile(self):
        """Directly create a :class:`Calibration` object without writing to file."""
        return Calibration.from_calobs(self)

    def inject(
        self,
        lna_s11: np.ndarray = None,
        source_s11s: dict[str, np.ndarray] = None,
        c1: np.ndarray = None,
        c2: np.ndarray = None,
        t_unc: np.ndarray = None,
        t_cos: np.ndarray = None,
        t_sin: np.ndarray = None,
        averaged_spectra: dict[str, np.ndarray] = None,
        thermistor_temp_ave: dict[str, np.ndarray] = None,
    ) -> CalibrationObservation:
        """Make a new :class:`CalibrationObservation` based on this, with injections.

        Parameters
        ----------
        lna_s11
            The LNA S11 as a function of frequency to inject.
        source_s11s
            Dictionary of ``{source: S11}`` for each source to inject.
        c1
            Scaling parameter as a function of frequency to inject.
        c2 : [type], optional
            Offset parameter to inject as a function of frequency.
        t_unc
            Uncorrelated temperature to inject (as function of frequency)
        t_cos
            Correlated temperature to inject (as function of frequency)
        t_sin
            Correlated temperature to inject (as function of frequency)
        averaged_spectra
            Dictionary of ``{source: spectrum}`` for each source to inject.

        Returns
        -------
        :class:`CalibrationObservation`
            A new observation object with the injected models.
        """
        new = copy(self)
        new.invalidate_cache()
        new._injected_lna_s11 = lna_s11
        new._injected_source_s11s = source_s11s
        new._injected_c1 = c1
        new._injected_c2 = c2
        new._injected_t_unc = t_unc
        new._injected_t_cos = t_cos
        new._injected_t_sin = t_sin
        new._injected_averaged_spectra = averaged_spectra
        new._injected_source_temps = thermistor_temp_ave

        return new


@attr.s
class _LittleS11:
    s11_model: Callable = attr.ib()


@attr.s
class _LittleSpectrum:
    averaged_Q: np.ndarray = attr.ib()
    variance_Q: np.ndarray = attr.ib()
    n_integrations: int = attr.ib()


@attr.s
class _LittleLoad:
    reflections: _LittleS11 = attr.ib()
    spectrum: _LittleSpectrum = attr.ib()
    temp_ave: np.ndarray = attr.ib()

    def s11_model(self, freq):
        return self.reflections.s11_model(freq)


class Calibration:
    def __init__(self, filename: str | Path):
        """
        A class defining an interface to a HDF5 file containing calibration information.

        Parameters
        ----------
        filename : str or Path
            The path to the calibration file.
        """
        self.calfile = Path(filename)

        with h5py.File(filename, "r") as fl:
            self.calobs_path = fl.attrs["path"]
            self.cterms = int(fl.attrs["cterms"])
            self.wterms = int(fl.attrs["wterms"])
            self.t_load = fl.attrs.get("t_load", 300)
            self.t_load_ns = fl.attrs.get("t_load_ns", 400)

            self.C1_poly = np.poly1d(fl["C1"][...])
            self.C2_poly = np.poly1d(fl["C2"][...])
            self.Tcos_poly = np.poly1d(fl["Tcos"][...])
            self.Tsin_poly = np.poly1d(fl["Tsin"][...])
            self.Tunc_poly = np.poly1d(fl["Tunc"][...])

            self.freq = FrequencyRange(fl["frequencies"][...])

            self._loads = {}
            if "loads" in fl:
                lg = fl["loads"]

                self.load_names = list(lg.keys())

                for name, grp in lg.items():
                    self._loads[name] = _LittleLoad(
                        reflections=_LittleS11(
                            s11_model=yaml.load(
                                grp.attrs["s11_model"], Loader=yaml.FullLoader
                            ).at(x=self.freq.freq)
                        ),
                        spectrum=_LittleSpectrum(
                            averaged_Q=grp["averaged_Q"][...],
                            variance_Q=grp["variance_Q"][...],
                            n_integrations=grp.attrs["n_integrations"],
                        ),
                        temp_ave=grp["temp_ave"][...],
                    )
                    setattr(self, name, self._loads[name])

            self._lna_s11_rl = Spline(self.freq.freq, fl["lna_s11_real"][...])
            self._lna_s11_im = Spline(self.freq.freq, fl["lna_s11_imag"][...])

            self._intsw_s11_rl = Spline(
                self.freq.freq, fl["internal_switch_s11_real"][...]
            )
            self._intsw_s11_im = Spline(
                self.freq.freq, fl["internal_switch_s11_imag"][...]
            )
            self._intsw_s12_rl = Spline(
                self.freq.freq, fl["internal_switch_s12_real"][...]
            )
            self._intsw_s12_im = Spline(
                self.freq.freq, fl["internal_switch_s12_imag"][...]
            )
            self._intsw_s22_rl = Spline(
                self.freq.freq, fl["internal_switch_s22_real"][...]
            )
            self._intsw_s22_im = Spline(
                self.freq.freq, fl["internal_switch_s22_imag"][...]
            )

    @classmethod
    def from_calobs(cls, calobs: CalibrationObservation) -> Calibration:
        """Generate a :class:`Calibration` from an in-memory observation."""
        tmp = tempfile.mktemp()
        calobs.write(tmp)
        return cls(tmp)

    def lna_s11(self, freq=None):
        """Get the LNA S11 at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self._lna_s11_rl(freq) + 1j * self._lna_s11_im(freq)

    def internal_switch_s11(self, freq=None):
        """Get the S11 of the internal switch at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self._intsw_s11_rl(freq) + 1j * self._intsw_s11_im(freq)

    def internal_switch_s12(self, freq=None):
        """Get the S12 of the internal switch at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self._intsw_s12_rl(freq) + 1j * self._intsw_s12_im(freq)

    def internal_switch_s22(self, freq=None):
        """Get the S22 of the internal switch at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self._intsw_s22_rl(freq) + 1j * self._intsw_s22_im(freq)

    def C1(self, freq=None):
        """Evaluate the Scale polynomial at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self.C1_poly(self.freq.normalize(freq))

    def C2(self, freq=None):
        """Evaluate the Offset polynomial at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self.C2_poly(self.freq.normalize(freq))

    def Tcos(self, freq=None):
        """Evaluate the cos temperature polynomial at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self.Tcos_poly(self.freq.normalize(freq))

    def Tsin(self, freq=None):
        """Evaluate the sin temperature polynomial at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self.Tsin_poly(self.freq.normalize(freq))

    def Tunc(self, freq=None):
        """Evaluate the uncorrelated temperature polynomial at given frequencies."""
        if freq is None:
            freq = self.freq.freq
        return self.Tunc_poly(self.freq.normalize(freq))

    def get_K(
        self, freq: np.ndarray | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the source-S11-dependent factors of Monsalve (2017) Eq. 7."""
        if freq is None:
            freq = self.freq.freq
            gamma_ants = self.s11_correction_models
        else:
            gamma_ants = {
                name: source.s11_model(freq) for name, source in self._loads.items()
            }

        lna_s11 = self.lna_s11(freq)
        return {
            name: rcf.get_K(gamma_rec=lna_s11, gamma_ant=gamma_ant)
            for name, gamma_ant in gamma_ants.items()
        }

    def _linear_coefficients(self, freq, ant_s11):
        return rcf.get_linear_coefficients(
            ant_s11,
            self.lna_s11(freq),
            self.C1(freq),
            self.C2(freq),
            self.Tunc(freq),
            self.Tcos(freq),
            self.Tsin(freq),
            self.t_load,
        )

    def calibrate_temp(self, freq: np.ndarray, temp: np.ndarray, ant_s11: np.ndarray):
        """
        Calibrate given uncalibrated spectrum.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to calibrate
        temp :  np.ndarray
            The temperatures to calibrate (in K).
        ant_s11 : np.ndarray
            The antenna S11 for the load.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.
        """
        a, b = self._linear_coefficients(freq, ant_s11)
        return temp * a + b

    def decalibrate_temp(self, freq, temp, ant_s11):
        """
        De-calibrate given calibrated spectrum.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to calibrate
        temp :  np.ndarray
            The temperatures to calibrate (in K).
        ant_s11 : np.ndarray
            The antenna S11 for the load.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.

        Notes
        -----
        Using this and then :method:`calibrate_temp` immediately should be an identity
        operation.
        """
        a, b = self._linear_coefficients(freq, ant_s11)
        return (temp - b) / a

    def calibrate_Q(
        self, freq: np.ndarray, q: np.ndarray, ant_s11: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate given power ratio spectrum.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to calibrate
        q :  np.ndarray
            The power ratio to calibrate.
        ant_s11 : np.ndarray
            The antenna S11 for the load.

        Returns
        -------
        temp : np.ndarray
            The calibrated temperature.
        """
        uncal_temp = self.t_load_ns * q + self.t_load

        return self.calibrate_temp(freq, uncal_temp, ant_s11)


def perform_term_sweep(
    calobs: CalibrationObservation,
    delta_rms_thresh: float = 0,
    max_cterms: int = 15,
    max_wterms: int = 15,
    explore_run_nums: bool = False,
    explore_repeat_nums: bool = False,
    direc=".",
    verbose=False,
) -> CalibrationObservation:
    """For a given calibration definition, perform a sweep over number of terms.

    There are options to save _every_ calibration solution, or just the "best" one.

    Parameters
    ----------
    calobs: :class:`CalibrationObservation` instance
        The definition calibration class. The `cterms` and `wterms` in this instance
        should define the *lowest* values of the parameters to sweep over.
    delta_rms_thresh : float
        The threshold in change in RMS between one set of parameters and the next that
        will define where to cut off. If zero, will run all sets of parameters up to
        the maximum terms specified.
    max_cterms : int
        The maximum number of cterms to trial.
    max_wterms : int
        The maximum number of wterms to trial.
    explore_run_nums : bool
        Whether to iterate over S11 run numbers to find the best residuals.
    explore_repeat_nums : bool
        Whether to iterate over S11 repeat numbers to find the best residuals.
    direc : str
        Directory to write resultant :class:`Calibration` file to.
    verbose : bool
        Whether to write out the RMS values derived throughout the sweep.

    Notes
    -----
    When exploring run/repeat nums, run nums are kept constant within a load (i.e. the
    match/short/open etc. all have either run_num=1 or run_num=2 for the same load.
    This is physically motivated.
    """
    cterms = range(calobs.cterms, max_cterms)
    wterms = range(calobs.wterms, max_wterms)

    winner = np.zeros(len(cterms), dtype=int)

    s11_keys = ["switching_state", "receiver_reading"] + list(io.LOAD_ALIASES.keys())
    if explore_repeat_nums:
        # Note that we don't explore run_nums for spectra/resistance, because it's rare
        # to have those, and they'll only exist if one got completely botched (and that
        # should be set by the user).
        rep_num = {
            k: range(1, getattr(calobs.io.s11, k).max_repeat_num + 1) for k in s11_keys
        }
    else:
        rep_num = {k: [getattr(calobs.io.s11, k).repeat_num] for k in s11_keys}

    rep_num = tools.dct_of_list_to_list_of_dct(rep_num)

    if explore_run_nums:
        run_num = {
            "switching_state": range(
                1, calobs.io.s11.get_highest_run_num("SwitchingState") + 1
            ),
            "receiver_reading": range(
                1, calobs.io.s11.get_highest_run_num("ReceiverReading") + 1
            ),
        }
    else:
        run_num = {
            "switching_state": [calobs.io.s11.switching_state.run_num],
            "receiver_reading": [calobs.io.s11.receiver_reading.run_num],
        }

    run_num = tools.dct_of_list_to_list_of_dct(run_num)

    best_rms = np.inf
    for this_rep_num in rep_num:
        for this_run_num in run_num:

            if verbose:
                print(
                    f"SWEEPING SwSt={calobs.io.s11.switching_state.repeat_num}, "
                    f"RcvRd={calobs.io.s11.receiver_reading.repeat_num} "
                    f"[Sw={calobs.io.s11.switching_state.run_num}, "
                    f"RR={calobs.io.s11.receiver_reading.run_num}, "
                    f"open={calobs.io.s11.open.run_num}, "
                    f"short={calobs.io.s11.short.run_num}, "
                    f"ambient={calobs.io.s11.ambient.run_num}, "
                    f"hot={calobs.io.s11.hot_load.run_num}]"
                )
                print("-" * 30)

            rms = np.zeros((len(cterms), len(wterms)))

            for i, c in enumerate(cterms):
                for j, w in enumerate(wterms):
                    clb = calobs.clone(
                        cterms=c,
                        wterms=w,
                        repeat_num=this_rep_num,
                        run_num=this_run_num,
                    )
                    res = clb.get_load_residuals()
                    dof = sum(len(r) for r in res.values()) - c - w

                    rms[i, j] = np.sqrt(
                        sum(np.nansum(np.square(x)) for x in res.values()) / dof
                    )

                    if verbose:
                        print(f"Nc = {c:02}, Nw = {w:02}; RMS/dof = {rms[i, j]:1.3e}")

                    # If we've decreased by more than the threshold, this wterms becomes
                    # the new winner (for this number of cterms)
                    if j > 0 and rms[i, j] >= rms[i, j - 1] - delta_rms_thresh:
                        winner[i] = j - 1
                        break

                if (
                    i > 0
                    and rms[i, winner[i]]
                    >= rms[i - 1, winner[i - 1]] - delta_rms_thresh
                ):
                    break

            if verbose:
                print(
                    f"Best parameters found for Nc={cterms[i-1]}, "
                    f"Nw={wterms[winner[i-1]]}, "
                    f"with RMS = {rms[i-1, winner[i-1]]}."
                )
                print()

            if rms[i - 1, winner[i - 1]] < best_rms:
                best_run_combo = (
                    clb.io.run_num,
                    clb.io.s11.receiver_reading.repeat_num,
                    clb.io.s11.switching_state.repeat_num,
                )
                best_cterms = cterms[i - 1]
                best_wterms = wterms[winner[i - 1]]

    if verbose and (explore_repeat_nums or explore_run_nums):
        print("The very best parameters were found were for:")
        print(f"\tSwitchingState Repeat = {best_run_combo[2]}")
        print(f"\tReceiverReading Repeat = {best_run_combo[1]}")
        print(f"\tRun Numbers = {best_run_combo[0]}")
        print(f"\t# C-terms = {best_cterms}")
        print(f"\t# W-terms = {best_wterms}")

    clb = calobs.clone(
        cterms=best_cterms,
        wterms=best_wterms,
        run_num=best_run_combo[0],
        repeat_num={
            "switching_state": best_run_combo[2],
            "receiver_reading": best_run_combo[1],
        },
    )

    if direc is not None:
        direc = Path(direc)
        if not direc.exists():
            direc.mkdir(parents=True)

        pth = Path(clb.path).parent.name

        pth = str(pth) + f"_c{clb.cterms}_w{clb.wterms}.h5"
        clb.write(direc / pth)

    return clb
