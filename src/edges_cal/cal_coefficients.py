"""
The main user-facing module of ``edges-cal``.

This module contains wrappers around lower-level functions in other modules, providing
a one-stop interface for everything related to calibration.
"""
from __future__ import annotations

import attr
import h5py
import hickle
import numpy as np
import warnings
from astropy import units as un
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io.misc import yaml as ayaml
from edges_io import h5, io
from edges_io.logging import logger
from functools import partial
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from . import modelling as mdl
from . import receiver_calibration_func as rcf
from . import reflection_coefficient as rc
from . import s11
from . import types as tp
from .cached_property import cached_property, safe_property
from .spectra import LoadSpectrum
from .tools import FrequencyRange, bin_array, get_data_path


@h5.hickleable()
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

    freq: FrequencyRange = attr.ib()
    raw_s11: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    raw_s12s21: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    raw_s22: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))

    n_terms: int = attr.ib(default=21, converter=int)

    @classmethod
    def from_file(
        cls,
        path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        **kwargs,
    ):
        """Instantiate the HotLoadCorrection from file.

        Parameters
        ----------
        path
            Path to the S-parameters file.
        f_low, f_high
            The min/max frequencies to use in the modelling.
        """
        path = get_data_path(path)

        data = np.genfromtxt(path)
        freq = FrequencyRange(data[:, 0] * un.MHz, f_low=f_low, f_high=f_high)

        data = data[freq.mask]

        if data.shape[1] == 7:  # Original file from 2015
            data = data[:, 1::2] + 1j * data[:, 2::2]
        elif data.shape[1] == 6:  # File from 2017
            data = np.array(
                [
                    data[:, 1] + 1j * data[:, 2],
                    data[:, 3],
                    data[:, 4] + 1j * data[:, 5],
                ]
            ).T
        return cls(
            freq=freq,
            raw_s11=data[:, 0],
            raw_s12s21=data[:, 1],
            raw_s22=data[:, 2],
            **kwargs,
        )

    def _get_model(self, raw_data: np.ndarray):
        model = mdl.Polynomial(
            n_terms=self.n_terms,
            transform=mdl.UnitTransform(
                range=(self.freq.min.to_value("MHz"), self.freq.max.to_value("MHz"))
            ),
        )
        model = mdl.ComplexMagPhaseModel(mag=model, phs=model)
        return model.fit(xdata=self.freq.freq, ydata=raw_data)

    @cached_property
    def s11_model(self):
        """The reflection coefficient."""
        return self._get_model(self.raw_s11)

    @cached_property
    def s12s21_model(self):
        """The transmission coefficient."""
        return self._get_model(self.raw_s12s21)

    @cached_property
    def s22_model(self):
        """The reflection coefficient from the other side."""
        return self._get_model(self.raw_s22)

    def power_gain(self, freq: tp.FreqType, hot_load_s11: s11.LoadS11) -> np.ndarray:
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
        assert isinstance(hot_load_s11, s11.LoadS11), "hot_load_s11 must be a LoadS11"
        assert (
            hot_load_s11.load_name == "hot_load"
        ), "hot_load_s11 must be a hot_load s11"

        return self.get_power_gain(
            {
                "s11": self.s11_model(freq.to_value("MHz")),
                "s12s21": self.s12s21_model(freq.to_value("MHz")),
                "s22": self.s22_model(freq.to_value("MHz")),
            },
            hot_load_s11.s11_model(freq.to_value("MHz")),
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


@h5.hickleable()
@attr.s(kw_only=True)
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
    reflections: s11.LoadS11 = attr.ib()
    _loss_model: Callable[
        [np.ndarray], np.ndarray
    ] | HotLoadCorrection | None = attr.ib(default=None)
    ambient_temperature: float = attr.ib(default=298.0)

    @property
    def loss_model(self):
        """The loss model as a callable function of frequency."""
        if isinstance(self._loss_model, HotLoadCorrection):
            return partial(self._loss_model.power_gain, hot_load_s11=self.reflections)
        else:
            return self._loss_model

    @property
    def load_name(self) -> str:
        """The name of the load."""
        return self.reflections.load_name

    @classmethod
    def from_io(
        cls,
        io_obj: io.CalibrationObservation,
        load_name: str,
        f_low: tp.FreqType = 40 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        reflection_kwargs: dict | None = None,
        spec_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        ambient_temperature: float | None = None,
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
        ambient_temperature
            The ambient temperature to use for the loss, if required (required for new
            hot loads). By default, read an ambient load's actual temperature reading
            from the io object.

        Returns
        -------
        load : :class:`Load`
            The load object, containing all info about spectra and S11's for that load.
        """
        if not spec_kwargs:
            spec_kwargs = {}
        if not reflection_kwargs:
            reflection_kwargs = {}
        loss_kwargs = loss_kwargs or {}
        # Fill up kwargs with keywords from this instance
        # TODO: here we only use the calkit defined for the FIRST switching_state,
        # instead of using each calkit for each switching_state. To fix this, we require
        # having meta information inside the S11/ directory.
        if "internal_switch_kwargs" not in reflection_kwargs:
            reflection_kwargs["internal_switch_kwargs"] = {}

        if "calkit" not in reflection_kwargs["internal_switch_kwargs"]:
            reflection_kwargs["internal_switch_kwargs"]["calkit"] = rc.get_calkit(
                rc.AGILENT_85033E,
                resistance_of_match=io_obj.definition["measurements"]["resistance_m"][
                    io_obj.s11.switching_state[0].run_num
                ],
            )

        spec = LoadSpectrum.from_io(
            io_obs=io_obj,
            load_name=load_name,
            f_low=f_low,
            f_high=f_high,
            **spec_kwargs,
        )

        refl = s11.LoadS11.from_io(
            io_obj.s11,
            load_name,
            f_low=f_low,
            f_high=f_high,
            **reflection_kwargs,
        )

        if load_name == "hot_load":
            hlc = HotLoadCorrection.from_file(f_low=f_low, f_high=f_high, **loss_kwargs)

            if ambient_temperature is None:
                ambient_temperature = LoadSpectrum.from_io(
                    io_obj,
                    load_name="ambient",
                    f_low=f_low,
                    f_high=f_high,
                    **spec_kwargs,
                ).temp_ave

            return cls(
                spectrum=spec,
                reflections=refl,
                loss_model=hlc,
                ambient_temperature=ambient_temperature,
            )
        else:
            return cls(spectrum=spec, reflections=refl)

    def get_temp_with_loss(self, freq: tp.FreqType | None = None):
        """Calculate the temperature of the load accounting for loss."""
        if freq is None:
            freq = self.freq.freq

        if self.loss_model is None:
            return self.spectrum.temp_ave * np.ones(len(freq))

        gain = self.loss_model(freq)
        return gain * self.spectrum.temp_ave + (1 - gain) * self.ambient_temperature

    @cached_property
    def temp_ave(self) -> np.ndarray:
        """The average temperature of the thermistor (over frequency and time)."""
        return self.get_temp_with_loss()

    @property
    def averaged_Q(self) -> np.ndarray:
        """The average spectrum power ratio, Q (over time)."""
        return self.spectrum.q

    @property
    def averaged_spectrum(self) -> np.ndarray:
        """The average uncalibrated spectrum (over time)."""
        return self.spectrum.averaged_spectrum

    @property
    def t_load(self) -> float:
        """The assumed temperature of the internal load."""
        return self.spectrum.t_load

    @property
    def t_load_ns(self) -> float:
        """The assumed temperature of the internal load + noise source."""
        return self.spectrum.t_load_ns

    @property
    def s11_model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Callable S11 model as function of frequency."""
        return self.reflections.s11_model

    @property
    def freq(self) -> FrequencyRange:
        """Frequencies of the spectrum."""
        return self.spectrum.freq

    def with_calkit(self, calkit: rc.Calkit):
        """Return a new Load with updated calkit."""
        if "calkit" not in self.reflections.metadata:
            raise RuntimeError(
                "Cannot clone with new calkit since calkit is unknown for the load"
            )

        loads11 = [
            attr.evolve(x, calkit=calkit)
            for x in self.reflections.metadata["load_s11s"]
        ]
        isw = self.reflections.internal_switch.with_new_calkit(calkit)

        return attr.evolve(
            self,
            reflections=s11.LoadS11.from_load_and_internal_switch(
                load_s11=loads11, internal_switch=isw, base=self.reflections
            ),
        )


@h5.hickleable()
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
    loads
        dictionary of load names to Loads
    receiver
        The object defining the reflection coefficient of the receiver.
    cterms
        The number of polynomial terms used for the scaling/offset functions
    wterms
        The number of polynomial terms used for the noise-wave parameters.
    metadata
        Metadata associated with the data.
    """

    loads: dict[str, Load] = attr.ib()
    receiver: s11.Receiver | s11.AveragedReceiver = attr.ib()
    cterms: int = attr.ib(default=5, kw_only=True)
    wterms: int = attr.ib(default=7, kw_only=True)

    _metadata: dict[str, Any] = attr.ib(default=attr.Factory(dict), kw_only=True)

    @property
    def metadata(self):
        """Metadata associated with the object."""
        return self._metadata

    @classmethod
    def from_io(
        cls,
        io_obj: io.CalibrationObservation,
        *,
        semi_rigid_path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
        freq_bin_size: int = 1,
        spectrum_kwargs: dict[str, dict[str, Any]] | None = None,
        s11_kwargs: dict[str, dict[str, Any]] | None = None,
        internal_switch_kwargs: dict[str, Any] | None = None,
        lna_kwargs: dict[str, Any] | None = None,
        f_low: tp.FreqType = 40.0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        sources: tuple[str] = ("ambient", "hot_load", "open", "short"),
        receiver_kwargs: dict[str, Any] | None = None,
        restrict_s11_model_freqs: bool = True,
        **kwargs,
    ) -> CalibrationObservation:
        """Create the object from an edges-io observation.

        Parameters
        ----------
        io_obj
            An calibration observation object from which all the data can be read.
        semi_rigid_path : str or Path, optional
            Path to a file containing S11 measurements for the semi rigid cable. Used to
            correct the hot load S11. Found automatically if not given.
        freq_bin_size
            The size of each frequency bin (of the spectra) in units of the raw size.
        spectrum_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadSpectrum`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        s11_kwargs
            Keyword arguments used to instantiate the calibrator :class:`LoadS11`
            objects. See its documentation for relevant parameters. Parameters specified
            here are used for _all_ calibrator sources.
        internal_switch_kwargs
            Keyword arguments used to instantiate the :class:`~s11.InternalSwitch`
            objects. See its documentation for relevant parameters. The same internal
            switch is used to calibrate the S11 for each input source.
        f_low : float
            Minimum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        f_high : float
            Maximum frequency to keep for all loads (and their S11's). If for some
            reason different frequency bounds are desired per-load, one can pass in
            full load objects through ``load_spectra``.
        sources
            A sequence of strings specifying which loads to actually use in the
            calibration. Default is all four standard calibrators.
        receiver_kwargs
            Keyword arguments used to instantiate the calibrator :class:`~s11.Receiver`
            objects. See its documentation for relevant parameters. ``lna_kwargs`` is a
            deprecated alias.
        restrict_s11_model_freqs
            Whether to restrict the S11 modelling (i.e. smoothing) to the given freq
            range. The final output will be calibrated only between the given freq
            range, but the S11 models themselves can be fit over a broader set of
            frequencies.
        """
        if f_high < f_low:
            raise ValueError("f_high must be larger than f_low!")

        if lna_kwargs is not None:
            warnings.warn(
                "Use of 'lna_kwargs' is deprecated, use 'receiver_kwargs' instead."
            )
            receiver_kwargs = lna_kwargs

        spectrum_kwargs = spectrum_kwargs or {}
        s11_kwargs = s11_kwargs or {}
        internal_switch_kwargs = internal_switch_kwargs or {}
        receiver_kwargs = receiver_kwargs or {}

        for v in [spectrum_kwargs, s11_kwargs, internal_switch_kwargs, receiver_kwargs]:
            assert isinstance(v, dict)

        f_low = f_low.to("MHz", copy=False)
        f_high = f_high.to("MHz", copy=False)

        if "calkit" not in receiver_kwargs:
            receiver_kwargs["calkit"] = rc.get_calkit(
                rc.AGILENT_85033E,
                resistance_of_match=io_obj.definition.get("measurements", {})
                .get("resistance_f", {})
                .get(io_obj.s11.receiver_reading[0].run_num, 50.0 * un.Ohm),
            )

        receiver = s11.Receiver.from_io(
            device=io_obj.s11.receiver_reading,
            f_low=f_low if restrict_s11_model_freqs else 0 * un.MHz,
            f_high=f_high if restrict_s11_model_freqs else np.inf * un.MHz,
            **receiver_kwargs,
        )

        f_low = max(receiver.freq.min, f_low)
        f_high = min(receiver.freq.max, f_high)

        if "default" not in spectrum_kwargs:
            spectrum_kwargs["default"] = {}

        if "freq_bin_size" not in spectrum_kwargs["default"]:
            spectrum_kwargs["default"]["freq_bin_size"] = freq_bin_size

        loads = {
            source: Load.from_io(
                io_obj=io_obj,
                load_name=source,
                f_low=f_low,
                f_high=f_high,
                reflection_kwargs={
                    **s11_kwargs.get("default", {}),
                    **s11_kwargs.get(source, {}),
                    **{"internal_switch_kwargs": internal_switch_kwargs},
                },
                spec_kwargs={
                    **spectrum_kwargs["default"],
                    **spectrum_kwargs.get(source, {}),
                },
                loss_kwargs={"path": semi_rigid_path},
            )
            for source in sources
        }

        return cls(
            loads=loads,
            receiver=receiver,
            metadata={
                "path": io_obj.path,
                "s11_kwargs": s11_kwargs,
                "lna_kwargs": lna_kwargs,
                "spectra": {
                    name: load.spectrum.metadata for name, load in loads.items()
                },
                "io": io_obj,
            },
            **kwargs,
        )

    def __getattr__(self, name):
        """Get loads as attributes."""
        if name in self.loads:
            return self.loads[name]

        raise AttributeError(f"{name} does not exist in CalibrationObservation!")

    def with_load_calkit(self, calkit, loads: Sequence[str] = None):
        """Return a new observation with loads having given calkit."""
        if loads is None:
            loads = self.load_names
        elif isinstance(loads, str):
            loads = [loads]

        loads = {
            name: load.with_calkit(calkit) if name in loads else load
            for name, load in self.loads.items()
        }

        return attr.evolve(self, loads=loads)

    @safe_property
    def t_load(self) -> float:
        """Assumed temperature of the load."""
        return self.loads[list(self.loads.keys())[0]].t_load

    @safe_property
    def t_load_ns(self) -> float:
        """Assumed temperature of the load + noise source."""
        return self.loads[list(self.loads.keys())[0]].t_load_ns

    @cached_property
    def freq(self) -> FrequencyRange:
        """The frequencies at which spectra were measured."""
        return self.loads[list(self.loads.keys())[0]].freq

    @safe_property
    def internal_switch(self):
        """The S11 object representing the internal switch."""
        return self.loads[self.load_names[0]].reflections.internal_switch

    @safe_property
    def load_names(self) -> tuple[str]:
        """Names of the loads."""
        return tuple(self.loads.keys())

    def new_load(
        self,
        load_name: str,
        io_obj: io.CalibrationObservation,
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

        spec_kwargs["freq_bin_size"] = self.freq.bin_size
        spec_kwargs["t_load"] = self.open.spectrum.t_load
        spec_kwargs["t_load_ns"] = self.open.spectrum.t_load_ns

        return Load.from_io(
            io_obj=io_obj,
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
                len(self.loads), 1, sharex=True, gridspec_kw={"hspace": 0.05}
            )

        for i, (name, load) in enumerate(self.loads.items()):
            ax[i].plot(load.freq.freq, load.averaged_spectrum)
            ax[i].set_ylabel("$T^*$ [K]")
            ax[i].set_title(name)
            ax[i].grid(True)
        ax[-1].set_xlabel("Frequency [MHz]")

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
            for name, source in self.loads.items()
        }
        out.update({"lna": self.receiver.plot_residuals(**kwargs)})
        return out

    @cached_property
    def s11_correction_models(self):
        """Dictionary of S11 correction models, one for each source."""
        try:
            return dict(self._injected_source_s11s)
        except (TypeError, AttributeError):
            return {
                name: source.s11_model(self.freq.freq.to_value("MHz"))
                for name, source in self.loads.items()
            }

    @cached_property
    def source_thermistor_temps(self) -> dict[str, float | np.ndarray]:
        """Dictionary of input source thermistor temperatures."""
        if (
            hasattr(self, "_injected_source_temps")
            and self._injected_source_temps is not None
        ):
            return self._injected_source_temps
        return {k: source.temp_ave for k, source in self.loads.items()}

    @cached_property
    def _calibration_coefficients(self):
        """The calibration polynomials, evaluated at `freq.freq`."""
        if (
            hasattr(self, "_injected_averaged_spectra")
            and self._injected_averaged_spectra is not None
        ):
            ave_spec = self._injected_averaged_spectra
        else:
            ave_spec = {k: source.averaged_spectrum for k, source in self.loads.items()}

        scale, off, Tu, TC, TS = rcf.get_calibration_quantities_iterative(
            self.freq.freq_recentred,
            temp_raw=ave_spec,
            gamma_rec=self.receiver_s11,
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

    def C1(self, f: tp.FreqType | None = None):  # noqa: N802
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

    def C2(self, f: tp.FreqType | None = None):  # noqa: N802
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

    def Tunc(self, f: tp.FreqType | None = None):  # noqa: N802
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

    def Tcos(self, f: tp.FreqType | None = None):  # noqa: N802
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

    def Tsin(self, f: tp.FreqType | None = None):  # noqa: N802
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
    def receiver_s11(self):
        """The corrected S11 of the LNA evaluated at the data frequencies."""
        if hasattr(self, "_injected_lna_s11") and self._injected_lna_s11 is not None:
            return self._injected_lna_s11
        else:
            return self.receiver.s11_model(self.freq.freq.to_value("MHz"))

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
            load_s11 = load.s11_model(self.freq.freq.to_value("MHz"))

        return rcf.get_linear_coefficients(
            load_s11,
            self.receiver_s11,
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
                load = self.loads[load]
            except (AttributeError, KeyError):
                raise AttributeError(
                    f"load must be a Load object or a string (one of {self.load_names})"
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

        if freq.max() > self.freq.freq.max():
            warnings.warn("The maximum frequency is outside the calibrated range ")

        a, b = self.get_linear_coefficients(load)
        return (temp - b) / a

    def get_K(
        self, freq: tp.FreqType | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get the source-S11-dependent factors of Monsalve (2017) Eq. 7."""
        if freq is None:
            freq = self.freq.freq
            gamma_ants = self.s11_correction_models
        else:
            gamma_ants = {
                name: source.s11_model(freq.to_value("MHz"))
                for name, source in self.loads.items()
            }

        lna_s11 = self.receiver.s11_model(freq.to_value("MHz"))
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
        label: str = "",
        as_residuals: bool = False,
        load_in_title: bool = False,
        rms_in_label: bool = True,
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
            freq_ave_cal = bin_array(temp_calibrated, size=bins)
            f = bin_array(self.freq.freq.to_value("MHz"), size=bins)
        else:
            freq_ave_cal = temp_calibrated
            f = self.freq.freq.to_value("MHz")

        freq_ave_cal[np.isinf(freq_ave_cal)] = np.nan

        rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

        ax.plot(
            f,
            freq_ave_cal,
            label=f"Calibrated {load.load_name} [RMS = {rms:.3f}]",
        )

        temp_ave = self.source_thermistor_temps.get(load.load_name, load.temp_ave)

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
        return {
            name: self.calibrate(load) - load.temp_ave
            for name, load in self.loads.items()
        }

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

    def plot_calibrated_temps(self, bins=64, fig=None, ax=None, **kwargs):
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
        if fig is None or ax is None or len(ax) != len(self.loads):
            fig, ax = plt.subplots(
                len(self.loads),
                1,
                sharex=True,
                gridspec_kw={"hspace": 0.05},
                figsize=(10, 12),
            )

        for i, source in enumerate(self.loads):
            self.plot_calibrated_temp(
                source,
                bins=bins,
                fig=fig,
                ax=ax[i],
                xlabel=i == (len(self.loads) - 1),
            )

        fig.suptitle("Calibrated Temperatures for Calibration Sources", fontsize=15)
        return fig

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
        # TODO: this is *not* all the metadata available when using edges-io. We should
        # build a better system of maintaining metadata in subclasses to be used here.
        with h5py.File(filename, "w") as fl:
            # Write attributes

            fl.attrs["cterms"] = self.cterms
            fl.attrs["wterms"] = self.wterms
            fl.attrs["t_load"] = self.open.spectrum.t_load
            fl.attrs["t_load_ns"] = self.open.spectrum.t_load_ns

            fl["C1"] = self.C1_poly.coefficients
            fl["C2"] = self.C2_poly.coefficients
            fl["Tunc"] = self.Tunc_poly.coefficients
            fl["Tcos"] = self.Tcos_poly.coefficients
            fl["Tsin"] = self.Tsin_poly.coefficients

            hickle.dump(self.freq, fl.create_group("frequencies"))
            hickle.dump(
                self.receiver,
                fl.create_group("receiver_s11"),
            )
            hickle.dump(
                self.internal_switch,
                fl.create_group("internal_switch"),
            )
            hickle.dump(
                self.metadata,
                fl.create_group("metadata"),
            )

    def to_calibrator(self):
        """Directly create a :class:`Calibrator` object without writing to file."""
        return Calibrator(
            cterms=self.cterms,
            wterms=self.wterms,
            t_load=self.t_load,
            t_load_ns=self.t_load_ns,
            C1=self.C1_poly,
            C2=self.C2_poly,
            Tunc=self.Tunc_poly,
            Tcos=self.Tcos_poly,
            Tsin=self.Tsin_poly,
            freq=self.freq,
            receiver_s11=self.receiver.s11_model,
            internal_switch=self.internal_switch,
            metadata=self.metadata,
        )

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
        new = self.clone()
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

    @classmethod
    def from_yaml(cls, config: tp.PathLike | dict, obs_path: tp.PathLike | None = None):
        """Create the calibration observation from a YAML configuration."""
        if not isinstance(config, dict):
            with open(config) as yml:
                config = ayaml.load(yml)

        iokw = config.pop("data", {})

        if not obs_path:
            obs_path = iokw.pop("path")

        from_def = iokw.pop("compile_from_def", False)

        if from_def:
            io_obs = io.CalibrationObservation.from_def(obs_path, **iokw)
        else:
            io_obs = io.CalibrationObservation(obs_path, **iokw)

        return cls.from_io(io_obs, **config)


@h5.hickleable()
@attr.s(kw_only=True)
class Calibrator:
    freq: FrequencyRange = attr.ib()

    cterms: int = attr.ib()
    wterms: int = attr.ib()

    _C1: Callable[[np.ndarray], np.ndarray] = attr.ib()
    _C2: Callable[[np.ndarray], np.ndarray] = attr.ib()
    _Tunc: Callable[[np.ndarray], np.ndarray] = attr.ib()
    _Tcos: Callable[[np.ndarray], np.ndarray] = attr.ib()
    _Tsin: Callable[[np.ndarray], np.ndarray] = attr.ib()
    _receiver_s11: Callable[[np.ndarray], np.ndarray] = attr.ib()
    internal_switch = attr.ib()
    t_load: float = attr.ib(300)
    t_load_ns: float = attr.ib(350)
    metadata: dict = attr.ib(default=attr.Factory(dict))

    def __attrs_post_init__(self):
        """Initialize properties of the class."""
        for key in ["C1", "C2", "Tunc", "Tcos", "Tsin"]:
            setattr(self, key, partial(self._call_func, key=key, norm=True))
        for key in [
            "receiver_s11",
        ]:
            setattr(self, key, partial(self._call_func, key=key, norm=False))

    def clone(self, **kwargs):
        """Clone the instance with new parameters."""
        return attr.evolve(self, **kwargs)

    @internal_switch.validator
    def _isw_vld(self, att, val):
        if isinstance(val, s11.InternalSwitch):
            return

        for key in ("s11", "s12", "s22"):
            if not hasattr(val, f"{key}_model") or not callable(
                getattr(val, f"{key}_model")
            ):
                raise ValueError(f"internal_switch must provide {key}_model method")

    def _call_func(self, freq: tp.FreqType | None = None, *, key=None, norm=False):
        if freq is None:
            freq = self.freq.freq

        if not hasattr(freq, "unit"):
            raise ValueError("freq must have units of frequency")

        if norm:
            freq = self.freq.normalize(freq)
        else:
            freq = freq.to_value("MHz")

        return getattr(self, "_" + key)(freq)

    @classmethod
    def from_calobs_file(cls, path: tp.PathLike) -> Calibrator:
        """Generate from calobs file."""
        calobs = hickle.load(path)
        return calobs.to_calibrator()

    @classmethod
    def from_calfile(cls, path: tp.PathLike) -> Calibrator:
        """Generate from calfile."""
        with h5py.File(path, "r") as fl:
            cterms = fl.attrs["cterms"]
            wterms = fl.attrs["wterms"]
            t_load = fl.attrs["t_load"]
            t_load_ns = fl.attrs["t_load_ns"]

            C1 = np.poly1d(fl["C1"])
            C2 = np.poly1d(fl["C2"])
            Tunc = np.poly1d(fl["Tunc"])
            Tcos = np.poly1d(fl["Tcos"])
            Tsin = np.poly1d(fl["Tsin"])

            freq = hickle.load(fl["frequencies"])
            receiver_s11 = hickle.load(fl["receiver_s11"])
            internal_switch = hickle.load(fl["internal_switch"])
            metadata = hickle.load(fl["metadata"])

        return cls(
            cterms=cterms,
            wterms=wterms,
            t_load=t_load,
            t_load_ns=t_load_ns,
            C1=C1,
            C2=C2,
            Tunc=Tunc,
            Tcos=Tcos,
            Tsin=Tsin,
            freq=freq,
            receiver_s11=receiver_s11.s11_model,
            internal_switch=internal_switch,
            metadata=metadata,
        )

    @classmethod
    def from_old_calfile(cls, path: tp.PathLike) -> Calibrator:
        """Read from older calfiles."""
        with h5py.File(path, "r") as fl:
            cterms = int(fl.attrs["cterms"])
            wterms = int(fl.attrs["wterms"])
            t_load = fl.attrs.get("t_load", 300)
            t_load_ns = fl.attrs.get("t_load_ns", 400)

            C1_poly = np.poly1d(fl["C1"][...])
            C2_poly = np.poly1d(fl["C2"][...])
            Tcos_poly = np.poly1d(fl["Tcos"][...])
            Tsin_poly = np.poly1d(fl["Tsin"][...])
            Tunc_poly = np.poly1d(fl["Tunc"][...])

            freq = FrequencyRange(fl["frequencies"][...] * un.MHz)

            try:
                metadata = dict(fl["metadata"].attrs)
            except KeyError:
                # For backwards compat
                metadata = {}

            _lna_s11_rl = Spline(freq.freq.to_value("MHz"), fl["lna_s11_real"][...])
            _lna_s11_im = Spline(freq.freq.to_value("MHz"), fl["lna_s11_imag"][...])

            _intsw_s11_rl = Spline(freq.freq, fl["internal_switch_s11_real"][...])
            _intsw_s11_im = Spline(freq.freq, fl["internal_switch_s11_imag"][...])
            _intsw_s12_rl = Spline(freq.freq, fl["internal_switch_s12_real"][...])
            _intsw_s12_im = Spline(freq.freq, fl["internal_switch_s12_imag"][...])
            _intsw_s22_rl = Spline(freq.freq, fl["internal_switch_s22_real"][...])
            _intsw_s22_im = Spline(freq.freq, fl["internal_switch_s22_imag"][...])

            internal_switch = SimpleNamespace(
                s11_model=lambda freq: _intsw_s11_rl(freq) + _intsw_s11_im(freq) * 1j,
                s12_model=lambda freq: _intsw_s12_rl(freq) + _intsw_s12_im(freq) * 1j,
                s22_model=lambda freq: _intsw_s22_rl(freq) + _intsw_s22_im(freq) * 1j,
            )

        return cls(
            C1=C1_poly,
            C2=C2_poly,
            Tunc=Tunc_poly,
            Tcos=Tcos_poly,
            Tsin=Tsin_poly,
            freq=freq,
            receiver_s11=lambda x: _lna_s11_rl(x) + 1j * _lna_s11_im(x),
            internal_switch=internal_switch,
            t_load=t_load,
            t_load_ns=t_load_ns,
            cterms=cterms,
            wterms=wterms,
            metadata=metadata,
        )

    @classmethod
    def from_calobs(cls, calobs: CalibrationObservation) -> Calibrator:
        """Generate a :class:`Calibration` from an in-memory observation."""
        return calobs.to_calibrator()

    def _linear_coefficients(self, freq, ant_s11):
        return rcf.get_linear_coefficients(
            ant_s11,
            self.receiver_s11(freq),
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
) -> CalibrationObservation:
    """For a given calibration definition, perform a sweep over number of terms.

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
    """
    cterms = range(calobs.cterms, max_cterms)
    wterms = range(calobs.wterms, max_wterms)

    winner = np.zeros(len(cterms), dtype=int)
    rms = np.ones((len(cterms), len(wterms))) * np.inf

    for i, c in enumerate(cterms):
        for j, w in enumerate(wterms):
            clb = calobs.clone(cterms=c, wterms=w)

            res = clb.get_load_residuals()
            dof = sum(len(r) for r in res.values()) - c - w

            rms[i, j] = np.sqrt(
                sum(np.nansum(np.square(x)) for x in res.values()) / dof
            )

            logger.info(f"Nc = {c:02}, Nw = {w:02}; RMS/dof = {rms[i, j]:1.3e}")

            # If we've decreased by more than the threshold, this wterms becomes
            # the new winner (for this number of cterms)
            if j > 0 and rms[i, j] >= rms[i, j - 1] - delta_rms_thresh:
                winner[i] = j - 1
                break

        if i > 0 and rms[i, winner[i]] >= rms[i - 1, winner[i - 1]] - delta_rms_thresh:
            break

    logger.info(
        f"Best parameters found for Nc={cterms[i-1]}, "
        f"Nw={wterms[winner[i-1]]}, "
        f"with RMS = {rms[i-1, winner[i-1]]}."
    )

    best = np.unravel_index(np.argmin(rms), rms.shape)
    return calobs.clone(
        cterms=cterms[best[0]],
        wterms=cterms[best[1]],
    )
