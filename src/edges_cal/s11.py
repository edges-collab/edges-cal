"""Classes representing S11 measurements.

There are classes here for S11s of external loads, the Receiver and also the
Internal Switch (necessary to correct the S11 of external loads).

The S11s of each "device" are assumed to be measured with a VNA and calibrated
using a Calkit containing Open/Short/Load standards. The formalism for this calibration
is defined in Monsalve et al., 2016. Methods for performing this S11 calibration are
in the :mod:`~.reflection_coefficient` module.

We attempt to keep the interface to each of the devices relatively consistent. Each
provides a `s11_model` method which is a function of frequency, outputting the
calibrated and smoothed S11, according to some smooth model.
"""
from __future__ import annotations

import attr
import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
from astropy import units
from astropy import units as un
from cached_property import cached_property
from edges_io import io
from pathlib import Path
from typing import Callable, Sequence

from . import reflection_coefficient as rc
from . import types as tp
from .modelling import (
    ComplexMagPhaseModel,
    ComplexRealImagModel,
    Model,
    Polynomial,
    UnitTransform,
    get_mdl,
)
from .tools import FrequencyRange, vld_unit


def _s1p_converter(s1p: tp.PathLike | io.S1P) -> io.S1P:
    try:
        s1p = Path(s1p)
        return io.S1P(s1p)
    except TypeError as e:
        if isinstance(s1p, io.S1P):
            return s1p
        else:
            raise TypeError(
                "s1p must be a path to an s1p file, or an io.S1P object"
            ) from e


def _tuplify(x):
    if not hasattr(x, "__len__"):
        return (int(x), int(x), int(x))
    else:
        return tuple(int(xx) for xx in x)


@attr.s(frozen=True)
class VNAReading:
    """
    An object representing the measurements of a VNA.

    Parameters
    ----------
    freq
        The raw frequencies of the measurement.
    s11
        The raw S11 measurements of the VNA.
    f_low, f_high
        The minimum/maximum frequency to keep.
    """

    _freq: tp.FreqType = attr.ib(
        validator=vld_unit("frequency"), eq=attr.cmp_using(eq=np.array_equal)
    )
    _s11: np.ndarray = attr.ib(
        validator=np.iscomplex, eq=attr.cmp_using(eq=np.array_equal)
    )
    f_low: tp.FreqType = attr.ib(
        default=0 * un.MHz, validator=vld_unit("frequency"), kw_only=True
    )
    f_high: float | None = attr.ib(
        default=np.inf * un.MHz, validator=vld_unit("frequency"), kw_only=True
    )

    @_s11.validator
    def _s11_vld(self, att, val):
        if val.shape != self._freq.shape:
            raise ValueError(
                "freq and s11 must have the same length! "
                f"Got {len(self._freq)} and {len(val)}"
            )

    @cached_property
    def freq(self) -> FrequencyRange:
        """The frequencies of the S11 measurement."""
        return FrequencyRange(
            self._freq.to_value("MHz"), f_low=self.f_low, f_high=self.f_high
        )

    @cached_property
    def s11(self) -> np.ndarray:
        """The S11 measurement."""
        return self._s11[self.freq.mask]

    @classmethod
    def from_s1p(cls, s1p: tp.PathLike | io.S1P, **kwargs) -> VNAReading:
        """Instantiate a :class:`VNAReading` object from an S1P file or object.

        Parameters
        ----------
        s1p
            The file or :class:`edges_io.S1P` object pointing to the file.

        Other Parameters
        ----------------
        Any parameter that instantiates :class:`VNAReading`, other than `freq` and
        `s11`. This includes `f_low` and `f_high`.
        """
        s1p = _s1p_converter(s1p)

        return cls(s1p.freq * un.Hz, s1p.s11, **kwargs)


@attr.s(kw_only=True)
class StandardsReadings:
    open: VNAReading = attr.ib(validator=attr.validators.instance_of(VNAReading))
    short: VNAReading = attr.ib(validator=attr.validators.instance_of(VNAReading))
    match: VNAReading = attr.ib(validator=attr.validators.instance_of(VNAReading))

    @short.validator
    def _short_vld(self, att, val):
        if val.freq != self.open.freq:
            raise ValueError(
                "short standard does not have same frequencies as open standard!"
            )

    @match.validator
    def _match_vld(self, att, val):
        if val.freq != self.open.freq:
            raise ValueError(
                "match standard does not have same frequencies as open standard!"
            )

    @property
    def freq(self) -> FrequencyRange:
        """Frequencies of the standards measurements."""
        return self.open.freq

    @classmethod
    def from_io(
        cls, device: io._S11SubDir, external: bool = False, **kwargs
    ) -> StandardsReadings:
        """Instantiate from a given edges-io object.

        Parameters
        ----------
        device
            The device for which the standards were measured. This is an edges-io
            object.
        external
            Whether to read the external standards measurements from that device,
            or the internal standards.

        Other Parameters
        ----------------
        kwargs
            Everything else is passed to the :class:`VNAReading` objects. This includes
            f_low and f_high.
        """
        ext = "external" if external else ""
        return cls(
            open=VNAReading.from_s1p(device.children[f"{ext}open"], **kwargs),
            short=VNAReading.from_s1p(device.children[f"{ext}short"], **kwargs),
            match=VNAReading.from_s1p(device.children[f"{ext}match"], **kwargs),
        )


@attr.s(kw_only=True, frozen=True)
class _S11Base(metaclass=ABCMeta):
    """
    An abstract base class for representing calibrated S11 measurements of a device.

    Parameters
    ----------
    device
        An instance of the basic ``io`` S11 folder.
    f_low : float
        Minimum frequency to use. Default is all frequencies.
    f_high : float
        Maximum frequency to use. Default is all frequencies.
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

    _complex_model_type_default = ComplexMagPhaseModel
    _model_type_default = "fourier"

    n_terms: tuple[int, int, int] = attr.ib(55, converter=_tuplify)
    model_type: tp.Modelable = attr.ib()
    complex_model_type: type[ComplexMagPhaseModel] | type[
        ComplexRealImagModel
    ] = attr.ib()

    @model_type.default
    def _mdl_type_default(self):
        return self._model_type_default

    @complex_model_type.default
    def _cmt_default(self):
        return self._complex_model_type_default

    @n_terms.validator
    def _nt_vld(self, att, val):
        if not (isinstance(val, int) and val % 2):
            raise ValueError(
                f"n_terms must be odd for S11 models. For {self.device_name} got "
                f"n_terms={val}."
            )

    @property
    @abstractmethod
    def freq(self) -> FrequencyRange:
        """The frequencies at which the internal standards were measured."""
        pass

    @cached_property
    @abstractmethod
    def calibrated_s11_raw(self) -> np.ndarray:
        pass  # pragma: no cover

    def get_s11_model(
        self,
        raw_s11: np.ndarray,
        *,
        freq: np.ndarray | None = None,
        n_terms: int | None = None,
        model_type: tp.Modelable | None = None,
    ):
        """Generate a callable model for the S11.

        This should closely match :method:`s11_correction`.

        Parameters
        ----------
        raw_s11
            The raw s11 of the

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
        freq = freq or self.freq.freq
        n_terms = n_terms or self.n_terms
        model_type = get_mdl(model_type or self.model_type)
        model = model_type(
            n_terms=n_terms,
            transform=UnitTransform(range=[self.freq.min, self.freq.max]),
        )
        emodel = model.at(x=freq)

        cmodel = self.complex_model_type(emodel, emodel)

        return cmodel.fit(ydata=raw_s11)

    @cached_property
    def s11_model(self) -> callable:
        """The S11 model."""
        return self.get_s11_model(self.calibrated_s11_raw)

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
        Plot the residuals of the S11 model compared to un-smoothed corrected data.

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
                axx.grid(True)
        ax[-1].set_xlabel("Frequency [MHz]")

        corr = self.calibrated_s11_raw
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
            title = f"{self.device_name} Reflection Coefficient Models"

        if title:
            fig.suptitle(
                f"{self.device_name} Reflection Coefficient Models", fontsize=14
            )
        if label:
            ax[0].legend()

        return fig


@attr.s(kw_only=True)
class _UncorrectedS11(_S11Base):
    standards: StandardsReadings = attr.ib(
        validator=attr.validators.instance_of(StandardsReadings)
    )
    calkit: rc.Calkit = attr.ib(validator=attr.validators.instance_of(rc.Calkit))

    def __getattr__(self, item):
        if item in self.standards:
            return self.standards[item]

        raise AttributeError(f"{item} does not exist in {self.__class__.__name__}!")

    @property
    def freq(self):
        return self.open.freq


@attr.s(kw_only=True)
class Receiver(_UncorrectedS11):
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

    receiver_reading: VNAReading = attr.ib()
    calkit: rc.Calkit = attr.ib(
        default=rc.get_calkit(rc.AGILENT_85033E, resistance_of_match=50.009 * un.Ohm),
        kw_only=True,
    )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        repeat_num: int | None = attr.NOTHING,
        run_num: int = 1,
        f_low=0.0 * un.MHz,
        f_high=np.inf * un.MHz,
        **kwargs,
    ) -> Receiver:
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
        receiver
            The Receiver object.
        """
        path = Path(path)
        device = io.ReceiverReading(
            path=path / "S11" / f"ReceiverReading{run_num:02}",
            repeat_num=repeat_num,
            fix=False,
        )

        return cls(
            standards=StandardsReadings.from_io(device, f_low=f_low, f_high=f_high),
            receiver_reading=VNAReading(
                device.childen["receiverreading"], f_low=f_low, f_high=f_high
            ),
            **kwargs,
        )

    @cached_property
    def calibrated_s11_raw(self):
        """Measured S11 of of the Receiver."""
        # Correction at switch
        return rc.de_embed(
            self.calkit.open.reflection_coefficient(self.freq.freq * un.MHz),
            self.calkit.short.reflection_coefficient(self.freq.freq * un.MHz),
            self.calkit.match.reflection_coefficient(self.freq.freq * un.MHz),
            self.open.s11,
            self.short.s11,
            self.match.s11,
            self.receiver_reading.s11,
        )[0]


@attr.s
class AveragedReceiver:
    measurements: Sequence[Receiver] = attr.ib(
        validator=attr.validators.deep_iterable(attr.validators.instance_of(Receiver))
    )

    @cached_property
    def calibrated_s11_raw(self):
        """Measured S11 of of the Receiver."""
        # Correction at switch
        return np.mean([m.calibrated_s11_raw for m in self.measurements], axis=0)

    @cached_property
    def s11_model(self) -> callable:
        """The S11 model."""
        return self.get_s11_model(self.calibrated_s11_raw)


@attr.s
class InternalSwitch:
    corrections: dict[str, np.ndarray] = attr.ib()
    freq: np.ndarray = attr.ib()
    model: Model = attr.ib()
    n_terms: tuple[int, int, int] | int = attr.ib(default=(7, 7, 7), converter=_tuplify)
    resistance: float | None = attr.ib(default=None)
    _calkit: rc.Calkit = attr.ib(default=rc.AGILENT_85033E)

    @corrections.validator
    def _corr_vld(self, att, val):
        if (
            len(val) != 3
            or "open" not in val
            or "short" not in val
            or "match" not in val
        ):
            raise ValueError(
                "'corrections' should have open/short/match keys. "
                f"Got {list(val.keys())}"
            )

        if len(val["short"]) != len(val["open"]):
            raise ValueError("short must have same shape as open")

        if len(val["match"]) != len(val["open"]):
            raise ValueError("match must have same shape as open")

    @model.default
    def _mdl_default(self):
        return Polynomial(
            n_terms=7,
            transform=UnitTransform(range=(self.freq.min(), self.freq.max())),
        )

    @classmethod
    def from_io(cls, internal_switch: io.InternalSwitch, **kwargs) -> InternalSwitch:
        """Initiate from an edges-io object."""
        corrections = _read_data_and_corrections(internal_switch)
        return cls(corrections=corrections, freq=internal_switch.freq, **kwargs)

    @cached_property
    def fixed_model(self):
        """The input model fixed to evaluate at the given frequencies."""
        return self.model.at(x=self.freq)

    @n_terms.validator
    def _n_terms_val(self, att, val):
        if len(val) != 3:
            raise TypeError(
                f"n_terms must be an integer or tuple of three integers "
                f"(for s11, s12, s22). Got {val}."
            )
        if any(v < 1 for v in val):
            raise ValueError(f"n_terms must be >0, got {val}.")

    @cached_property
    def calkit(self) -> rc.Calkit:
        """The calkit used for the InternalSwitch."""
        return rc.get_calkit(self._calkit, resistance_of_match=self.resistance)

    @cached_property
    def s11_data(self):
        """The measured S11."""
        return self._de_embedded_reflections[0]

    @cached_property
    def s12_data(self):
        """The measured S12."""
        return self._de_embedded_reflections[1]

    @cached_property
    def s22_data(self):
        """The measured S22."""
        return self._de_embedded_reflections[2]

    @cached_property
    def _s11_model(self):
        """The input unfit S11 model."""
        model = self.model.with_nterms(n_terms=self.n_terms[0])
        return ComplexRealImagModel(real=model, imag=model)

    @cached_property
    def _s12_model(self):
        """The input unfit S12 model."""
        model = self.model.with_nterms(n_terms=self.n_terms[1])
        return ComplexRealImagModel(real=model, imag=model)

    @cached_property
    def _s22_model(self):
        """The input unfit S22 model."""
        model = self.model.with_nterms(n_terms=self.n_terms[2])
        return ComplexRealImagModel(real=model, imag=model)

    @cached_property
    def s11_model(self) -> Callable:
        """The fitted S11 model."""
        return self._get_reflection_model("s11")

    @cached_property
    def s12_model(self) -> Callable:
        """The fitted S12 model."""
        return self._get_reflection_model("s12")

    @cached_property
    def s22_model(self) -> Callable:
        """The fitted S22 model."""
        return self._get_reflection_model("s22")

    @cached_property
    def _de_embedded_reflections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get de-embedded reflection parameters for the internal switch."""
        return rc.get_sparams(
            self.calkit.open.reflection_coefficient(self.freq * units.MHz),
            self.calkit.short.reflection_coefficient(self.freq * units.MHz),
            self.calkit.match.reflection_coefficient(self.freq * units.MHz),
            self.corrections["open"],
            self.corrections["short"],
            self.corrections["match"],
        )

    def _get_reflection_model(self, kind: str) -> Model:
        # 'kind' should be 's11', 's12' or 's22'
        data = getattr(self, f"{kind}_data")
        return getattr(self, f"_{kind}_model").fit(xdata=self.freq, ydata=data)


@attr.s(kw_only=True, frozen=True)
class LoadPlusSwitchS11(_UncorrectedS11):
    """S11 for a lab calibration load including the internal switch.

    Note that this class is generally not used directly, as we require the S11 of the
    load after *correcting* for the switch. See :class:`LoadS11` for this.

    Parameters
    ----------
    internal_switch : :class:`s11.InternalSwitch`
        The internal switch state corresponding to the load.

    Other Parameters
    ----------------
    Passed through to :class:`_S11Base`.
    """

    @classmethod
    def from_io(
        cls,
        load_io: io.LoadS11,
        f_low=0 * un.MHz,
        f_high=np.inf * un.MHz,
        **kwargs,
    ):
        """
        Create a new object from a given path and load name.

        Parameters
        ----------
        load_io
            The io.LoadS11 that this will be based off.
        f_low, f_high
            Min/max frequencies to keep in the modelling.

        Returns
        -------
        s11 : :class:`LoadPlusSwitchS11`
            The S11 of the load + internal switch.
        """
        return cls(
            standards=StandardsReadings.from_io(load_io, f_low=f_low, f_high=f_high),
            **kwargs,
        )

    @cached_property
    def calibrated_s11_raw(self):
        """The measured S11 of the load, calculated from raw internal standards."""
        return rc.de_embed(
            self.calkit.open.intrinsic_gamma,
            self.calkit.short.intrinsic_gamma,
            0.0,  # TODO: self.calkit.match.intrinsic_gamma, ??
            self.open.s11,
            self.short.s11,
            self.match.s11,
            self.external.s11,
        )[0]

    @cached_property
    def corrected_s11(self) -> np.ndarray:
        """The measured S11 of the load, corrected for the internal switch."""
        return rc.gamma_de_embed(
            self.internal_switch.s11_model(self.freq.freq),
            self.internal_switch.s12_model(self.freq.freq),
            self.internal_switch.s22_model(self.freq.freq),
            self.calibrated_s11_raw,
        )


class LoadS11(_S11Base):
    load_s11: LoadPlusSwitchS11 = attr.ib(type=LoadPlusSwitchS11)
    internal_switch: InternalSwitch = attr.ib(type=InternalSwitch)

    @internal_switch.validator
    def _isw_vld(self, att, val):
        if val.calkit != self.load_s11.calkit:
            raise ValueError(
                "The calkit used for the internal switch must match the that "
                "used for the load."
            )

    @cached_property
    def calibrated_s11_raw(self):
        """The calibrated S11 at raw frequencies (with noise)."""
        return rc.gamma_de_embed(
            self.internal_switch.s11_model(self.freq.freq),
            self.internal_switch.s12_model(self.freq.freq),
            self.internal_switch.s22_model(self.freq.freq),
            self.load_s11.calibrated_s11_raw,
        )

    @classmethod
    def from_io(
        cls,
        s11_io: io.S11Dir,
        load_name: str,
        internal_switch_kwargs=None,
        load_kw=None,
    ) -> LoadS11:
        """Instantiate from an :class:`edges_io.io.S11Dir` object."""
        internal_switch = InternalSwitch.from_io(
            s11_io.children["switchingstate"], **(internal_switch_kwargs or {})
        )
        return cls(
            load_s11=LoadPlusSwitchS11.from_io(
                s11_io.children[load_name], **(load_kw or {})
            ),
            internal_switch=internal_switch,
        )


def _read_data_and_corrections(switching_state: io.SwitchingState):

    # Standards assumed at the switch
    sw = {
        "open": 1 * np.ones_like(switching_state.freq),
        "short": -1 * np.ones_like(switching_state.freq),
        "match": np.zeros_like(switching_state.freq),
    }

    return {
        kind: rc.de_embed(
            sw["open"],
            sw["short"],
            sw["match"],
            getattr(switching_state, "open").s11,
            getattr(switching_state, "short").s11,
            getattr(switching_state, "match").s11,
            getattr(switching_state, "external%s" % kind).s11,
        )[0]
        for kind in sw
    }
