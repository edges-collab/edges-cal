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
from astropy import units as un
from cached_property import cached_property
from edges_io import h5, io
from pathlib import Path
from typing import Callable, Sequence

from . import reflection_coefficient as rc
from . import types as tp
from .modelling import (
    ComplexMagPhaseModel,
    ComplexRealImagModel,
    Fourier,
    Model,
    Modelable,
    Polynomial,
    UnitTransform,
    get_mdl,
)
from .tools import FrequencyRange, vld_unit


def _s1p_converter(s1p: tp.PathLike | io.S1P, check: bool = False) -> io.S1P:
    try:
        s1p = Path(s1p)
        return io.S1P(s1p, check=check)
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
    _s11: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    f_low: tp.FreqType = attr.ib(
        default=0 * un.MHz, validator=vld_unit("frequency"), kw_only=True
    )
    f_high: float | None = attr.ib(
        default=np.inf * un.MHz, validator=vld_unit("frequency"), kw_only=True
    )

    @_s11.validator
    def _s11_vld(self, att, val):
        if not np.iscomplexobj(val):
            raise ValueError("s11 must be a complex quantity")

        if val.shape != self._freq.shape:
            raise ValueError(
                "freq and s11 must have the same length! "
                f"Got {len(self._freq)} and {len(val)}"
            )

    @cached_property
    def freq(self) -> FrequencyRange:
        """The frequencies of the S11 measurement."""
        return FrequencyRange(self._freq, f_low=self.f_low, f_high=self.f_high)

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
        return cls(s1p.freq, s1p.s11, **kwargs)


@attr.s
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


@h5.hickleable()
@attr.s(kw_only=True, frozen=True)
class S11Model:
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

    _default_nterms = 55
    _complex_model_type_default = ComplexMagPhaseModel
    _model_type_default = "fourier"

    _raw_s11: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    freq: FrequencyRange = attr.ib()
    n_terms: int = attr.ib(converter=int)
    model_type: Modelable = attr.ib()
    complex_model_type: type[ComplexMagPhaseModel] | type[
        ComplexRealImagModel
    ] = attr.ib()
    model_delay: tp.Time = attr.ib(0 * un.s)

    metadata: dict = attr.ib(default=attr.Factory(dict), eq=False)

    @freq.validator
    def _fv(self, att, val):
        if val.n != len(self._raw_s11) and len(val._f) != len(self._raw_s11):
            raise ValueError(
                f"len(freq) != len(raw_s11) [{len(val)},{len(self._raw_s11)}]"
            )

    @property
    def raw_s11(self):
        """The raw S11 measurements at different frequencies."""
        if len(self._raw_s11) == self.freq.n:
            return self._raw_s11
        else:
            return self._raw_s11[self.freq.mask]

    @model_type.default
    def _mdl_type_default(self):
        return self._model_type_default

    @complex_model_type.default
    def _cmt_default(self):
        return self._complex_model_type_default

    @n_terms.default
    def _nt_default(self):
        return int(self._default_nterms)

    @n_terms.validator
    def _nt_vld(self, att, val):
        if get_mdl(self.model_type) == Fourier and not val % 2:
            raise ValueError(
                f"n_terms must be odd for Fourier models. For {self} got "
                f"n_terms={val}."
            )

    def clone(self, **kwargs):
        """Clone with new parameters."""
        return attr.evolve(self, **kwargs)

    def get_s11_model(
        self,
        raw_s11: np.ndarray,
        *,
        freq: tp.FreqType | None = None,
        n_terms: int | None = None,
        model_type: Modelable | None = None,
    ) -> ComplexMagPhaseModel | ComplexRealImagModel:
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
            transform=UnitTransform(
                range=[self.freq.min.to_value("MHz"), self.freq.max.to_value("MHz")]
            ),
        )
        emodel = model.at(x=freq.to_value("MHz"))

        cmodel = self.complex_model_type(emodel, emodel)

        return cmodel.fit(
            ydata=raw_s11 * np.exp(1j * self.model_delay * freq).to_value("")
        )

    @cached_property
    def _s11_model(self) -> callable:
        """The S11 model."""
        return self.get_s11_model(self.raw_s11)

    def s11_model(self, freq: np.ndarray | tp.FreqType) -> np.ndarray:
        """Compute the S11 at a specific set of frequencies."""
        if hasattr(freq, "unit"):
            freq = freq.to_value("MHz")

        return self._s11_model(freq) * np.exp(
            -1j * self.model_delay.to_value("microsecond") * freq
        )

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

        corr = self.raw_s11
        model = self.s11_model(self.freq.freq.to_value("MHz"))

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

        lname = (
            self.load_name if hasattr(self, "load_name") else self.__class__.__name__
        )

        if title is None:
            title = f"{lname} Reflection Coefficient Models"

        if title:
            fig.suptitle(f"{lname} Reflection Coefficient Models", fontsize=14)
        if label:
            ax[0].legend()

        return fig


@h5.hickleable()
@attr.s(kw_only=True)
class Receiver(S11Model):
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

    _default_nterms = 37

    @classmethod
    def from_io(
        cls,
        device: io.ReceiverReading,
        calkit: rc.Calkit = rc.AGILENT_85033E,
        resistance: float | None = None,
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
        if resistance is not None:
            calkit = rc.get_calkit(calkit, resistance_of_match=resistance)

        if not hasattr(device, "__len__"):
            device = [device]

        s11s = []
        for dv in device:
            standards = StandardsReadings.from_io(dv, f_low=f_low, f_high=f_high)
            receiver_reading = VNAReading.from_s1p(
                dv.children["receiverreading"], f_low=f_low, f_high=f_high
            )
            freq = standards.freq

            calibrated_s11_raw = rc.de_embed(
                calkit.open.reflection_coefficient(freq.freq),
                calkit.short.reflection_coefficient(freq.freq),
                calkit.match.reflection_coefficient(freq.freq),
                standards.open.s11,
                standards.short.s11,
                standards.match.s11,
                receiver_reading.s11,
            )[0]
            s11s.append(calibrated_s11_raw)

        metadata = {"devices": device, "calkit": calkit}

        return cls(
            raw_s11=np.mean(s11s, axis=0), freq=freq, metadata=metadata, **kwargs
        )

    def with_new_calkit(self, calkit: rc.Calkit):
        """Get a new Receiver with different calkit."""
        if "devices" not in self.metadata:
            raise RuntimeError("Cannot clone with new calkit, since it isn't known.")

        fields = attr.fields(self.__class__)
        kwargs = {
            k.name: getattr(self, k.name)
            for k in fields
            if k.name not in ("_raw_s11", "freq", "metadata")
        }

        return self.from_io(
            device=self.metadata["devices"],
            calkit=calkit,
            f_low=self.freq.min,
            f_high=self.freq.max,
            **kwargs,
        )


@h5.hickleable()
@attr.s
class InternalSwitch:
    _freq: tp.Freqtype = attr.ib(
        validator=vld_unit("frequency"), eq=attr.cmp_using(eq=np.array_equal)
    )
    s11_data: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    s12_data: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    s22_data: np.ndarray = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    model: Model = attr.ib()
    n_terms: tuple[int, int, int] | int = attr.ib(default=(7, 7, 7), converter=_tuplify)
    metadata: dict = attr.ib(default=attr.Factory(dict), eq=False)

    @property
    def freq(self):
        """Frequencies of the measurement."""
        return FrequencyRange(self._freq)  # , f_low=self.f_low, f_high=self.f_high)

    @model.default
    def _mdl_default(self):
        return Polynomial(
            n_terms=7,
            transform=UnitTransform(
                range=(self.freq.min.to_value("MHz"), self.freq.max.to_value("MHz"))
            ),
        )

    @classmethod
    def from_io(
        cls,
        internal_switch: io.InternalSwitch | Sequence[io.InternalSwitch],
        calkit=rc.AGILENT_85033E,
        resistance=None,
        f_low=0 * un.MHz,
        f_high=np.inf * un.MHz,
        **kwargs,
    ) -> InternalSwitch:
        """Initiate from an edges-io object."""
        if not hasattr(internal_switch, "__len__"):
            internal_switch = [internal_switch]

        if resistance is not None:
            calkit = rc.get_calkit(calkit, resistance_of_match=resistance)

        freq = FrequencyRange(internal_switch[0].freq, f_low=f_low, f_high=f_high)

        # TODO: not clear why we use the ideal values of 1,-1,0 instead of the physical
        # expected values of calkit.match.intrinsic_gamma etc.
        corrections = []
        for isw in internal_switch:
            corrections.append(
                {
                    kind: rc.de_embed(
                        1,
                        -1,
                        0,
                        isw.open.s11,
                        isw.short.s11,
                        isw.match.s11,
                        getattr(isw, "external%s" % kind).s11,
                    )[0]
                    for kind in ("open", "short", "match")
                }
            )

        s11, s12, s22 = cls.get_sparams_from_corrections(freq, corrections, calkit)

        metadata = {
            "calkit": calkit,
            "internal_switches": internal_switch,
            "corrections": corrections,
        }

        return cls(
            freq=freq.freq,
            s11_data=s11,
            s12_data=s12,
            s22_data=s22,
            metadata=metadata,
            **kwargs,
        )

    @staticmethod
    def get_sparams_from_corrections(freq, corrections, calkit):
        """Get S-parameters from a set of measured corrections."""
        s11s, s12s, s22s = [], [], []

        for cc in corrections:
            s11, s12s21, s22 = rc.get_sparams(
                calkit.open.reflection_coefficient(freq.freq),
                calkit.short.reflection_coefficient(freq.freq),
                calkit.match.reflection_coefficient(freq.freq),
                cc["open"][freq.mask],
                cc["short"][freq.mask],
                cc["match"][freq.mask],
            )
            s11s.append(s11)
            s12s.append(s12s21)
            s22s.append(s22)

        return np.mean(s11s, axis=0), np.mean(s12s, axis=0), np.mean(s22s, axis=0)

    def with_new_calkit(self, calkit: rc.Calkit):
        """Obtain a new InternalSwitch using a new calkit."""
        if "corrections" not in self.metadata:
            raise RuntimeError(
                "Cannot update calkit when the object was not made with a calkit."
            )

        s11, s12, s22 = self.get_sparams_from_corrections(
            self.freq, self.metadata["corrections"], calkit
        )
        return attr.evolve(self, s11_data=s11, s12_data=s12, s22_data=s22)

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
        if "calkit" in self.metadata:
            return self.metadata["calkit"]
        else:
            raise AttributeError("calkit not known!")

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

    def _get_reflection_model(self, kind: str) -> Model:
        # 'kind' should be 's11', 's12' or 's22'
        data = getattr(self, f"{kind}_data")
        return getattr(self, f"_{kind}_model").fit(
            xdata=self.freq.freq.to_value("MHz"), ydata=data
        )


@h5.hickleable()
@attr.s(kw_only=True, frozen=True)
class LoadPlusSwitchS11:
    """S11 for a lab calibration load including the internal switch.

    Note that this class is generally not used directly, as we require the S11 of the
    load after *correcting* for the switch. See :class:`LoadS11` for this.

    Parameters
    ----------
    standards
        The internal VNA standards readings of the full system (input + internalswitch).
    external_match
        The reading of the VNA from the match standard as applied externally.
    load_name
        Optional name for the input device.

    Other Parameters
    ----------------
    Passed through to :class:`_S11Base`.
    """

    standards: StandardsReadings = attr.ib()
    external_match: VNAReading = attr.ib(
        validator=attr.validators.instance_of(VNAReading)
    )
    load_name: str | None = attr.ib(None)

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
            external_match=VNAReading.from_s1p(
                load_io.children["external"], f_low=f_low, f_high=f_high
            ),
            load_name=load_io.load_name,
            **kwargs,
        )

    def get_calibrated_s11(self):
        """The measured S11 of the load, calculated from raw internal standards."""
        # TODO: It's not clear exactly why we use the completely ideal values for the
        # OSL standards here, instead of their predicted values from physical parameters
        # eg. calkit.match.intrinsic_gamma.
        return rc.de_embed(
            1,
            -1,
            0.0,
            self.standards.open.s11,
            self.standards.short.s11,
            self.standards.match.s11,
            self.external_match.s11,
        )[0]

    @property
    def freq(self):
        """Frequencies of the observation."""
        return self.standards.open.freq


@h5.hickleable()
@attr.s
class LoadS11(S11Model):
    """S11 of an input Load."""

    internal_switch: InternalSwitch = attr.ib()
    load_name: str | None = attr.ib(None)

    @classmethod
    def from_load_and_internal_switch(
        cls,
        load_s11: LoadPlusSwitchS11 | Sequence[LoadPlusSwitchS11],
        internal_switch: InternalSwitch,
        base: LoadS11 | None = None,
        **kwargs,
    ) -> LoadS11:
        """Generate the LoadS11 from an uncalibrated load and internal switch."""
        if not hasattr(load_s11, "__len__"):
            load_s11 = [load_s11]

        freq = load_s11[0].freq

        default_nterms = {
            "ambient": 37,
            "hot_load": 37,
            "open": 105,
            "short": 105,
        }

        if "n_terms" not in kwargs:
            kwargs["n_terms"] = default_nterms.get(
                load_s11[0].load_name, cls._default_nterms
            )

        s11s = []
        for load in load_s11:
            gamma = rc.gamma_de_embed(
                internal_switch.s11_model(freq.freq.to_value("MHz")),
                internal_switch.s12_model(freq.freq.to_value("MHz")),
                internal_switch.s22_model(freq.freq.to_value("MHz")),
                load.get_calibrated_s11(),
            )
            s11s.append(gamma)

        metadata = {"load_s11s": load_s11}
        metadata.update(getattr(internal_switch, "metadata", {}))

        if base is None:
            return cls(
                freq=freq,
                raw_s11=np.mean(s11s, axis=0),
                metadata=metadata,
                load_name=load_s11[0].load_name,
                internal_switch=internal_switch,
                **kwargs,
            )
        else:
            return attr.evolve(
                base,
                freq=freq.freq,
                raw_s11=np.mean(s11s, axis=0),
                metadata=metadata,
                load_name=load_s11[0].load_name,
            )

    @classmethod
    def from_io(
        cls,
        s11_io: io.S11Dir,
        load_name: str,
        internal_switch_kwargs=None,
        load_kw=None,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        **kwargs,
    ) -> LoadS11:
        """Instantiate from an :class:`edges_io.io.S11Dir` object."""
        internal_switch_kwargs = internal_switch_kwargs or {}
        internal_switch_kwargs["f_low"] = f_low
        internal_switch_kwargs["f_high"] = f_high

        load_kw = load_kw or {}
        load_kw["f_low"] = f_low
        load_kw["f_high"] = f_high

        internal_switch = InternalSwitch.from_io(
            s11_io.switching_state, **internal_switch_kwargs
        )

        load_s11s = []
        for xx in getattr(s11_io, load_name):
            load_s11s.append(LoadPlusSwitchS11.from_io(xx, **load_kw))

        return cls.from_load_and_internal_switch(
            load_s11=load_s11s, internal_switch=internal_switch, **kwargs
        )
