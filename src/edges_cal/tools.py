"""Tools to use in other modules."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any, Callable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import attr
import numpy as np
from astropy import units
from astropy import units as u
from edges_io import types as tp
from hickleable import hickleable
from pygsdata import GSData
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.ndimage import convolve1d

from . import DATA_PATH
from .cached_property import cached_property


def get_data_path(pth: str | Path) -> Path:
    """Impute the global data path to a given input in place of a colon."""
    if isinstance(pth, str):
        return DATA_PATH / pth[1:] if pth.startswith(":") else Path(pth)
    return pth


def is_unit(unit: str) -> bool:
    """Whether the given input is a recognized unit."""
    if isinstance(unit, u.Unit):
        return True

    try:
        u.Unit(unit)
        return True
    except ValueError:
        return False


def vld_unit(
    unit: str | u.Unit, equivalencies=()
) -> Callable[[Any, attr.Attribute, Any], None]:
    """Attr validator to check physical type."""
    utype = is_unit(unit)
    if not utype:
        # must be a physical type. This errors with ValueError if unit is not
        # really a physical type.
        u.get_physical_type(unit)

    def _check_type(self: Any, att: attr.Attribute, val: Any):
        if not isinstance(val, u.Quantity):
            raise TypeError(f"{att.name} must be an astropy Quantity!")

        if utype and not val.unit.is_equivalent(unit, equivalencies):
            raise u.UnitConversionError(
                f"{att.name} not convertible to {unit}. Got {val.unit}"
            )

        if not utype and val.unit.physical_type != unit:
            raise u.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. "
                f"Got '{val.unit.physical_type}'"
            )

    return _check_type


def unit_convert_or_apply(
    x: float | units.Quantity,
    unit: str | units.Unit,
    in_place: bool = False,
    warn: bool = False,
) -> units.Quantity:
    """Safely convert a given value to a quantity."""
    if warn and not isinstance(x, units.Quantity):
        warnings.warn(
            f"Value passed without units, assuming '{unit}'. "
            "Consider specifying units for future compatibility.",
            stacklevel=2,
        )

    return units.Quantity(x, unit, copy=not in_place)


def unit_converter(
    unit: str | units.Unit,
) -> Callable[[float | units.Quantity], units.Quantity]:
    """Return a function that will convert values to a given quantity."""
    return lambda x: unit_convert_or_apply(x, unit)


def as_readonly(x: np.ndarray) -> np.ndarray:
    """Get a read-only view into an array without copying."""
    result = x.view()
    result.flags.writeable = False
    return result


def dct_of_list_to_list_of_dct(dct: dict[str, Sequence]) -> list[dict]:
    """Take a dict of key: list pairs and turn it into a list of all combos of dicts.

    Parameters
    ----------
    dct
        A dictionary for which each value is an iterable.

    Returns
    -------
    list
        A list of dictionaries, each having the same keys as the input ``dct``, but
        in which the values are the elements of the original iterables.

    Examples
    --------
    >>> dct_of_list_to_list_of_dct(
    >>>    { 'a': [1, 2], 'b': [3, 4]}
    [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
    """
    lists = dct.values()

    prod = product(*lists)

    return [dict(zip(dct.keys(), p)) for p in prod]


@hickleable()
@attr.s
class FrequencyRange:
    """
    Class defining a set of frequencies.

    A given frequency range can be cut on either end, and be made more sparse.

    Parameters
    ----------
    f
        An array of frequencies defining a given spectrum.
    f_low
        A minimum frequency to keep in the array. Default is min(f).
    f_high
        A minimum frequency to keep in the array. Default is min(f).
    """

    _f: tp.FreqType = attr.ib(
        eq=attr.cmp_using(eq=np.array_equal), validator=vld_unit("frequency")
    )
    _f_low: tp.FreqType = attr.ib(
        0 * u.MHz, validator=vld_unit("frequency"), kw_only=True
    )
    _f_high: float = attr.ib(
        np.inf * u.MHz, validator=vld_unit("frequency"), kw_only=True
    )

    @_f.validator
    def _f_validator(self, att, val):
        if np.any(val < 0 * u.MHz):
            raise ValueError("Cannot have negative input frequencies!")
        if val.ndim > 1:
            raise ValueError("Frequency array must be 1D!")
        if np.any(np.diff(val) < 0):
            raise ValueError("Input frequencies must be in increasing order!")

    @_f_high.validator
    def _fhigh_validator(self, att, val):
        if val <= self._f_low:
            raise ValueError("Cannot have f_high <= f_low.")

    @property
    def nfull(self) -> int:
        """The number of frequencies in the full array."""
        return self.freq_full.size

    @property
    def n(self) -> int:
        """The number of frequencies in the (masked) array."""
        return self.freq.size

    @cached_property
    def df(self) -> float:
        """Resolution of the frequencies."""
        if not np.allclose(np.diff(self.freq, 2), 0):
            warnings.warn(
                "Not all frequency intervals are even, so using df is ill-advised!",
                stacklevel=2,
            )
        return self.freq[1] - self.freq[0]

    @property
    def freq_full(self):
        """Alias for `f`."""
        return self._f

    @property
    def min(self):
        """Minimum frequency in the array."""
        return self.freq[0]

    @property
    def max(self):
        """Maximum frequency in the array."""
        return self.freq[-1]

    @cached_property
    def mask(self) -> slice:
        """Mask used to take input frequencies to output frequencies."""
        return slice(
            np.nonzero(self.freq_full >= self._f_low)[0][0],
            np.nonzero(self.freq_full <= self._f_high)[0][-1] + 1,
        )

    @property
    def freq(self):
        """The frequency array."""
        return self.freq_full[self.mask]

    @cached_property
    def range(self):
        """Full range of the frequencies."""
        return self.max - self.min

    @cached_property
    def center(self):
        """The center of the frequency array."""
        return self.min + self.range / 2.0

    @cached_property
    def freq_recentred(self):
        """The frequency array re-centred so that it extends from -1 to 1."""
        return self.normalize(self.freq)

    def normalize(self, f) -> np.ndarray:
        """
        Normalise a set of frequencies.

        Normalizes such that -1 aligns with ``min`` and +1 aligns with ``max``.

        Parameters
        ----------
        f : array_like
            Frequencies to normalize

        Returns
        -------
        array_like, shape [f,]
            The normalized frequencies.
        """
        return (2 * (f - self.center) / self.range).value

    def denormalize(self, f):
        """
        De-normalise a set of frequencies.

        Normalizes such that -1 aligns with ``min`` and +1 aligns with ``max``.

        Parameters
        ----------
        f : array_like
            Frequencies to de-normalize

        Returns
        -------
        array_like, shape [f,]
            The de-normalized frequencies.
        """
        return f * self.range / 2 + self.center

    @classmethod
    def from_edges(
        cls,
        n_channels: int = 16384 * 2,
        max_freq: float = 200.0 * u.MHz,
        keep_full: bool = True,
        f_low=0 * u.MHz,
        f_high=np.inf * u.MHz,
        **kwargs,
    ) -> FrequencyRange:
        """Construct a :class:`FrequencyRange` object with underlying EDGES freqs.

        Parameters
        ----------
        n_channels : int
            Number of channels
        max_freq : float
            Maximum frequency in original measurement.
        keep_full
            Whether to keep the full underlying frequency array, or just the part
            of the array inside the mask.
        f_low, f_high
            A frequency range to keep.
        kwargs
            All other arguments passed through to :class:`FrequencyRange`.

        Returns
        -------
        :class:`FrequencyRange`
            The FrequencyRange object with the correct underlying frequencies.

        Notes
        -----
        This is correct. The channel width is the important thing.
        The channel width is given by the FFT. We actually take
        32678*2 samples of data at 400 Mega-samples per second.
        We only use the first half of the samples (since it's real input).
        Regardless, the frequency channel width is thus
        400 MHz / (32678*2) == 200 MHz / 32678 ~ 6.103 kHz

        """
        n_channels = int(n_channels)

        if n_channels < 100:
            raise ValueError("Shouldn't have less than 100 channels for EDGES!")

        df = max_freq / n_channels

        # The final frequency here will be slightly less than 200 MHz. 200 MHz
        # corresponds to the centre of the N+1 bin, which doesn't actually exist.
        f = np.arange(0, max_freq.value, df.value) * max_freq.unit

        if not keep_full:
            f = f[(f >= f_low) * (f <= f_high)]

        return cls(f=f, f_low=f_low, f_high=f_high, **kwargs)

    def clone(self, **kwargs):
        """Make a new frequency range object with updated parameters."""
        return attr.evolve(self, **kwargs)

    def decimate(
        self, bin_size: int, decimate_at: int | str = "centre", embed_mask: bool = True
    ) -> Self:
        """Decimate the frequency array.

        Parameters
        ----------
        bin_size
            The number of raw bins to combine into one.
        decimate_at
            Where to start the decimation from. If 'centre', then the new
            frequency array will be the mean frequency in each bin (equivalent
            to `decimate_at=bin_size//2` for odd `bin_size`, but for even bin
            size, the new frequencies are between the central two of the bin).
            If an integer, then the decimation starts at that index.
        embed_mask
            Whether to embed the mask in the new frequency array. IF False, the
            returned object will have an underlying frequency array with the full
            range of data, but with a mask similar to this object. If True, the
            returned object will have the frequencies outside the mask range
            removed completely.
        """
        freq = self.freq if embed_mask else self._f

        if decimate_at == "centre":
            new_freq = bin_array(freq, size=bin_size)
        else:
            new_freq = freq[decimate_at::bin_size]

        return FrequencyRange(
            f=new_freq,
            f_low=0 * u.MHz if embed_mask else self._f_low,
            f_high=np.inf * u.MHz if embed_mask else self._f_high,
        )


def bin_array(x: np.ndarray, size: int = 1) -> np.ndarray:
    """Simple unweighted mean-binning of an array.

    Parameters
    ----------
    x
        The array to be binned. Only the last axis will be binned.
    size
        The size of the bins.

    Notes
    -----
    The last axis of `x` is binned. It is assumed that the coordinates corresponding
    to `x` are regularly spaced, so the final average just takes `size` values and
    averages them together.

    If the array is not divisible by `size`, the last values are left out.

    Examples
    --------
    Simple 1D example::

        >>> x = np.array([1, 1, 2, 2, 3, 3])
        >>> bin_array(x, size=2)
        [1, 2, 3]

    The last remaining values are left out::

        >>> x = np.array([1, 1, 2, 2, 3, 3, 4])
        >>> bin_array(x, size=2)
        [1, 2, 3]
    """
    if size == 1:
        return x

    n = x.shape[-1]
    nn = size * (n // size)
    return np.nanmean(x[..., :nn].reshape(x.shape[:-1] + (-1, size)), axis=-1)


def gauss_smooth(
    x: np.ndarray, size: int, decimate_at: int | None = None
) -> np.ndarray:
    """Smooth x with a Gaussian function, and reduces the size of the array."""
    assert isinstance(size, int)

    if decimate_at is None:
        decimate_at = size // 2

    assert isinstance(decimate_at, int)
    assert decimate_at < size

    # This choice of size scaling corresponds to Alan's C code.
    y = np.arange(-size * 4, size * 4 + 1) * 2 / size
    window = np.exp(-(y**2) * 0.69)

    sums = convolve1d(x, window, mode="nearest")[..., decimate_at::size]
    wghts = convolve1d(np.ones_like(x), window, mode="nearest")[..., decimate_at::size]

    return sums / wghts


def dicke_calibration(data: GSData) -> GSData:
    """Calibrate field data using the Dicke switch data."""
    iant = data.loads.index("ant")
    iload = data.loads.index("internal_load")
    ilns = data.loads.index("internal_load_plus_noise_source")

    with np.errstate(divide="ignore", invalid="ignore"):
        q = (data.data[iant] - data.data[iload]) / (data.data[ilns] - data.data[iload])

    return data.update(
        data=q[np.newaxis],
        data_unit="uncalibrated",
        times=data.times[:, [iant]],
        lsts=data.lsts[:, [iant]],
        time_ranges=data.time_ranges[:, [iant]],
        lst_ranges=data.lst_ranges[:, [iant]],
        loads=("ant",),
        nsamples=data.nsamples[[iant]],
        flags={name: flag.any(axis="load") for name, flag in data.flags.items()},
        residuals=None,
        effective_integration_time=data.effective_integration_time[[iant]],
    )


def temperature_thermistor(
    resistance: float | np.ndarray,
    coeffs: str | Sequence = "oven_industries_TR136_170",
    kelvin: bool = True,
):
    """
    Convert resistance of a thermistor to temperature.

    Uses a pre-defined set of standard coefficients.

    Parameters
    ----------
    resistance : float or array_like
        The measured resistance (Ohms).
    coeffs : str or len-3 iterable of floats, optional
        If str, should be an identifier of a standard set of coefficients, otherwise,
        should specify the coefficients.
    kelvin : bool, optional
        Whether to return the temperature in K or C.

    Returns
    -------
    float or array_like
        The temperature for each `resistance` given.
    """
    # Steinhart-Hart coefficients
    _coeffs = {"oven_industries_TR136_170": [1.03514e-3, 2.33825e-4, 7.92467e-8]}

    if isinstance(coeffs, str):
        coeffs = _coeffs[coeffs]

    assert len(coeffs) == 3

    # TK in Kelvin
    temp = 1 / (
        coeffs[0]
        + coeffs[1] * np.log(resistance)
        + coeffs[2] * (np.log(resistance)) ** 3
    )

    # Kelvin or Celsius
    if kelvin:
        return temp
    return temp - 273.15


class ComplexSpline:
    """Return a complex spline object."""

    def __init__(self, x, y, **kwargs):
        self.real = Spline(x, y.real, **kwargs)
        self.imag = Spline(x, y.imag, **kwargs)

    def __call__(self, x):
        """Compute the interpolation at x."""
        return self.real(x) + 1j * self.imag(x)
