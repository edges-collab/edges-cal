"""Tools to use in other modules."""
from __future__ import annotations

import attr
import numpy as np
import warnings
from itertools import product
from pathlib import Path
from typing import Sequence

from . import DATA_PATH
from .cached_property import cached_property


def get_data_path(pth: str | Path) -> Path:
    """Impute the global data path to a given input in place of a colon."""
    if isinstance(pth, str):
        if pth.startswith(":"):
            return DATA_PATH / pth[1:]
        else:
            return Path(pth)
    else:
        return pth


def as_readonly(x: np.ndarray) -> np.ndarray:
    """Get a read-only view into an array without copying."""
    result = x.view()
    result.flags.writeable = False
    return result


def dct_of_list_to_list_of_dct(dct: dict[str, Sequence]) -> list[dict]:
    """Take a dict of key: list pairs and turn it into a list of all combinations of dicts.

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

    return [{k: v for k, v in zip(dct.keys(), p)} for p in prod]


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
    bin_size
        Bin input frequencies into bins of this size.
    """

    _f: np.ndarray = attr.ib(converter=np.array)
    _f_low: float = attr.ib(converter=attr.converters.optional(float), kw_only=True)
    _f_high: float = attr.ib(converter=attr.converters.optional(float), kw_only=True)
    bin_size: int = attr.ib(default=1, converter=int, kw_only=True)

    @_f_low.default
    def _flow_default(self):
        return self._f.min()

    @_f_high.default
    def _fhigh_default(self):
        return self._f.max()

    @bin_size.validator
    def _bin_size_validator(self, att, val):
        if val < 1:
            raise ValueError("Cannot use bin_size < 1")

        if val > self._f.size:
            raise ValueError(
                "Cannot use bin_size larger than the total number of freqs."
            )

    @_f.validator
    def _f_validator(self, att, val):
        if np.any(val < 0):
            raise ValueError("Cannot have negative input frequencies!")
        if val.ndim > 1:
            raise ValueError("Frequency array must be 1D!")

    @_f_high.validator
    def _fhigh_validator(self, att, val):
        if val <= self._f_low:
            raise ValueError("Cannot have f_high <= f_low.")

    @property
    def n(self) -> int:
        """The number of frequencies in the (masked) array."""
        return self.freq.size

    @cached_property
    def df(self) -> float:
        """Resolution of the frequencies."""
        if not np.allclose(np.diff(self.freq, 2), 0):
            warnings.warn(
                "Not all frequency intervals are even, so using df is ill-advised!"
            )
        return self.freq[1] - self.freq[0]

    @property
    def freq_full(self):
        """Alias for `f`."""
        return self._f

    @cached_property
    def min(self):  # noqa
        """Minimum frequency in the array."""
        return self.freq.min()

    @cached_property
    def max(self):  # noqa
        """Maximum frequency in the array."""
        return self.freq.max()

    @cached_property
    def mask(self):
        """Mask used to take input frequencies to output frequencies."""
        return np.logical_and(
            self.freq_full >= self._f_low, self.freq_full <= self._f_high
        )

    @cached_property
    def freq(self):
        """The frequency array."""
        return bin_array(self.freq_full[self.mask], self.bin_size)

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

    def normalize(self, f):
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
        return 2 * (f - self.center) / self.range

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
        cls, n_channels: int = 16384 * 2, max_freq: float = 200.0, **kwargs
    ) -> FrequencyRange:
        """Construct a :class:`FrequencyRange` object with underlying EDGES freqs.

        Parameters
        ----------
        n_channels : int
            Number of channels
        max_freq : float
            Maximum frequency in original measurement.
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
        max_freq = float(max_freq)

        if n_channels < 100:
            raise ValueError("Shouldn't have less than 100 channels for EDGES!")

        df = max_freq / n_channels

        # The final frequency here will be slightly less than 200 MHz. 200 MHz
        # corresponds to the centre of the N+1 bin, which doesn't actually exist.
        f = np.arange(0, max_freq, df)

        return cls(f=f, **kwargs)

    def clone(self, **kwargs):
        """Make a new frequency range object with updated parameters."""
        return attr.evolve(self, **kwargs)

    def with_new_mask(self, **kwargs):
        """Make a new read-only frequency range object with the same freqs."""
        f = as_readonly(self.freq_full)
        return attr.evolve(self, f=f, **kwargs)


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
