"""Module dealing with calibration spectra and thermistor measurements."""
from __future__ import annotations

import attr
import h5py
import hickle
import numpy as np
import warnings
from astropy import units as un
from datetime import datetime, timedelta
from edges_io import h5, io
from edges_io import utils as iou
from edges_io.h5 import HDF5RawSpectrum
from edges_io.logging import logger
from functools import partial
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from . import receiver_calibration_func as rcf
from . import tools
from . import types as tp
from . import xrfi
from .cached_property import cached_property
from .config import config
from .tools import FrequencyRange


def read_spectrum(
    spec_obj: Sequence[io.Spectrum],
    freq: FrequencyRange | None = None,
    ignore_times: float | int = 0,
) -> dict[str, np.ndarray]:
    """
    Read the contents of the spectrum files into memory.

    Removes a starting percentage of times, and masks out certain frequencies.

    Returns
    -------
    dict :
        A dictionary of the contents of the file. Usually p0, p1, p2 (un-normalised
        powers of source, load, and load+noise respectively), and Q (the
        uncalibrated ratio).
    """
    if freq is None:
        freq = FrequencyRange.from_edges()

    data = [o.data for o in spec_obj]

    n_times = sum(len(d["time_ancillary"]["times"]) for d in data)
    nfreq = np.sum(freq.mask)
    out = {
        "p0": np.empty((nfreq, n_times)),
        "p1": np.empty((nfreq, n_times)),
        "p2": np.empty((nfreq, n_times)),
        "Q": np.empty((nfreq, n_times)),
    }

    if ignore_times < 1:
        index_start_spectra = int(ignore_times * n_times)
    else:
        assert isinstance(ignore_times, int)
        index_start_spectra = ignore_times

    for key, val in out.items():
        nn = 0
        for d in data:
            n = len(d["time_ancillary"]["times"])
            val[:, nn : (nn + n)] = d["spectra"][key][freq.mask]
            nn += n

        out[key] = val[:, index_start_spectra:]

    return out


def get_spectrum_ancillary(
    spec_obj: Sequence[io.Spectrum], ignore_times_percent: float = 0
) -> dict[str, np.ndarray]:
    """Ancillary data from the spectrum measurements."""
    anc = [s.data["time_ancillary"] for s in spec_obj]

    n_times = sum(len(a["times"]) for a in anc)

    index_start_spectra = int((ignore_times_percent / 100) * n_times)

    return {
        key: np.hstack(tuple(a[key].T for a in anc)).T[index_start_spectra:]
        for key in anc[0]
    }


@h5.hickleable()
@attr.s
class ThermistorReadings:
    """
    Object containing thermistor readings.

    Parameters
    ----------
    data
        The data array containing the readings.
    ignore_times_percent
        The fraction of readings to ignore at the start of the observation. If greater
        than 100, will be interpreted as being a number of seconds to ignore.
    """

    _data: np.ndarray = attr.ib()
    mask: np.ndarray = attr.ib()

    @_data.validator
    def _data_vld(self, att, val):
        if "start_time" not in val.dtype.names:
            for key in ["time", "date", "load_resistance"]:
                if key not in val.dtype.names:
                    raise ValueError(
                        f"{key} must be in the data for ThermistorReadings"
                    )

    @mask.validator
    def _mask_vld(self, att, val):
        if len(val) != len(self._data):
            raise ValueError("mask must be the same length as data")
        if val.dtype != bool:
            raise TypeError("mask must be a boolean array")

    @mask.default
    def _mask_default(self):
        return np.ones(len(self._data), dtype=bool)

    @property
    def data(self):
        """The associated data, without initial ignored times."""
        return self._data[self.mask]

    @property
    def raw_data(self):
        """The associated data, without initial ignored times."""
        return self._data

    @classmethod
    def from_io(cls, resistance_obj: io.Resistance, **kwargs) -> ThermistorReadings:
        """Generate the object from an io.Resistance object."""
        return cls(data=resistance_obj.read()[0], **kwargs)

    def get_timestamps(self, with_mask: bool = True) -> list[datetime]:
        """Timestamps of all the thermistor measurements."""
        if with_mask:
            d = self.data
        else:
            d = self.raw_data

        if "time" in self._data.dtype.names:
            times = d["time"]
            dates = d["date"]
            times = [
                datetime.strptime(d + ":" + t, "%m/%d/%Y:%H:%M:%S")
                for d, t in zip(dates.astype(str), times.astype(str))
            ]
        else:
            times = [
                datetime.strptime(d.split(".")[0], "%m/%d/%Y %H:%M:%S")
                for d in d["start_time"].astype(str)
            ]

        return times

    def get_physical_temperature(self, with_mask: bool = True) -> np.ndarray:
        """The associated thermistor temperature in K."""
        if with_mask:
            return rcf.temperature_thermistor(self.data["load_resistance"])
        else:
            return rcf.temperature_thermistor(self.raw_data["load_resistance"])

    @cached_property
    def load_temperature(self) -> np.ndarray:
        """The associated thermistor temperature in K."""
        return self.get_physical_temperature()

    @cached_property
    def raw_load_temperature(self) -> np.ndarray:
        """The associated thermistor temperature in K."""
        return self.get_physical_temperature(False)

    def get_thermistor_indices(self, timestamps) -> list[int | np.nan]:
        """Get the index of the closest thermistor measurement for each timestamp."""
        closest = []
        indx = 0
        thermistor_timestamps = self.get_timestamps(False)

        deltat = thermistor_timestamps[1] - thermistor_timestamps[0]

        for d in timestamps:
            if indx >= len(thermistor_timestamps):
                closest.append(np.nan)
                continue

            for i, td in enumerate(thermistor_timestamps[indx:], start=indx):

                if timedelta(0) < d - td <= deltat:
                    closest.append(i)
                    break
                if d - td > timedelta(0):
                    indx += 1

            else:
                closest.append(np.nan)

        return closest

    def with_temp_range(self, temp_range: tuple[float, float] | float):
        """Return a new object with a mask applied to the temperature range."""
        if isinstance(temp_range, float):
            med = np.median(self.load_temperature)
            temp_range = (med - temp_range / 2, med + temp_range / 2)

        mask = self.mask & (self.raw_load_temperature > temp_range[0])
        mask = mask & (self.raw_load_temperature < temp_range[1])
        return attr.evolve(self, mask=mask)

    def with_median_filter(self, thresh: float = 5.0):
        """Return a new object with a median filter applied to the data."""
        mask = np.ones(len(self._data), dtype=bool)
        old_mask = np.zeros(len(self._data), dtype=bool)

        t = self.raw_load_temperature
        i = 0
        while not np.all(old_mask == mask) and i < 20:
            old_mask[:] = mask[:]
            tmask = t[self.mask & mask]
            med = np.median(tmask)
            mad = np.median(np.abs(tmask - med))
            mask = np.abs(t - med) < thresh * mad
            i += 1

        if i == 19:
            warnings.warn("Median filter did not converge")

        return attr.evolve(self, mask=mask & self.mask)

    def in_time_range(self, start: datetime, end: datetime):
        """Return a new object with a mask applied to the time range."""
        t = self.get_timestamps(False)
        mask = self.mask & np.array([start <= ti <= end for ti in t])
        return attr.evolve(self, mask=mask)

    def mask_timestamps(self, timestamps: list[datetime]) -> np.array:
        """Create a mask for a given list of timestamps, based on thermistory data."""
        mask = np.ones(len(timestamps), dtype=bool)
        closest = self.get_thermistor_indices(timestamps)
        for i, c in enumerate(closest):
            if np.isnan(c) or not self.mask[c]:
                mask[i] = False
        return mask


def get_ave_and_var_spec(
    spec_obj: list[HDF5RawSpectrum],
    load_name: str,
    freq,
    freq_bin_size,
    rfi_threshold,
    rfi_kernel_width_freq,
    frequency_smoothing: str,
    thermistor: ThermistorReadings,
    ignore_ninteg: int = 0,
    spec_timestamps: list[datetime] = None,
) -> tuple[dict, dict, int]:
    """Get the mean and variance of the spectra.

    Parameters
    ----------
    freqeuncy_smoothing
        How to average frequency bins together. Default is to merely bin them
        directly. Other options are 'gauss' to do Gaussian filtering (this is the
        same as Alan's C pipeline).
    """
    logger.info(f"Reducing {load_name} spectra...")
    spectra = read_spectrum(spec_obj=spec_obj, freq=freq, ignore_times=ignore_ninteg)

    if spec_timestamps is None:
        spec_timestamps = get_spec_timestamps(spec_obj)[ignore_ninteg:]

    means = {}
    variances = {}

    for key, spec in spectra.items():
        # Weird thing where there are zeros in the spectra.
        # For the Q-ratio, zero values are perfectly fine.
        if key.lower() != "q":
            spec[spec == 0] = np.nan

        if freq_bin_size > 1:
            if frequency_smoothing == "bin":
                spec = tools.bin_array(spec.T, size=freq_bin_size).T
            elif frequency_smoothing == "gauss":
                # We only really allow Gaussian smoothing so that we can match Alan's
                # pipeline. In that case, the frequencies actually kept start from the
                # 0th index, instead of taking the centre of each new bin. Thus we
                # set decimate_at = 0.
                spec = tools.gauss_smooth(spec.T, size=freq_bin_size, decimate_at=0).T
            else:
                raise ValueError("frequency_smoothing must be one of ('bin', 'gauss').")

        temp_mask = thermistor.mask_timestamps(spec_timestamps)
        spec[:, ~temp_mask] = np.nan

        mean = np.nanmean(spec, axis=1)
        var = np.nanvar(spec, axis=1)

        nsample = np.sum(~np.isnan(spec), axis=1)

        width = max(1, rfi_kernel_width_freq // freq_bin_size)

        varfilt = xrfi.flagged_filter(var, size=2 * width + 1)
        resid = mean - xrfi.flagged_filter(mean, size=2 * width + 1)
        flags = np.logical_or(
            resid > rfi_threshold * np.sqrt(varfilt / nsample),
            var - varfilt > rfi_threshold * np.sqrt(2 * varfilt**2 / (nsample - 1)),
        )

        mean[flags] = np.nan
        var[flags] = np.nan

        means[key] = mean
        variances[key] = var

    return means, variances, nsample


def get_spec_timestamps(
    spec_obj: list[HDF5RawSpectrum], time_coordinate_swpos: int | tuple[int, int] = 0
) -> tuple[list[datetime], datetime]:
    """Get a list of timestamps for the given specta, as well as a zeroth stamp."""
    spec_anc = get_spectrum_ancillary(spec_obj, ignore_times_percent=0)

    try:
        base_time, time_coordinate_swpos = time_coordinate_swpos
    except Exception:
        base_time = time_coordinate_swpos

    spec_timestamps = spec_obj[0].data.get_times(
        str_times=spec_anc["times"], swpos=time_coordinate_swpos
    )

    if base_time == time_coordinate_swpos:
        return spec_timestamps, spec_timestamps[0]
    else:
        return (
            spec_timestamps,
            spec_obj[0].data.get_times(
                str_times=spec_anc["times"][:1], swpos=base_time
            )[0],
        )


def _get_ignore_ninteg(
    spec_timestamps: dict[str, np.ndarray],
    spec_timestamp0: datetime = None,
    ignore: float | timedelta = 0,
) -> int:
    if spec_timestamp0 is None:
        spec_timestamp0 = spec_timestamps[0]
    if (
        isinstance(ignore, timedelta)
        or isinstance(ignore, un.Quantity)
        or ignore > 100.0
    ):
        # Interpret as a number of seconds.
        if not isinstance(ignore, timedelta):
            ignore = timedelta(
                seconds=ignore.to_value("s")
                if isinstance(ignore, un.Quantity)
                else ignore
            )

        # The first time could be measured from a different swpos than the one we are
        # measuring it to.
        for i, t in enumerate(spec_timestamps):
            if (t - spec_timestamp0) > ignore:
                break
        ignore_ninteg = i
    else:
        ignore_ninteg = int(len(spec_timestamps) * ignore / 100.0)

    return ignore_ninteg


@h5.hickleable()
@attr.s(kw_only=True, frozen=True)
class LoadSpectrum:
    """A class representing a measured spectrum from some Load averaged over time.

    Parameters
    ----------
    freq
        The frequencies associated with the spectrum.
    q
        The measured power-ratios of the three-position switch averaged over time.
    variance
        The variance of *a single* time-integration as a function of frequency.
    n_integrations
        The number of integrations averaged over.
    temp_ave
        The average measured physical temperature of the load while taking spectra.
    t_load_ns
        The "assumed" temperature of the load + noise source
    t_load
        The "assumed" temperature of the load
    _metadata
        A dictionary of metadata items associated with the spectrum.
    """

    freq: FrequencyRange = attr.ib()
    q: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    variance: np.ndarray | None = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    n_integrations: np.array = attr.ib()
    temp_ave: float = attr.ib()
    _metadata: dict[str, Any] = attr.ib(default=attr.Factory(dict))

    @q.validator
    @variance.validator
    def _q_vld(self, att, val):
        if val.shape != (self.freq.n,):
            raise ValueError(
                f"{att.name} must be one-dimensional with same length as un-masked "
                f"frequency. Got {val.shape}, needed ({self.freq.n},)"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata associated with the object."""
        return {
            **self._metadata,
            **{
                "n_integrations": self.n_integrations,
                "f_low": self.freq.min,
                "f_high": self.freq.max,
            },
        }

    @classmethod
    def from_h5(cls, path):
        """Read the contents of a .h5 file into a LoadSpectrum."""

        def read_group(grp):
            return cls(
                freq=FrequencyRange(grp["frequency"][...] * un.MHz),
                q=grp["Q_mean"][...],
                variance=grp["Q_var"],
                n_integrations=grp["n_integrations"],
                temp_ave=grp["temp_ave"],
                t_load_ns=grp["t_load_ns"],
                t_load=grp["t_load"],
                metadata=dict(grp.attrs),
            )

        if isinstance(path, (str, Path)):
            with h5py.File(path, "r") as fl:
                return read_group(fl)
        else:
            return read_group(path)

    @classmethod
    def _get_hash(cls, spec, res, dct: dict):
        dd = {**dct, "res": res, "spec": spec}

        return iou.stable_hash(tuple(dd.values()) + (__version__.split(".")[0],))

    @classmethod
    def _check_if_exists(
        cls, io_obs: io.CalibrationObservation, load_name: str, **kwargs
    ):
        """Check if the spectrum exists in the database."""
        spec = getattr(io_obs.spectra, load_name)
        res = getattr(io_obs.resistance, load_name)
        hsh = cls._get_hash(spec, res, kwargs)

        cache_dir = config["cal"]["cache-dir"]
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            fname = cache_dir / f"{load_name}_{hsh}.h5"

            if fname.exists():
                return fname

        return None

    @classmethod
    def from_io(
        cls,
        io_obs: io.CalibrationObservation,
        load_name: str,
        **kwargs,
    ):
        """Instantiate the class from a given load name and directory.

        Parameters
        ----------
        io_obs
            The CalibrationObservation object to load the spectrum from.
        load_name
            The load to read from the observation.

        Other Parameters
        ----------------
        Everything else is passed through to :meth:`UnaveragedLoadSpectrum.from_io`.

        Returns
        -------
        :class:`LoadSpectrum`.
        """
        fname = cls._check_if_exists(io_obs, load_name, **kwargs)
        if fname is not None:
            return hickle.load(fname)

        out = UnaveragedLoadSpectrum.from_io(
            io_obs, load_name, **kwargs
        ).to_load_spectrum()

        cache_dir = config["cal"]["cache-dir"]
        if cache_dir is not None:
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
            hickle.dump(out, fname)

        return out

    def between_freqs(self, f_low: tp.FreqType, f_high: tp.FreqType = np.inf * un.MHz):
        """Return a new LoadSpectrum that is masked between new frequencies."""
        mask = (self.freq.freq >= f_low) & (self.freq.freq <= f_high)
        return attr.evolve(
            self,
            freq=self.freq.with_new_mask(post_bin_f_low=f_low, post_bin_f_high=f_high),
            q=self.q[mask],
            variance=self.variance[mask],
        )

    @property
    def averaged_Q(self) -> np.ndarray:
        """Ratio of powers averaged over time.

        Notes
        -----
        The formula is

        .. math:: Q = (P_source - P_load)/(P_noise - P_load)
        """
        return self.q

    @property
    def variance_Q(self) -> np.ndarray:
        """Variance of Q across time (see averaged_Q)."""
        return self.variance


@h5.hickleable()
@attr.s(kw_only=True, frozen=True)
class UnaveragedLoadSpectrum:
    """Spectrum of a load over time, including thermistor measurements.

    Parameters
    ----------
    freq
        The frequencies associated with the spectrum.
    spec_times
        The times associated with the spectrum.
    q
        The measured power-ratios of the three-position switch averaged over time.
    p_ant
        The measured power of the antenna input.
    p_load
        The measured power of the internal load.
    p_ns
        The measured power of the internal noise source.
    spec_anc
        Ancillary data associated with the spectrum.
    flags
        Flags applied to the spectrum, e.g. RFI or times with no associated thermistor
        readings.
    temp_ave
        The average measured physical temperature of the load while taking spectra.
    thermistor
        The thermistor readings associated with the spectrum.
    metadata
        A dictionary of metadata items associated with the spectrum.
    """

    freq: FrequencyRange = attr.ib()
    spec_times: list[datetime.datetime] = attr.ib()

    q: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    p_ant: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    p_load: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    p_ns: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    spec_anc: dict[str, np.ndarray] = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )

    flags: np.ndarray = attr.ib(
        eq=attr.cmp_using(eq=partial(np.array_equal, equal_nan=True))
    )
    _temp_ave: float | None = attr.ib()

    thermistor: ThermistorReadings = attr.ib()
    metadata: dict[str, Any] = attr.ib(factory=dict)

    @q.validator
    @p_ant.validator
    @p_load.validator
    @p_ns.validator
    def _q_vld(self, att, val):
        if val.shape != (self.freq.n, len(self.spec_times)):
            raise ValueError(
                f"{att.name} must be two-dimensional with same length as un-masked "
                f"frequency. Got {val.shape}, needed ({self.freq.n}, "
                f"{len(self.spec_times)})"
            )

    @p_ant.validator
    @p_load.validator
    @p_ns.validator
    @flags.validator
    def _shape_vld(self, att, val):
        if val.shape != self.q.shape:
            raise ValueError(
                f"{att.name} must be two-dimensional with same shape as q. "
                f"Got {val.shape}, needed {self.q.shape}"
            )

    @classmethod
    def from_io(
        cls,
        io_obs: io.CalibrationObservation,
        load_name: str,
        f_low=40.0 * un.MHz,
        f_high=np.inf * un.MHz,
        f_range_keep: tuple[tp.FreqType, tp.Freqtype] | None = None,
        freq_bin_size=1,
        spectrum_warmup_time: un.Quantity[un.s] = 0.0 * un.s,
        rfi_threshold: float = 6.0,
        rfi_kernel_width_freq: int = 16,
        temperature_range: float | tuple[float, float] | None = 1.5,
        frequency_smoothing: str = "bin",
        time_coordinate_swpos: int = 0,
        temperature: float | None = None,
        temperature_median_filter: float | None = None,
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
        freqeuncy_smoothing
            How to average frequency bins together. Default is to merely bin them
            directly. Other options are 'gauss' to do Gaussian filtering (this is the
            same as Alan's C pipeline).
        ignore_times_percent
            The fraction of readings to ignore at the start of the observation. If
            greater than 100, will be interpreted as being a number of seconds to
            ignore.
        kwargs :
            All other arguments to :class:`LoadSpectrum`.

        Returns
        -------
        :class:`LoadSpectrum`.
        """
        spec = getattr(io_obs.spectra, load_name)
        res = getattr(io_obs.resistance, load_name)

        freq = FrequencyRange.from_edges(
            f_low=f_low,
            f_high=f_high,
            bin_size=freq_bin_size,
            alan_mode=frequency_smoothing == "gauss",
        )

        spec_times, timestamp0 = get_spec_timestamps(spec, time_coordinate_swpos)
        ignore_spec_nintegrations = _get_ignore_ninteg(
            spec_times, spec_timestamp0=timestamp0, ignore=spectrum_warmup_time
        )

        thermistor = ThermistorReadings.from_io(res)
        # Do the median filter first so that when doing the temp range the median is
        # more robust.
        if temperature_median_filter is not None:
            thermistor = thermistor.with_median_filter(temperature_median_filter)
        if temperature_range is not None:
            thermistor = thermistor.with_temp_range(temperature_range)

        thermistor = thermistor.in_time_range(
            spec_times[ignore_spec_nintegrations], spec_times[-1]
        )

        spectra = read_spectrum(spec_obj=spec, freq=freq)
        temp_mask = thermistor.mask_timestamps(spec_times)[np.newaxis]

        allflags = {}
        for key, sp in spectra.items():
            # Weird thing where there are zeros in the spectra.
            # For the Q-ratio, zero values are perfectly fine.
            if key.lower() != "q":
                sp[sp == 0] = np.nan

            if freq_bin_size > 1:
                if frequency_smoothing == "bin":
                    sp = tools.bin_array(sp.T, size=freq_bin_size).T
                elif frequency_smoothing == "gauss":
                    # We only really allow Gaussian smoothing so that we can
                    # match Alan's pipeline. In that case, the frequencies actually
                    # kept start from the 0th index, instead of taking the centre of
                    # each new bin. Thus we set decimate_at = 0.
                    sp = tools.gauss_smooth(sp.T, size=freq_bin_size, decimate_at=0).T
                else:
                    raise ValueError(
                        "frequency_smoothing must be one of ('bin', 'gauss')."
                    )

            flags = np.isnan(sp) | ~temp_mask
            mspec = np.where(flags, np.nan, sp)
            mean = np.nanmean(mspec, axis=1)
            var = np.nanvar(mspec, axis=1)

            nsample = np.sum(~np.isnan(mspec), axis=1)

            width = max(1, rfi_kernel_width_freq // freq_bin_size)

            varfilt = xrfi.flagged_filter(var, size=2 * width + 1)
            resid = mean - xrfi.flagged_filter(mean, size=2 * width + 1)
            flags = (
                flags.T
                | np.logical_or(
                    resid > rfi_threshold * np.sqrt(varfilt / nsample),
                    var - varfilt
                    > rfi_threshold * np.sqrt(2 * varfilt**2 / (nsample - 1)),
                )
            ).T
            allflags[key] = flags

        flags = np.any(list(allflags.values()), axis=0)

        out = cls(
            freq=freq,
            q=spectra["Q"],
            p_ant=spectra["p0"],
            p_load=spectra["p1"],
            p_ns=spectra["p2"],
            flags=flags,
            thermistor=thermistor,
            spec_anc=get_spectrum_ancillary(spec),
            spec_times=spec_times,
            temp_ave=temperature,
            metadata={
                "spectra_path": spec[0].path,
                "resistance_path": res.path,
                "freq_bin_size": freq_bin_size,
                "spectrum_warmup_time": spectrum_warmup_time,
                "rfi_threshold": rfi_threshold,
                "rfi_kernel_width_freq": rfi_kernel_width_freq,
                "temperature_range": temperature_range,
                "frequency_smoothing": frequency_smoothing,
                "temperature_median_filter": temperature_median_filter,
            },
            **kwargs,
        )

        if f_range_keep is not None:
            out = out.between_freqs(*f_range_keep)

        return out

    def between_freqs(self, f_low: tp.FreqType, f_high: tp.FreqType = np.inf * un.MHz):
        """Return a new LoadSpectrum that is masked between new frequencies."""
        mask = (self.freq.freq >= f_low) & (self.freq.freq <= f_high)
        return attr.evolve(
            self,
            freq=self.freq.with_new_mask(post_bin_f_low=f_low, post_bin_f_high=f_high),
            q=self.q[mask],
        )

    def _get_ave(self, item: str) -> np.ndarray:
        """Get the average of the given item."""
        weights = np.sum(~self.flags, axis=1)
        sm = np.nansum(getattr(self, item) * ~self.flags, axis=1)
        out = np.zeros_like(sm) * np.nan
        out[weights > 0] = sm[weights > 0] / weights[weights > 0]
        return out

    @cached_property
    def averaged_Q(self) -> np.ndarray:
        """Ratio of powers averaged over time.

        Notes
        -----
        The formula is

        .. math:: Q = (P_source - P_load)/(P_noise - P_load)
        """
        return self._get_ave("q")

    @cached_property
    def averaged_pant(self) -> np.ndarray:
        """Antenna temperature averaged over time."""
        return self._get_ave("p_ant")

    @cached_property
    def averaged_pload(self) -> np.ndarray:
        """Load temperature averaged over time."""
        return self._get_ave("p_load")

    @cached_property
    def averaged_pns(self) -> np.ndarray:
        """Source temperature averaged over time."""
        return self._get_ave("p_ns")

    @cached_property
    def variance_Q(self) -> np.ndarray:
        """Variance of Q across time (see averaged_Q)."""
        q = np.where(self.flags, np.nan, self.q)
        return np.nanvar(q, axis=1)

    @cached_property
    def n_integrations(self) -> np.ndarray:
        """Number of integrations averaged over."""
        return np.sum(~self.flags, axis=1)

    @cached_property
    def temp_ave(self):
        """Average temperature of the thermistor."""
        if self._temp_ave is None:
            return np.mean(self.thermistor.get_physical_temperature())
        else:
            return self._temp_ave

    def to_load_spectrum(self) -> LoadSpectrum:
        """Convert to a LoadSpectrum object."""
        return LoadSpectrum(
            freq=self.freq.freq,
            q=self.averaged_Q,
            metadata=self.metadata,
            variance=self.variance_Q,
            n_integrations=self.n_integrations,
            temp_ave=self.temp_ave,
        )
