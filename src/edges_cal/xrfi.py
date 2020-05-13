import warnings
from typing import Tuple, Union

import numpy as np
import yaml
from astropy.convolution import convolve
from scipy import ndimage
from scipy.signal import medfilt, medfilt2d

from .modelling import ModelFit


def _check_convolve_dims(data, half_size: [None, Tuple[int, None]] = None):
    """Check the kernel sizes to be used in various convolution-like operations.
    If the kernel sizes are too big, replace them with the largest allowable size
    and issue a warning to the user.

    .. note:: ripped from here: https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        1- or 2-D array that will undergo convolution-like operations.
    half_size : tuple
        Tuple of ints or None's with length ``data.ndim``. They represent the half-size
        of the kernel to be used (or, rather the kernel will be 2*half_size+1 in each
        dimension). None uses half_size=data.shape.

    Returns
    -------
    size : tuple
        The kernel size in each dimension.

    Raises
    ------
    ValueError:
        If half_size does not match the number of dimensions.
    """
    if half_size is None:
        half_size = (None,) * data.ndim

    if len(half_size) != data.ndim:
        raise ValueError(
            "Number of kernel dimensions does not match number of data dimensions."
        )

    out = []
    for data_shape, hsize in zip(data.shape, half_size):
        if hsize is None or hsize > data_shape:
            out.append(data_shape)
        elif hsize < 0:
            out.append(0)
        else:
            out.append(hsize)

    return tuple(out)


def robust_divide(num, den):
    """Prevent division by zero.
    This function will compute division between two array-like objects by setting
    values to infinity when the denominator is small for the given data type. This
    avoids floating point exception warnings that may hide genuine problems
    in the data.
    Parameters
    ----------
    num : array
        The numerator.
    den : array
        The denominator.
    Returns
    -------
    out : array
        The result of dividing num / den. Elements where b is small (or zero) are set
        to infinity.
    """
    thresh = np.finfo(den.dtype).eps

    den_mask = np.abs(den) > thresh

    out = np.true_divide(num, den, where=den_mask)
    out[~den_mask] = np.inf

    # If numerator is also small, set to zero (better for smooth stuff)
    out[~den_mask & (np.abs(num) <= thresh)] = 0
    return out


def flagged_filter(
    data: np.ndarray,
    size: [int, Tuple[int]],
    kind: str = "median",
    flags: [None, np.ndarray] = None,
    mode: [None, str] = None,
    interp_flagged=True,
    **kwargs,
):
    """
    Perform an n-dimensional filter operation on optionally flagged data.

    Parameters
    ----------
    data : np.ndarray
        The data to filter. Can be of arbitrary dimension.
    size : int or tuple
        The size of the filtering convolution kernel. If tuple, one entry per dimension
        in `data`.
    kind : str, optional
        The function to apply in each window. Typical options are `mean` and `median`.
        For this function to work, the function kind chosen here must have a correspdoning
        `nan<function>` implementation in numpy.
    flags : np.ndarray, optional
        A boolean array specifying data to omit from the filtering.
    mode : str, optional
        The mode of the filter. See ``scipy.ndimage.generic_filter` for details. By default,
        'nearest' if size < data.size otherwise 'reflect'.
    interp_flagged : bool, optional
        Whether to fill in flagged entries with its filtered value. Otherwise,
        flagged entries are set to their original value.
    kwargs :
        Other options to pass to the generic filter function.

    Returns
    -------
    np.ndarray :
        The filtered array, of the same shape and type as ``data``.

    Notes
    -----
    This function can typically be used to implement a flagged median filter. It does
    have some limitations in this regard, which we will now describe.

    It would be expected that a perfectly smooth
    monotonic function, after median filtering, should remain identical to the input.
    This is only the case for the default 'nearest' mode. For the alternative 'reflect'
    mode, the edge-data will be corrupted from the input. On the other hand, it may be
    expected that if the kernel width is equal to or larger than the data size, that
    the operation is merely to perform a full collapse over that dimension. This is the
    case only for mode 'reflect', while again mode 'nearest' will continue to yield (a
    very slow) identity operation. By default, the mode will be set to 'reflect' if
    the size is >= the data size, with an emitted warning.

    Furthermore, a median filter is *not* an identity operation, even on monotonic
    functions, for an even-sized kernel (in this case it's the average of the two
    central values).

    Also, even for an odd-sized kernel, if using flags, some of the windows will contain
    an odd number of useable data, in which case the data surrounding the flag will not
    be identical to the input.

    Finally, flags near the edges can have strange behaviour, depending on the mode.
    """
    if mode is None:
        if (isinstance(size, int) and size >= min(data.shape)) or (
            isinstance(size, tuple) and any(s > d for s, d in zip(size, data.shape))
        ):
            warnings.warn(
                "Setting default mode to reflect because a large size was set."
            )
            mode = "reflect"
        else:
            mode = "nearest"

    if flags is not None and np.any(flags):
        fnc = getattr(np, "nan" + kind)
        assert flags.shape == data.shape
        orig_flagged_data = data[flags].copy()
        data[flags] = np.nan
        filtered = ndimage.generic_filter(data, fnc, size=size, mode=mode, **kwargs)
        if not interp_flagged:
            filtered[flags] = orig_flagged_data
        data[flags] = orig_flagged_data

    else:
        if kind == "mean":
            kind = "uniform"
        filtered = getattr(ndimage, kind + "_filter")(
            data, size=size, mode=mode, **kwargs
        )

    return filtered


def detrend_medfilt(
    data: np.ndarray,
    flags: [None, np.ndarray] = None,
    half_size: [None, Tuple[int, None]] = None,
):
    """Detrend array using a median filter.

    .. note:: ripped from here: https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        Data to detrend. Can be an array of any number of dimensions.
    flags : boolean array, optional
        Flags specifying data to ignore in the detrend. If not given, don't ignore
        anything.
    half_size : tuple of int/None
        The half-size of the kernel to convolve (kernel size will be 2*half_size+1).
        Value of zero (for any dimension) omits that axis from the kernel, effectively
        applying the detrending for each subarray along that axis. Value of None will
        effectively (but slowly) perform a median along the entire axis before running
        the kernel over the other axis.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as `data`.

    Notes
    -----
    This detrending is very good for data with large RFI compared to the noise, but also
    reasonably large noise compared to the spectrum steepness. If the noise is small
    compared to the steepness of the spectrum, individual windows can become *almost always*
    monotonic, in which case the randomly non-monotonic bins "stick out" and get wrongly
    flagged. This can be helped three ways:
    1) Use a smaller bin width. This helps by reducing the probability that a bin will
       be randomly non-monotonic. However it also loses signal-to-noise on the RFI.
    2) Pre-fit a smooth model that "flattens" the spectrum. This helps by reducing the
       probability that bins will be monotonic (higher noise level wrt steepness). It
       has the disadvantage that fitted models can be wrong when there's RFI there.
    3) Follow the medfilt with a meanfilt: if the medfilt is able to flag most/all of
       the RFI, then a following meanfilt will tend to "unfilter" the wrongly flagged
       parts.

    """
    half_size = _check_convolve_dims(data, half_size)
    size = tuple(2 * s + 1 for s in half_size)

    d_sm = flagged_filter(data, size=size, kind="median", flags=flags)
    d_rs = data - d_sm
    d_sq = d_rs ** 2

    # Remember that d_sq will be zero for any window in which the data is monotonic (but
    # could also be zero for non-monotonic windows where the two halves of the window
    # are self-contained). Most smooth functions will be monotonic in small enough
    # windows. If noise is of low-enough amplitude wrt the steepness of the smooth
    # underlying function, there is a good chance the resulting data will also be
    # monotonic. Nevertheless, any RFI that is large enough will cause the value of
    # that channel to *not* be the central value, and it will have d_sq > 0.

    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(flagged_filter(d_sq, size=size, kind="median", flags=flags) / 0.456)

    # don't divide by zero, instead turn those entries into +inf
    return robust_divide(d_rs, sig)


def detrend_meanfilt(
    data: np.ndarray,
    flags: [None, np.ndarray] = None,
    half_size: [None, Tuple[int, None]] = None,
):
    """Detrend array using a mean filter.

    Parameters
    ----------
    data : array
        Data to detrend. Can be an array of any number of dimensions.
    flags : boolean array, optional
        Flags specifying data to ignore in the detrend. If not given, don't ignore
        anything.
    half_size : tuple of int/None
        The half-size of the kernel to convolve (kernel size will be 2*half_size+1).
        Value of zero (for any dimension) omits that axis from the kernel, effectively
        applying the detrending for each subarray along that axis. Value of None will
        effectively (but slowly) perform a median along the entire axis before running
        the kernel over the other axis.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as `data`.

    Notes
    -----
    This detrending is very good for data that has most of the RFI flagged already, but
    will perform very poorly when un-flagged RFI still exists. It is often useful to
    precede this with a median filter.
    """

    half_size = _check_convolve_dims(data, half_size)
    size = tuple(2 * s + 1 for s in half_size)

    d_sm = flagged_filter(data, size=size, kind="mean", flags=flags)
    d_rs = data - d_sm
    d_sq = d_rs ** 2

    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(flagged_filter(d_sq, size=size, kind="mean", flags=flags))

    # don't divide by zero, instead turn those entries into +inf
    return robust_divide(d_rs, sig)


def xrfi_medfilt(
    spectrum: np.ndarray,
    threshold: float = 6,
    flags: [None, np.ndarray] = None,
    kf: [int, None] = 8,
    kt: [int, None] = 8,
    inplace: bool = True,
    max_iter: int = 1,
    poly_order=0,
    accumulate=False,
    use_meanfilt=True,
):
    """Generate RFI flags for a given spectrum using a median filter.

    Parameters
    ----------
    spectrum : array-like
        Either a 1D array of shape ``(NFREQS,)`` or a 2D array of shape
        ``(NTIMES, NFREQS)`` defining the measured raw spectrum.
        If 2D, a 2D filter in freq*time will be applied by default. One can perform
        the filter just over frequency (in the case that `NTIMES > 1`) by setting
        `kt=0`.
    threshold : float, optional
        Number of effective sigma at which to clip RFI.
    flags : array-like, optional
        Boolean array of pre-existing flagged data to ignore in the filtering.
    kt, kf : tuple of int/None
        The half-size of the kernel to convolve (eg. kernel size over frequency
        will be ``2*kt+1``).
        Value of zero (for any dimension) omits that axis from the kernel, effectively
        applying the detrending for each subarray along that axis. Value of None will
        effectively (but slowly) perform a median along the entire axis before running
        the kernel over the other axis.
    inplace : bool, optional
        If True, and flags are given, update the flags in-place instead of creating a
        new array.
    max_iter : int, optional
        Maximum number of iterations to perform. Each iteration uses the flags of the
        previous iteration to achieve a more robust estimate of the flags. Multiple
        iterations are more useful if ``poly_order > 0``.
    poly_order : int, optional
        If greater than 0, fits a polynomial to the spectrum before performing
        the median filter. Only allowed if spectrum is 1D. This is useful for getting
        the number of false positives down. If max_iter>1, the polynomial will be refit
        on each iteration (using new flags).
    accumulate : bool,optional
        If True, on each iteration, accumulate flags. Otherwise, use only flags from the
        previous iteration and then forget about them. Recommended to be False.
    use_meanfilt : bool, optional
        Whether to apply a mean filter *after* the median filter. The median filter is
        good at getting RFI, but can also pick up non-RFI if the spectrum is steep
        compared to the noise. The mean filter is better at only getting RFI if the RFI
        has already been flagged.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.

    Notes
    -----
    The default combination of using a median filter followed by a mean filter works
    quite well. The median filter works quite well at picking up large RFI (wrt to the
    noise level), but can also create false positives if the noise level is small wrt
    the steepness of the slope. Following by a flagged mean filter tends to remove these
    false positives (as it doesn't get pinned to zero when the function is monotonic).

    It is unclear whether performing an iterative filtering is very useful unless using
    a polynomial subtraction. With polynomial subtraction, one should likely use at least
    a few iterations, without accumulation, so that the polynomial is not skewed by the
    as-yet-unflagged RFI.

    Choice of kernel size can be important. The wider the kernel, the more "signal-to-noise"
    one will get on the RFI. Also, if there is a bunch of RFI all clumped together, it will
    definitely be missed by a kernel window of order double the size of the clump or less.
    By increasing the kernel size, these clumps are picked up, but edge-effects become
    more prevalent in this case. One option here would be to iterate over kernel sizes
    (getting smaller), such that very large blobs are first flagged out, then progressively
    finer detail is added. Use ``xrfi_iterative_medfilt`` for that.
    """
    iter = 0

    if flags is None:
        new_flags = np.zeros(spectrum.shape, dtype=bool)
    else:
        new_flags = flags if inplace else flags.copy()

    nflags = -1

    nflags_list = []
    resid_list = []
    assert max_iter > 0
    resid = spectrum.copy()

    size = (kf,) if spectrum.ndim == 1 else (kt, kf)
    while iter < max_iter and np.sum(new_flags) > nflags:
        nflags = np.sum(new_flags)

        if spectrum.ndim == 1 and poly_order:
            # Subtract a smooth polynomial first.
            # The point of this is that steep spectra with only a little bit of noise
            # tend to detrend to exactly zero, but randomly may detrend to something non-zero.
            # In this case, the behaviour is to set the significance to infinity. This is not
            # a problem for data in which the noise is large compared to the signal. We can
            # force this by initially detrending by some flexible polynomial over the whole
            # band. This is not guaranteed to work -- the poly fit itself could over-fit
            # for RFI. Therefore the order of the fit should be low. Its purpose is not to
            # do a "good fit" to the data, but rather to get the residuals "flat enough" that
            # the median filter works.
            # TODO: the following is pretty limited (why polynomial?) but it seems to do
            # reasonably well.
            f = np.linspace(0, 1, len(spectrum))
            resid[~new_flags] = (
                spectrum[~new_flags]
                - ModelFit(
                    "polynomial",
                    f[~new_flags],
                    spectrum[~new_flags],
                    n_terms=poly_order,
                ).evaluate()
            )
            resid_list.append(resid)
        else:
            resid = spectrum

        med_significance = detrend_medfilt(resid, half_size=size, flags=new_flags)

        if use_meanfilt:
            medfilt_flags = np.abs(med_significance) > threshold
            significance = detrend_meanfilt(resid, half_size=size, flags=medfilt_flags)
        else:
            significance = med_significance

        if accumulate:
            new_flags |= np.abs(significance) > threshold
        else:
            new_flags = np.abs(significance) > threshold

        iter += 1
        nflags_list.append(np.sum(new_flags))

    if 1 < max_iter == iter and np.sum(new_flags) > nflags:
        warnings.warn("Median filter reached max_iter and is still finding new RFI.")

    return (
        new_flags,
        {
            "significance": significance,
            "median_significance": med_significance,
            "iters": iter,
            "nflags": nflags_list,
            "residuals": resid_list,
        },
    )


def xrfi_iterative_medfilt(
    spectrum: np.ndarray,
    threshold: float = 6,
    flags: [None, np.ndarray] = None,
    min_kf: [int, None] = 8,
    min_kt: [int, None] = 8,
    max_kf: [int, None] = None,
    max_kt: [int, None] = None,
    inplace: bool = True,
    accumulate=False,
    use_meanfilt=True,
):
    """
    An iterative median filter, in which the window size is progressively reduced.

    Parameters
    ----------
    spectrum
    threshold
    flags
    kf
    kt
    inplace
    max_iter
    poly_order
    accumulate
    use_meanfilt

    Returns
    -------

    """
    raise NotImplementedError("This has not been implemented yet.")


def xrfi_explicit(f, rfi_file=None, extra_rfi=None):
    """
    Excise RFI from given data using a explicitly set list of flag ranges.

    Parameters
    ----------
    f : array-like
        Frequencies, in MHz, of the data.
    rfi_file : str, optional
        A YAML file containing the key 'rfi_ranges', which should be a list of 2-tuples
        giving the (min, max) frequency range of known RFI channels (in MHz). By default,
        uses a file included in `edges-analysis` with known RFI channels from the MRO.
    extra_rfi : list, optional
        A list of extra RFI channels (in the format of the `rfi_ranges` from the `rfi_file`).

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """

    rfi_freqs = []
    if rfi_file:
        with open(rfi_file, "r") as fl:
            rfi_freqs += yaml.load(fl, Loader=yaml.FullLoader)["rfi_ranges"]

    if extra_rfi:
        rfi_freqs += extra_rfi

    flags = np.zeros(len(f), dtype=bool)
    for low, high in rfi_freqs:
        flags[(f > low) & (f < high)] = True

    return flags


def _get_mad(x):
    med = np.median(x)
    # Factor of 0.456 to scale median back to Gaussian std dev.
    return np.median(np.abs(x - med)) / np.sqrt(0.456)


def xrfi_poly_filter(
    spectrum,
    weights=None,
    window_width=100,
    n_poly=4,
    n_bootstrap=20,
    n_sigma=2.5,
    use_median=False,
    flip=False,
):
    """
    Flag RFI by using a moving window and a low-order polynomial to detrend.

    This is similar to :func:`xrfi_medfilt`, except that within each sliding window,
    a low-order polynomial is fit, and the std dev of the residuals is used as the
    underlying distribution width at which to clip RFI.

    Parameters
    ----------
    spectrum : array-like
        A 1D or 2D array, where the last axis corresponds to frequency. The data
        measured at those frequencies.
    weights : array-like
        The weights associated with the data (same shape as `spectrum`).
    window_width : int, optional
        The width of the moving window in number of channels.
    n_poly : int, optional
        Number of polynomial terms to fit in each sliding window. Should be significantly
        smaller than ``window_width``.
    n_bootstrap : int, optional
        Number of bootstrap samples to take to estimate the standard deviation of
        the data without RFI.
    n_sigma : float, optional
        The number of sigma at which to threshold RFI.
    use_median : bool, optional
        Instead of using bootstrap for the initial window, use Median Absolute Deviation.
    flip : bool, optional
        Whether to *also* do the analysis backwards, doing a logical OR on the final
        flags.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    nf = spectrum.shape[-1]
    f = np.linspace(-1, 1, window_width)
    flags = np.zeros(spectrum.shape, dtype=bool)

    if weights is not None:
        flags |= weights <= 0

    class NoDataError(Exception):
        pass

    def compute_resid(d, flagged):
        mask = ~flagged
        if np.any(mask):
            par = np.polyfit(f[mask], d[mask], n_poly - 1)
            return d[mask] - np.polyval(par, f[mask]), mask
        else:
            raise NoDataError

    # Compute residuals for initial section
    got_init = False
    window = np.arange(window_width)
    while not got_init and window[-1] < nf:
        try:
            r, mask = compute_resid(spectrum[window], flags[window])
            got_init = True

        except NoDataError:
            window += 1

    if not got_init:
        raise NoDataError(
            "There were no windows of data with enough data to perform xrfi."
        )

    # Computation of STD for initial section using the median statistic
    if not use_median:
        r_choice_std = [
            np.std(np.random.choice(r, len(r) // 2)) for _ in range(n_bootstrap)
        ]
        r_std = np.median(r_choice_std)
    else:
        r_std = _get_mad(r)

    print(r_std)

    # Set this window's flags to true.
    flags[:window_width][mask] |= np.abs(r) > n_sigma * r_std

    # Initial window limits
    window += 1
    while window[-1] < nf:
        # Selecting section of data of width "window_width"
        try:
            r, fmask = compute_resid(spectrum[window], flags[window])
        except NoDataError:
            continue

        flags[window][fmask][np.abs(r) > n_sigma * r_std] = True

        # Update std dev. estimate for the next window.
        r_std = _get_mad(r) if use_median else np.std(r)
        window += 1

    if flip:
        flip_flags = xrfi_poly_filter(
            np.flip(spectrum),
            np.flip(weights) if weights is not None else None,
            window_width=window_width,
            n_poly=n_poly,
            n_bootstrap=n_bootstrap,
            n_sigma=n_sigma,
            use_median=use_median,
            flip=False,
        )
        flags |= np.flip(flip_flags)

    return flags


def xrfi_poly(
    spectrum,
    flags=None,
    f_ratio=None,
    f_log=False,
    t_log=True,
    n_signal=3,
    n_resid=-1,
    threshold=10,
    max_iter=20,
    accumulate=False,
    increase_order=True,
    decrement_threshold=0,
    min_threshold=5,
    inplace=True,
    watershed: [None, int, Tuple[int, float], np.ndarray] = None,
    return_models=False,
):
    """
    Flag RFI by subtracting a smooth polynomial and iteratively removing outliers.

    On each iteration, a polynomial is fit to the unflagged data, and a lower-order
    polynomial is fit to the absolute residuals of the data with the model polynomial.
    Bins with absolute residuals greater than `n_abs_resid_threshold` are flagged,
    and the process is repeated until no new flags are found.

    Parameters
    ----------
    spectrum : array-like
        A 1D or 2D array, where the last axis corresponds to frequency. The data
        measured at those frequencies.
    flags : array-like, optional
        The flags associated with the data (same shape as `spectrum`).
    f_ratio : float, optional
        The ratio of the max to min frequency to be fit. Only required if ``f_log``
        is True.
    f_log : bool, optional
        Whether to fit the signal with log-spaced frequency values.
    t_log : bool, optional
        Whether to fit the signal with log temperature.
    n_signal : int, optional
        The number of polynomial terms to use to fit the signal.
    n_resid : int, optional
        The number of polynomial terms to use to fit the residuals.
    threshold : float, optional
        The factor by which the absolute residual model is multiplied to determine
        outliers.
    max_iter : int, optional
        The maximum number of iterations to perform.
    accumulate : bool, optional
        Whether to accumulate flags on each iteration.
    increase_order : bool, optional
        Whether to increase the order of the polynomial on each iteration.
    decrement_threshold : float, optional
        An amount to decrement the threshold by every iteration. Threshold will never
        go below ``min_threshold``.
    min_threshold : float, optional
        The minimum threshold to decrement to.
    return_models : bool, optional
        Whether to return the full models at each iteration.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    if min_threshold > threshold:
        warnings.warn(
            f"You've set a threshold smaller than the min_threshold of {min_threshold}. Will use threshold={min_threshold}."
        )
        threshold = min_threshold

    if f_log and not f_ratio:
        raise ValueError("If fitting in log(freq), you must provide f_ratio.")

    assert threshold > 3

    nf = spectrum.shape[-1]
    f = np.linspace(-1, 1, nf) if not f_log else np.logspace(0, f_ratio, nf)

    orig_flags = flags if flags is not None else np.zeros(nf, dtype=bool)
    orig_flags |= (spectrum <= 0) | np.isnan(spectrum) | np.isinf(spectrum)

    flags = orig_flags.copy()

    if not increase_order:
        assert n_resid < n_signal

    # Set the watershed as a small array that will overlay a flag.
    if watershed is not None and len(watershed) == 2:
        watershed = np.ones(watershed[0] * 2 + 1) * watershed[1]
    if isinstance(watershed, int):
        watershed = np.ones(watershed * 2 + 1)

    n_flags_changed = 1
    counter = 0

    n_flags_changed_list = []
    total_flags_list = []
    model_list = []
    model_std_list = []
    while n_flags_changed > 0 and counter < max_iter and np.sum(~flags) > n_signal * 2:
        ff = f[~flags]
        s = spectrum[~flags]

        if t_log:
            s = np.log(s)

        par = np.polyfit(ff, s, n_signal)
        model = np.polyval(par, f)
        if return_models:
            model_list.append(par)
        if t_log:
            model = np.exp(model)

        res = spectrum - model

        par = np.polyfit(
            ff, np.abs(res[~flags]), n_resid if n_resid > 0 else n_signal + n_resid
        )
        model_std = np.polyval(par, f)
        if return_models:
            model_std_list.append(par)

        if accumulate:
            nflags = np.sum(flags[~flags])
            flags[~flags] |= np.abs(res)[~flags] > threshold * model_std[~flags]
            n_flags_changed = np.sum(flags[~flags]) - nflags
        else:
            new_flags = orig_flags | (np.abs(res) > threshold * model_std)

            # Apply a watershed -- assume surrounding channels will succumb to RFI.
            if watershed is not None:
                watershed_flags = np.zeros_like(new_flags)
                for i, flag in enumerate(new_flags):
                    if flag:
                        rng = range(
                            max(0, i - len(watershed) // 2),
                            min(len(new_flags), i + len(watershed) // 2),
                        )
                        wrng_min = max(0, -(i - len(watershed) // 2))
                        wrng = range(wrng_min, wrng_min + len(rng))

                        watershed_flags[rng] |= (
                            np.abs(res[rng])
                            > watershed[wrng] * threshold * model_std[rng]
                        )
                new_flags |= watershed_flags

            n_flags_changed = np.sum(flags ^ new_flags)
            flags = new_flags.copy()

        counter += 1
        if increase_order:
            n_signal += 1

        threshold = max(threshold - decrement_threshold, min_threshold)

        n_flags_changed_list.append(n_flags_changed)
        total_flags_list.append(np.sum(flags))

    if counter == max_iter:
        warnings.warn(
            f"max iterations ({max_iter}) reached, not all RFI might have been caught."
        )

    if np.sum(~flags) <= n_signal * 2:
        warnings.warn(
            "Termination of iterative loop due to too many flags. Reduce n_signal or check data."
        )

    if inplace:
        orig_flags |= flags

    return (
        flags,
        {
            "n_flags_changed": n_flags_changed_list,
            "total_flags": total_flags_list,
            "models": model_list,
            "model_std": model_std_list,
        },
    )
