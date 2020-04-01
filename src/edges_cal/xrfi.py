import warnings

import numpy as np
import yaml
from astropy.convolution import convolve
from scipy.signal import medfilt, medfilt2d


def _check_convolve_dims(data, K1=None, K2=None):
    """Check the kernel sizes to be used in various convolution-like operations.
    If the kernel sizes are too big, replace them with the largest allowable size
    and issue a warning to the user.

    .. note:: ripped from here: https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        1- or 2-D array that will undergo convolution-like operations.
    K1 : int, optional
        Integer representing box dimension in first dimension to apply statistic.
        Defaults to None (see Returns)
    K2 : int, optional
        Integer representing box dimension in second dimension to apply statistic.
        Only used if data is two dimensional
    Returns
    -------
    K1 : int
        Input K1 or data.shape[0] if K1 is larger than first dim of arr.
        If K1 is not provided, will return data.shape[0].
    K2 : int (only if data is two dimensional)
        Input K2 or data.shape[1] if K2 is larger than second dim of arr.
        If data is 2D but K2 is not provided, will return data.shape[1].
    Raises
    ------
    ValueError:
        If the number of dimensions of the arr array is not 1 or 2, a ValueError is raised;
        If K1 < 1, or if data is 2D and K2 < 1.
    """
    if data.ndim not in (1, 2):
        raise ValueError("Input to filter must be 1- or 2-D array.")
    if K1 is None:
        warnings.warn(
            "No K1 input provided. Using the size of the data for the " "kernel size."
        )
        K1 = data.shape[0]
    elif K1 > data.shape[0]:
        warnings.warn(
            "K1 value {0:d} is larger than the data of dimension {1:d}; "
            "using the size of the data for the kernel size".format(K1, data.shape[0])
        )
        K1 = data.shape[0]
    elif K1 < 1:
        raise ValueError("K1 must be greater than or equal to 1.")
    if (data.ndim == 2) and (K2 is None):
        warnings.warn(
            "No K2 input provided. Using the size of the data for the " "kernel size."
        )
        K2 = data.shape[1]
    elif (data.ndim == 2) and (K2 > data.shape[1]):
        warnings.warn(
            "K2 value {0:d} is larger than the data of dimension {1:d}; "
            "using the size of the data for the kernel size".format(K2, data.shape[1])
        )
        K2 = data.shape[1]
    elif (data.ndim == 2) and (K2 < 1):
        raise ValueError("K2 must be greater than or equal to 1.")
    if data.ndim == 1:
        return K1
    else:
        return K1, K2


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
    out = np.true_divide(num, den, where=(np.abs(den) > thresh))
    out = np.where(
        np.logical_and(np.abs(den) > thresh, np.abs(num) > thresh), out, np.inf
    )

    # If numerator is also small, set to zero (better for smooth stuff)
    out = np.where(np.logical_and(np.abs(den) <= thresh, np.abs(num) <= thresh), 0, out)
    return out


def detrend_medfilt(data, Kt=8, Kf=8):
    """Detrend array using a median filter.

    .. note:: ripped from here: https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        2D data array to detrend.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over. Default
        is 8 pixels.
    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as spectrum.
    """

    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    data = np.concatenate([data[Kt - 1 :: -1], data, data[: -Kt - 1 : -1]], axis=0)
    data = np.concatenate(
        [data[:, Kf - 1 :: -1], data, data[:, : -Kf - 1 : -1]], axis=1
    )
    if np.iscomplexobj(data):
        d_sm_r = medfilt2d(data.real, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm_i = medfilt2d(data.imag, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm = d_sm_r + 1j * d_sm_i
    else:
        d_sm = medfilt2d(data, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
    d_rs = data - d_sm
    d_sq = np.abs(d_rs) ** 2
    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(medfilt2d(d_sq, kernel_size=(2 * Kt + 1, 2 * Kf + 1)) / 0.456)
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[Kt:-Kt, Kf:-Kf]


def detrend_medfilt_1d(data, K=8):
    """Detrend array using a median filter.

    .. note:: ripped from here: https://github.com/HERA-Team/hera_qm/blob/master/hera_qm/xrfi.py

    Parameters
    ----------
    data : array
        2D data array to detrend.
    K : int, optional
        The box size to apply medfilt over. Default is 8 pixels.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as spectrum.
    """

    K = _check_convolve_dims(data, K)
    data = np.concatenate([data[K - 1 :: -1], data, data[: -K - 1 : -1]])

    d_sm = medfilt(data, kernel_size=2 * K + 1)

    d_rs = data - d_sm
    d_sq = np.abs(d_rs) ** 2
    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(medfilt(d_sq, kernel_size=2 * K + 1) / 0.456)
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[K:-K]


def detrend_meanfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using a mean filter.
    Parameters
    ----------
    data : array
        2D data array to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for spectrum.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over.
        Default is 8 pixels.
    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as spectrum.
    """

    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    kernel = np.ones((2 * Kt + 1, 2 * Kf + 1))
    # do a mirror extend, like in scipy's convolve, which astropy doesn't support
    data = np.concatenate([data[Kt - 1 :: -1], data, data[: -Kt - 1 : -1]], axis=0)
    data = np.concatenate(
        [data[:, Kf - 1 :: -1], data, data[:, : -Kf - 1 : -1]], axis=1
    )
    if flags is not None:
        flags = np.concatenate(
            [flags[Kt - 1 :: -1], flags, flags[: -Kt - 1 : -1]], axis=0
        )
        flags = np.concatenate(
            [flags[:, Kf - 1 :: -1], flags, flags[:, : -Kf - 1 : -1]], axis=1
        )
    d_sm = convolve(data, kernel, mask=flags, boundary="extend")
    d_rs = data - d_sm
    d_sq = np.abs(d_rs) ** 2
    sig = np.sqrt(convolve(d_sq, kernel, mask=flags))
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[Kt:-Kt, Kf:-Kf]


def detrend_meanfilt_1d(data, flags=None, K=8):
    """Detrend array using a mean filter.

    Parameters
    ----------
    data : array
        1D data array to detrend.
    flags : array, optional
        1D flag array to be interpretted as mask for spectrum.
    K : int, optional
        The box size  apply medfilt over. Default is 8 pixels.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as spectrum.
    """

    K = _check_convolve_dims(data, K)
    kernel = np.ones(2 * K + 1)

    # do a mirror extend, like in scipy's convolve, which astropy doesn't support
    data = np.concatenate([data[K - 1 :: -1], data, data[: -K - 1 : -1]])

    if flags is not None:
        flags = np.concatenate([flags[K - 1 :: -1], flags, flags[: -K - 1 : -1]])

    d_sm = convolve(data, kernel, mask=flags, boundary="extend")
    d_rs = data - d_sm
    d_sq = np.abs(d_rs) ** 2
    sig = np.sqrt(convolve(d_sq, kernel, mask=flags))
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[K:-K]


def xrfi_medfilt(spectrum, threshold=6, Kt=16, Kf=16):
    """Generate RFI flags for a given spectrum using a median filter.

    Parameters
    ----------
    spectrum : array-like
        Either a 1D array of shape ``(NFREQS,)`` or a 2D array of shape
        ``(NFREQS, NTIMES)`` defining the measured raw spectrum.
        If 2D, a 2D filter in freq*time will be applied.
    threshold : float
        Number of effective sigma at which to clip RFI.
    Kt : int
        Window size in the time dimension (if used).
    Kf : int
        Window size in the frequency dimension.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    if spectrum.ndim == 2:
        significance = detrend_medfilt(spectrum.T, Kt=Kt, Kf=Kf)

        flags = np.abs(significance) > threshold  # worse than 5 sigma!

        significance = detrend_meanfilt(spectrum.T, flags, Kt=Kt, Kf=Kf)
        flags = np.abs(significance) > threshold
        return flags.T
    elif spectrum.ndim == 1:
        significance = detrend_medfilt_1d(spectrum, K=Kf)

        flags = np.abs(significance) > threshold  # worse than 5 sigma!

        significance = detrend_meanfilt_1d(spectrum, flags, K=Kf)
        flags = np.abs(significance) > threshold
        return flags


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
    weights,
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

    #    index = np.arange(len(f))
    nf = spectrum.shape[-1]
    f = np.linspace(-1, 1, window_width)
    flags = np.zeros(spectrum.shape, dtype=bool)

    flags[weights <= 0] = True

    class NoDataError(Exception):
        pass

    def compute_resid(d, w, flagged):
        mask = w > 0 & ~flagged
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
            r, mask = compute_resid(spectrum[window], weights[window], flags[window])
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

    # Set this window's flags to true.
    flags[:window_width][mask][np.abs(r) > n_sigma * r_std] = True

    # Initial window limits
    window += 1
    while window[-1] < nf:
        # Selecting section of data of width "window_width"
        try:
            r, fmask = compute_resid(spectrum[window], weights[window], flags[window])
        except NoDataError:
            continue

        flags[window][fmask][np.abs(r) > n_sigma * r_std] = True

        # Update std dev. estimate for the next window.
        r_std = _get_mad(r) if use_median else np.std(r)
        window += 1

    if flip:
        flip_flags = xrfi_poly_filter(
            np.flip(spectrum),
            np.flip(weights),
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
    weights,
    f_ratio=None,
    f_log=True,
    t_log=True,
    n_signal=10,
    n_resid=3,
    n_abs_resid_threshold=5,
    max_iter=20,
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
    weights : array-like
        The weights associated with the data (same shape as `spectrum`).
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
    n_abs_resid_threshold : float, optional
        The factor by which the absolute residual model is multiplied to determine
        outliers.
    max_iter : int, optional
        The maximum number of iterations to perform.

    Returns
    -------
    flags : array-like
        Boolean array of the same shape as ``spectrum`` indicated which channels/times
        have flagged RFI.
    """
    if f_log and not f_ratio:
        raise ValueError("If fitting in log(freq), you must provide f_ratio.")

    assert n_abs_resid_threshold > 1.5
    assert n_resid < n_signal

    nf = spectrum.shape[-1]
    flags = np.zeros(nf, dtype=bool)
    f = np.linspace(-1, 1, nf) if not f_log else np.logspace(0, f_ratio, nf)

    data_mask = (
        (spectrum > 0) & (weights > 0) & ~np.isnan(spectrum) & ~np.isinf(spectrum)
    )
    flags |= ~data_mask

    n_flags = np.sum(flags)
    n_flags_new = n_flags + 1
    counter = 0
    while (
        n_flags < n_flags_new and counter < max_iter and nf - n_flags_new > n_signal * 2
    ):
        n_flags = 1 * n_flags_new

        ff = f[~flags]
        s = spectrum[~flags]

        if t_log:
            s = np.log(s)

        par = np.polyfit(ff, s, n_signal - 1)
        model = np.polyval(par, f)

        if t_log:
            model = np.exp(model)

        res = s - model[~flags]

        par = np.polyfit(ff, np.abs(res), n_resid - 1)
        model_std = np.polyval(par, ff)

        flags[~flags] |= np.abs(res) > n_abs_resid_threshold * model_std

        n_flags_new = np.sum(flags)
        counter += 1

    if counter == max_iter:
        warnings.warn(
            f"max iterations ({max_iter}) reached, not all RFI might have been caught."
        )

    if nf - n_flags_new >= n_signal * 2:
        warnings.warn(
            "Termination of iterative loop due to too many flags. Reduce n_signal or check data."
        )

    return flags
