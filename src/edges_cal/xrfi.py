import warnings

import numpy as np
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
    out = np.where(np.abs(den) > thresh, out, np.inf)
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
        An array containing the outlier significance metric. Same type and size as d.
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
        An array containing the outlier significance metric. Same type and size as d.
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
        2D flag array to be interpretted as mask for d.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over.
        Default is 8 pixels.
    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.
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
        1D flag array to be interpretted as mask for d.
    K : int, optional
        The box size  apply medfilt over. Default is 8 pixels.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.
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


def remove_rfi(spectrum):
    """Spectrum should have shape (NFREQS, NTIMES)"""
    if spectrum.dim == 2:
        significance = detrend_medfilt(spectrum.T, Kt=8, Kf=17)

        flags = np.abs(significance) > 5  # worse than 5 sigma!

        significance = detrend_meanfilt(spectrum.T, flags, Kt=8, Kf=17)
        spectrum.T[np.abs(significance) > 5] = np.nan
        return spectrum
    elif spectrum.dim == 1:
        significance = detrend_medfilt_1d(spectrum, K=8)

        flags = np.abs(significance) > 5  # worse than 5 sigma!

        significance = detrend_meanfilt_1d(spectrum, flags, K=8)
        spectrum[np.abs(significance) > 5] = np.nan
        return spectrum
