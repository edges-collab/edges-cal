# -*- coding: utf-8 -*-
"""Functions for calibrating the receiver."""
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
from typing import Dict, Iterable, List, Optional


def temperature_thermistor(
    resistance: [float, np.ndarray],
    coeffs: [str, Iterable] = "oven_industries_TR136_170",
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

    if type(coeffs) is str:
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
    else:
        return temp - 273.15


def noise_wave_param_fit(
    f_norm: np.ndarray,
    gamma_rec: np.ndarray,
    gamma_open: np.ndarray,
    gamma_short: np.ndarray,
    temp_raw_open: np.ndarray,
    temp_raw_short: np.ndarray,
    temp_thermistor_open: np.ndarray,
    temp_thermistor_short: np.ndarray,
    wterms: int,
):
    """
    Fit noise-wave polynomial parameters.

    Parameters
    ----------
    f_norm : array_like
        Normalized frequencies (arbitrarily normalised, but standard assumption is
        that the centre is zero, and the scale is such that the range is (-1, 1))
    gamma_rec : array-like
        Reflection coefficient, as function of frequency, of the receiver.
    gamma_open : array-like
        Reflection coefficient, as function of frequency, of the open load.
    gamma_short : array-like
        Reflection coefficient, as function of frequency, of the shorted load.
    temp_raw_open : array-like
        Raw measured spectrum temperature of open load.
    temp_raw_short : array-like
        Raw measured spectrum temperature of shorted load.
    temp_thermistor_open : array-like
        Measured (known) temperature of open load.
    temp_thermistor_short : array-like
        Measured (known) temperature of shorted load.
    wterms : int
        The number of polynomial terms to use for each of the noise-wave functions.

    Returns
    -------
    Tunc, Tcos, Tsin : array_like
        The solutions to each of T_unc, T_cos and T_sin as functions of frequency.
    """
    # S11 quantities
    Fo = get_F(gamma_rec, gamma_open)
    Fs = get_F(gamma_rec, gamma_short)
    alpha_open = get_alpha(gamma_rec, gamma_open)
    alpha_short = get_alpha(gamma_rec, gamma_short)

    G = 1 - np.abs(gamma_rec) ** 2
    K1o = (1 - np.abs(gamma_open) ** 2) * (np.abs(Fo) ** 2) / G
    K1s = (1 - np.abs(gamma_short) ** 2) * (np.abs(Fs) ** 2) / G

    K2o = (np.abs(gamma_open) ** 2) * (np.abs(Fo) ** 2) / G
    K2s = (np.abs(gamma_short) ** 2) * (np.abs(Fs) ** 2) / G

    K3o = (np.abs(gamma_open) * np.abs(Fo) / G) * np.cos(alpha_open)
    K3s = (np.abs(gamma_short) * np.abs(Fs) / G) * np.cos(alpha_short)
    K4o = (np.abs(gamma_open) * np.abs(Fo) / G) * np.sin(alpha_open)
    K4s = (np.abs(gamma_short) * np.abs(Fs) / G) * np.sin(alpha_short)

    # Matrices A and b
    A = np.zeros((3 * wterms, 2 * len(f_norm)))
    for i in range(wterms):
        A[i, :] = np.append(K2o * f_norm ** i, K2s * f_norm ** i)
        A[i + 1 * wterms, :] = np.append(K3o * f_norm ** i, K3s * f_norm ** i)
        A[i + 2 * wterms, :] = np.append(K4o * f_norm ** i, K4s * f_norm ** i)
    b = np.append(
        (temp_raw_open - temp_thermistor_open * K1o),
        (temp_raw_short - temp_thermistor_short * K1s),
    )

    # Transposing matrices so 'frequency' dimension is along columns
    M = A.T
    ydata = np.reshape(b, (-1, 1))

    # Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
    Q1, R1 = sp.linalg.qr(M, mode="economic")
    param = sp.linalg.solve(R1, np.dot(Q1.T, ydata)).flatten()

    TU = np.poly1d(param[:wterms][::-1])
    TC = np.poly1d(param[wterms : 2 * wterms][::-1])
    TS = np.poly1d(param[2 * wterms : 3 * wterms][::-1])

    return TU, TC, TS


def get_F(gamma_rec: np.ndarray, gamma_ant: np.ndarray) -> np.ndarray:  # noqa: N802
    """Get the F parameter for a given receiver and antenna.

    Parameters
    ----------
    gamma_rec : np.ndarray
        The reflection coefficient (S11) of the receiver.
    gamma_ant : np.ndarray
        The reflection coefficient (S11) of the antenna

    Returns
    -------
    F : np.ndarray
        The F parameter (see M17)
    """
    return np.sqrt(1 - np.abs(gamma_rec) ** 2) / (1 - gamma_ant * gamma_rec)


def get_alpha(gamma_rec: np.ndarray, gamma_ant: np.ndarray) -> np.ndarray:
    """Get the alpha parameter for a given receiver and antenna.

    Parameters
    ----------
    gamma_rec : np.ndarray
        The reflection coefficient of the receiver.
    gamma_ant : np.ndarray
        The reflection coefficient fo the antenna.
    """
    return np.angle(gamma_ant * get_F(gamma_rec, gamma_ant))


def power_ratio(
    temp_ant,
    gamma_ant,
    gamma_rec,
    scale,
    offset,
    temp_unc,
    temp_cos,
    temp_sin,
    temp_noise_source,
    temp_load,
    return_terms=False,
):
    """
    Compute the ratio of raw powers from the three-position switch.

    Parameters
    ----------
    temp_ant : array_like, shape (NFREQS,)
        Temperature of the antenna, or simulator.
    gamma_ant : array_like, shape (NFREQS,)
        S11 of the antenna (or simulator)
    gamma_rec : array_like, shape (NFREQS,)
        S11 of the receiver.
    scale : :class:`np.poly1d`
        A polynomial representing the C_1 term.
    offset : :class:`np.poly1d`
        A polynomial representing the C_2 term
    temp_unc : :class:`np.poly1d`
        A polynomial representing the uncorrelated noise-wave parameter
    temp_cos : :class:`np.poly1d`
        A polynomial representing the cosine noise-wave parameter
    temp_sin : :class:`np.poly1d`
        A polynomial representing the sine noise-wave parameter
    temp_noise_source : array_like, shape (NFREQS,)
        Temperature of the internal noise source.
    temp_load : array_like, shape (NFREQS,)
        Temperature of the internal load
    return_terms : bool, optional
        If True, return the terms of Qp, rather than the sum of them._

    Returns
    -------
    array_like : the quantity Q_P as a function of frequency.

    Notes
    -----
    Computes (as a model)

    .. math :: Q_P = (P_ant - P_L)/(P_NS - P_L)

    """
    K = get_K(gamma_rec, gamma_ant)

    terms = [t * k for t, k in zip([temp_ant, temp_unc, temp_cos, temp_sin], K)] + [
        (offset - temp_load),
        scale * temp_noise_source,
    ]

    if return_terms:
        return terms
    else:
        return sum(terms[:5]) / terms[5]


def get_K(gamma_rec, gamma_ant, f_ratio=None, alpha=None, gain=None):  # noqa: N802
    """
    Determine the S11-dependent factors for each term in Eq. 7 (Monsalve 2017).

    Parameters
    ----------
    gamma_rec : array_like
        Receiver S11
    gamma_ant : array_like
        Antenna (or load) S11.
    f_ratio : array_like, optional
        The F factor (Eq. 3 of Monsalve 2017). Computed if not given.
    alpha : array_like, optional
        The alpha factor (Eq. 4 of Monsalve, 2017). Computed if not given.
    gain : array_like, optional
        The transmission function, (1 - Gamma_rec^2). Computed if not given.

    Returns
    -------
    K0, K1, K2, K3: array_like
        Factors corresponding to T_ant, T_unc, T_cos, T_sin respectively.
    """
    # Get F and alpha for each load (Eqs. 3 and 4)
    if f_ratio is None:
        f_ratio = get_F(gamma_rec=gamma_rec, gamma_ant=gamma_ant)

    if alpha is None:
        alpha = get_alpha(gamma_rec=gamma_rec, gamma_ant=gamma_ant)

    # The denominator of each term in Eq. 7
    if gain is None:
        gain = 1 - np.abs(gamma_rec) ** 2

    f_ratio = np.abs(f_ratio)
    gant = np.abs(gamma_ant)
    fgant = gant * f_ratio / gain

    K2 = fgant ** 2 * gain
    K1 = f_ratio ** 2 / gain - K2
    K3 = fgant * np.cos(alpha)
    K4 = fgant * np.sin(alpha)

    return K1, K2, K3, K4


def get_calibration_quantities_iterative(
    f_norm: np.ndarray,
    temp_raw: dict,
    gamma_rec: np.ndarray,
    gamma_ant: dict,
    temp_ant: dict,
    cterms: int,
    wterms: int,
    temp_amb_internal: float = 300,
):
    """
    Derive calibration parameters using the scheme laid out in Monsalve (2017) [arxiv:1602.08065].

    All equation numbers and symbol names come from M17.

    Parameters
    ----------
    f_norm : array_like
        Normalized frequencies (arbitrarily normalised, but standard assumption is
        that the centre is zero, and the scale is such that the range is (-1, 1))
    temp_raw : dict
        Dictionary of antenna uncalibrated temperatures, with keys
        'ambient', 'hot_load, 'short' and 'open'. Each value is an array with the same
        length as f_norm.
    gamma_rec : float array
        Receiver S11 as a function of frequency.
    gamma_ant : dict
        Dictionary of antenna S11, with keys 'ambient', 'hot_load, 'short'
        and 'open'. Each value is an array with the same length as f_norm.
    temp_ant : dict
        Dictionary like `gamma_ant`, except that the values are modelled/smoothed
        thermistor temperatures for each source load.
    cterms : int
        Number of polynomial terms for the C_i
    wterms : int
        Number of polynonmial temrs for the T_i
    temp_amb_internal : float
        The ambient internal temperature, interpreted as T_L.
        Note: this must be the same as the T_L used to generate T*.


    Returns
    -------
    sca, off, tu, tc, ts : np.poly1d
        1D polynomial fits for each of the Scale (C_1), Offset (C_2), and noise-wave
        temperatures for uncorrelated, cos and sin components.
    """
    mask = (
        (~np.isnan(temp_raw["short"]))
        * (~np.isnan(temp_raw["ambient"]))
        * (~np.isnan(temp_raw["hot_load"]))
        * ~np.isnan(temp_raw["open"])
    )

    fmask = f_norm[mask]
    gamma_ant = {key: value[mask] for key, value in gamma_ant.items()}
    temp_raw = {key: value[mask] for key, value in temp_raw.items()}
    gamma_rec = gamma_rec[mask]
    temp_ant["hot_load"] = temp_ant["hot_load"][mask]

    # Get F and alpha for each load (Eqs. 3 and 4)
    F = {k: get_F(gamma_rec, v) for k, v in gamma_ant.items()}
    alpha = {k: get_alpha(gamma_rec, v) for k, v in gamma_ant.items()}

    # The denominator of each term in Eq. 7
    G = 1 - np.abs(gamma_rec) ** 2

    K1, K2, K3, K4 = {}, {}, {}, {}
    for k, gamma_a in gamma_ant.items():
        K1[k], K2[k], K3[k], K4[k] = get_K(
            gamma_rec, gamma_a, f_ratio=F[k], gain=G, alpha=alpha[k]
        )

    # Initializing arrays
    niter = 4
    ta_iter = np.zeros((niter, len(fmask)))
    th_iter = np.zeros((niter, len(fmask)))

    sca = np.zeros((niter, len(fmask)))
    off = np.zeros((niter, len(fmask)))

    # Calibrated temperature iterations
    temp_cal_iter = {k: np.zeros((niter, len(fmask))) for k in temp_ant}

    tunc = np.zeros((niter, len(fmask)))
    tcos = np.zeros((niter, len(fmask)))
    tsin = np.zeros((niter, len(fmask)))

    # Calibration loop
    for i in range(niter):
        # Step 1: approximate physical temperature
        if i == 0:
            ta_iter[i, :] = temp_raw["ambient"] / K1["ambient"]
            th_iter[i, :] = temp_raw["hot_load"] / K1["hot_load"]
        else:
            for load, arry in zip(["ambient", "hot_load"], (ta_iter, th_iter)):
                noise_wave_param = (
                    tunc[i - 1, :] * K2[load]
                    + tcos[i - 1, :] * K3[load]
                    + tsin[i - 1, :] * K4[load]
                )
                arry[i, :] = (temp_cal_iter[load][i - 1, :] - noise_wave_param) / K1[
                    load
                ]

        # Step 2: scale and offset

        # Updating scale and offset
        sca_new = (temp_ant["hot_load"] - temp_ant["ambient"]) / (
            th_iter[i, :] - ta_iter[i, :]
        )
        off_new = ta_iter[i, :] - temp_ant["ambient"]

        if i == 0:
            sca_raw = sca_new
            off_raw = off_new
        else:
            sca_raw = sca[i - 1, :] * sca_new
            off_raw = off[i - 1, :] + off_new

        # Modeling scale
        p_sca = np.polyfit(fmask, sca_raw, cterms - 1)
        sca[i, :] = np.polyval(p_sca, fmask)

        # Modeling offset
        p_off = np.polyfit(fmask, off_raw, cterms - 1)
        off[i, :] = np.polyval(p_off, fmask)

        # Step 3: corrected "uncalibrated spectrum" of cable
        for k, v in temp_cal_iter.items():
            v[i, :] = (
                (temp_raw[k] - temp_amb_internal) * sca[i, :]
                + temp_amb_internal
                - off[i, :]
            )

        # Step 4: computing NWP
        tu, tc, ts = noise_wave_param_fit(
            fmask,
            gamma_rec,
            gamma_ant["open"],
            gamma_ant["short"],
            temp_cal_iter["open"][i, :],
            temp_cal_iter["short"][i, :],
            temp_ant["open"],
            temp_ant["short"],
            wterms,
        )

        tunc[i] = tu(fmask)
        tcos[i] = tc(fmask)
        tsin[i] = ts(fmask)

    return np.poly1d(p_sca), np.poly1d(p_off), tu, tc, ts


def get_linear_coefficients(
    gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """
    Use Monsalve (2017) Eq. 7 to determine a and b, such that T = aT* + b.

    Parameters
    ----------
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    K = get_K(gamma_rec, gamma_ant)

    return get_linear_coefficients_from_K(K, sca, off, t_unc, t_cos, t_sin, t_load)


def get_linear_coefficients_from_K(  # noqa: N802
    k, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """Calculate linear coefficients a and b from noise-wave parameters K0-4.

    Parameters
    ----------
    k : np.ndarray
        Shape (4, nfreq) array with each of the K-coefficients.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    # Noise wave contribution
    noise_wave_terms = t_unc * k[1] + t_cos * k[2] + t_sin * k[3]

    return sca / k[0], (t_load - off - noise_wave_terms - t_load * sca) / k[0]


def calibrated_antenna_temperature(
    temp_raw, gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """
    Use M17 Eq. 7 to determine calibrated temperature from an uncalibrated temperature.

    Parameters
    ----------
    temp_raw : array_like
        The raw (uncalibrated) temperature spectrum, T*.
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    a, b = get_linear_coefficients(
        gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load
    )

    return temp_raw * a + b


def uncalibrated_antenna_temperature(
    temp, gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """
    Use M17 Eq. 7 to determine uncalibrated temperature from a calibrated temperature.

    Parameters
    ----------
    temp : array_like
        The true (or calibrated) temperature spectrum.
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    a, b = get_linear_coefficients(
        gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load
    )
    return (temp - b) / a


def model_q_no_noise_wave(
    t_ant: [float, np.ndarray],
    t_ns: [float, np.ndarray],
    t_l: [float, np.ndarray],
    t_lna: [float, np.ndarray],
    alpha: float,
) -> [float, np.ndarray]:
    """
    Analytic model for Q given internal temperatures, assuming perfect match.

    Parameters
    ----------
    t_ant
        Antenna temperature
    t_ns
        Noise-source temperature
    t_l
        Load temperature
    t_lna
        Receiver/LNA temperature
    alpha
        Inverse of effective number of samples in the integration, 1 / (t * df).

    Returns
    -------
    Q
        The 3-position switch ratio model.
    """
    front = (t_ant - t_l) / t_ns
    return front * (
        1
        - alpha * (t_l ** 2 + t_lna ** 2) / (t_ns * (t_ant - t_l))
        + alpha * (2 * t_l ** 2 + t_ns ** 2 + 2 * t_lna ** 2) / t_ns ** 2
    )


def var_q_no_noise_wave(
    t_ant: [float, np.ndarray],
    t_ns: [float, np.ndarray],
    t_l: [float, np.ndarray],
    t_lna: [float, np.ndarray],
    alpha: float,
) -> [float, np.ndarray]:
    """
    Analytic model for variance of Q given internal temperatures, assuming perfect match.

    Parameters
    ----------
    t_ant
        Antenna temperature
    t_ns
        Noise-source temperature
    t_l
        Load temperature
    t_lna
        Receiver/LNA temperature
    alpha
        Inverse of effective number of samples in the integration, 1 / (t * df).

    Returns
    -------
    varQ
        The 3-position switch ratio variance model.
    """
    front = alpha * (t_ant - t_l) ** 2 / t_ns ** 2
    return front * (
        (t_ant ** 2 + 2 * t_lna ** 2 + t_l ** 2) / (t_ant - t_l) ** 2
        - 2 * (t_l ** 2 + t_lna ** 2) / (t_ns * (t_ant - t_l))
        + (2 * t_l ** 2 + 2 * t_lna ** 2 + t_ns ** 2) / t_ns ** 2
    )


def get_calibration_quantities_no_noise_wave(
    freq: np.ndarray,
    q_measured: Dict[str, np.ndarray],
    temp_ant: Dict[str, np.ndarray],
    n_terms: int,
    alpha: float,
    guess: List[float],
    var_q_measured: Optional[Dict[str, np.ndarray]] = None,
):
    """Obtain calibration quantities using the likelihood formalism in Murray 2020.

    Parameters
    ----------
    freq
    q_measured
    var_q_measured
    temp_ant
    n_terms

    Returns
    -------
    t_ns, t_l, t_lna
        Numpy polynomials for noise-source, load and receiver temperature.
    """

    def get_poly(p):
        t_ns = np.poly1d(p[:n_terms])
        t_l = np.poly1d(p[n_terms : 2 * n_terms])
        t_lna = np.poly1d(p[2 * n_terms : 3 * n_terms])
        return t_ns, t_l, t_lna

    # Define a model of Q
    def q_model(p, t_ant: [float, np.ndarray], get_var=False):
        t_ns, t_l, t_lna = get_poly(p)
        t_ns = t_ns(freq)
        t_l = t_l(freq)
        t_lna = t_lna(freq)
        mean = model_q_no_noise_wave(t_ant, t_ns, t_l, t_lna, alpha)
        if get_var:
            var = var_q_no_noise_wave(t_ant, t_ns, t_l, t_lna, alpha)
            return mean, var
        return mean

    # Set initial positions to be constant temperatures at the "guess"
    p0 = np.zeros(3 * n_terms)
    p0[::n_terms] = guess

    if var_q_measured is not None:
        # Use robust LM method to fit

        # Define the chi^2 log probability
        def obj_func(freq, *p):
            out = np.zeros(len(temp_ant) * len(freq))

            for i, (name, temp) in enumerate(temp_ant.items()):
                mean = q_model(p, temp)
                out[i * len(freq) : (i + 1) * len(freq)] = mean

            return out

        # Perform a fit
        ydata = np.zeros(len(temp_ant) * len(freq))
        sigma = np.zeros(len(temp_ant) * len(freq))
        for i, (name, temp) in enumerate(q_measured.items()):
            ydata[i * len(freq) : (i + 1) * len(freq)] = temp
            sigma[i * len(freq) : (i + 1) * len(freq)] = np.sqrt(var_q_measured[name])

        fit, cov = curve_fit(
            obj_func, freq, ydata, p0=p0, sigma=sigma, absolute_sigma=True,
        )

        return get_poly(fit)

    else:
        # Use standard minimize routine
        def log_prob(p):
            lnl = 0
            for i, (name, temp) in enumerate(temp_ant.items()):
                mean, var = q_model(p, temp, get_var=True)
                lnl += np.sum(
                    norm.logpdf(q_measured[name], loc=mean, scale=np.sqrt(var))
                )

            return -lnl

        res = minimize(log_prob, p0)

        return get_poly(res.x)
