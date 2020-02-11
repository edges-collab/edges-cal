# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 14:02:33 2018

@author: Nivedita
"""
import numpy as np
import scipy as sp


def temperature_thermistor(resistance, coeffs="oven_industries_TR136_170", kelvin=True):
    """
    Convert a resistance measurement to a temperature, using a pre-defined set
    of standard coefficients.

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


def NWP_fit(
    f_norm,
    gamma_rec,
    gamma_open,
    gamma_short,
    T_raw_open,
    T_raw_short,
    T_thermistor_open,
    T_thermistor_short,
    wterms,
):
    """
    Fit noise-wave polynomial parameters.

    Parameters
    ----------
    f_norm : array_like
        Normalized frequencies (arbitrarily normalised, but standard assumption is
        that the centre is zero, and the scale is such that the range is (-1, 1))
    gamma_rec
    gamma_open
    gamma_short
    T_raw_open
    T_raw_short
    T_thermistor_open
    T_thermistor_short
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
        (T_raw_open - T_thermistor_open * K1o), (T_raw_short - T_thermistor_short * K1s)
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

    # # Evaluating TU, TC, and TS
    # TU = np.zeros(len(f_norm))
    # TC = np.zeros(len(f_norm))
    # TS = np.zeros(len(f_norm))
    #
    # for i in range(wterms):
    #     TU = TU + param[i, 0] * f_norm ** i
    #     TC = TC + param[i + 1 * wterms, 0] * f_norm ** i
    #     TS = TS + param[i + 2 * wterms, 0] * f_norm ** i

    # return TU, TC, TS


def get_F(gamma_rec, gamma_ant):
    """Get the F parameter for a given receiver and antenna"""
    return np.sqrt(1 - np.abs(gamma_rec) ** 2) / (1 - gamma_ant * gamma_rec)


def get_alpha(gamma_rec, gamma_ant):
    """Get the alpha parameter for a given receiver and antenna"""
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

    Computes (as a model)

    .. math :: Q_P = (P_ant - P_L)/(P_NS - P_L)

    Parameters
    ----------
    freqs : array_like
        Frequencies of the LoadSpectrum.
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


def get_K(gamma_rec, gamma_ant, F=None, alpha=None, G=None):
    """
    Determine the S11-dependent factors for each term in Eq. 7 (Monsalve 2017).

    Parameters
    ----------
    gamma_rec : array_like
        Receiver S11
    gamma_ant : array_like
        Antenna (or load) S11.
    F : array_like, optional
        The F factor (Eq. 3 of Monsalve 2017). Computed if not given.
    alpha : array_like, optional
        The alpha factor (Eq. 4 of Monsalve, 2017). Computed if not given.
    G : array_like, optional
        The transmission function, (1 - Gamma_rec^2). Computed if not given.

    Returns
    -------
    K0, K1, K2, K3: array_like
        Factors corresponding to T_ant, T_unc, T_cos, T_sin respectively.
    """
    # Get F and alpha for each load (Eqs. 3 and 4)
    if F is None:
        F = get_F(gamma_rec=gamma_rec, gamma_ant=gamma_ant)

    if alpha is None:
        alpha = get_alpha(gamma_rec=gamma_rec, gamma_ant=gamma_ant)

    # The denominator of each term in Eq. 7
    if G is None:
        G = 1 - np.abs(gamma_rec) ** 2

    F = np.abs(F)
    gant = np.abs(gamma_ant)
    fgant = gant * F / G

    K2 = fgant ** 2 * G
    K1 = F ** 2 / G - K2
    K3 = fgant * np.cos(alpha)
    K4 = fgant * np.sin(alpha)

    return K1, K2, K3, K4


def get_calibration_quantities_iterative(
    f_norm, T_raw, gamma_rec, gamma_ant, T_ant, cterms, wterms, Tamb_internal=300
):
    """
    Derive calibration parameters {C1, C2, Tunc, Tcos, Tsin} using the scheme laid
    out in Monsalve (2017) [arxiv:1602.08065].

    All equation numbers and symbol names come from M17.

    Parameters
    ----------
    f_norm : array_like
        Normalized frequencies (arbitrarily normalised, but standard assumption is
        that the centre is zero, and the scale is such that the range is (-1, 1))
    T_raw : dict
        Dictionary of antenna uncalibrated temperatures, with keys
        'ambient', 'hot_load, 'short' and 'open'. Each value is an array with the same
        length as f_norm.
    gamma_rec : float array
        Receiver S11 as a function of frequency.
    gamma_ant : dict
        Dictionary of antenna S11, with keys 'ambient', 'hot_load, 'short'
        and 'open'. Each value is an array with the same length as f_norm.
    T_ant : dict
        Dictionary like `gamma_ant`, except that the values are modelled/smoothed
        thermistor temperatures for each source load.
    cterms : int
        Number of polynomial terms for the C_i
    wterms : int
        Number of polynonmial temrs for the T_i
    Tamb_internal : float
        The ambient internal temperature, interpreted as T_L.
        Note: this must be the same as the T_L used to generate T*.


    Returns
    -------

    """
    mask = (
        (~np.isnan(T_raw["short"]))
        * (~np.isnan(T_raw["ambient"]))
        * (~np.isnan(T_raw["hot_load"]))
        * ~np.isnan(T_raw["open"])
    )

    fmask = f_norm[mask]
    gamma_ant = {key: value[mask] for key, value in gamma_ant.items()}
    T_raw = {key: value[mask] for key, value in T_raw.items()}
    gamma_rec = gamma_rec[mask]
    T_ant["hot_load"] = T_ant["hot_load"][mask]

    # Get F and alpha for each load (Eqs. 3 and 4)
    F = {k: get_F(gamma_rec, v) for k, v in gamma_ant.items()}
    alpha = {k: get_alpha(gamma_rec, v) for k, v in gamma_ant.items()}

    # The denominator of each term in Eq. 7
    G = 1 - np.abs(gamma_rec) ** 2

    K1, K2, K3, K4 = {}, {}, {}, {}
    for k, gamma_a in gamma_ant.items():
        K1[k], K2[k], K3[k], K4[k] = get_K(
            gamma_rec, gamma_a, F=F[k], G=G, alpha=alpha[k]
        )

    # Initializing arrays
    niter = 4
    Ta_iter = np.zeros((niter, len(fmask)))
    Th_iter = np.zeros((niter, len(fmask)))

    sca = np.zeros((niter, len(fmask)))
    off = np.zeros((niter, len(fmask)))

    # Calibrated temperature iterations
    T_cal_iter = {k: np.zeros((niter, len(fmask))) for k in T_ant}

    TU = np.zeros((niter, len(fmask)))
    TC = np.zeros((niter, len(fmask)))
    TS = np.zeros((niter, len(fmask)))

    # Calibration loop
    for i in range(niter):
        # Step 1: approximate physical temperature
        if i == 0:
            Ta_iter[i, :] = T_raw["ambient"] / K1["ambient"]
            Th_iter[i, :] = T_raw["hot_load"] / K1["hot_load"]
        else:
            for load, arry in zip(["ambient", "hot_load"], (Ta_iter, Th_iter)):
                noise_wave_param = (
                    TU[i - 1, :] * K2[load]
                    + TC[i - 1, :] * K3[load]
                    + TS[i - 1, :] * K4[load]
                )
                arry[i, :] = (T_cal_iter[load][i - 1, :] - noise_wave_param) / K1[load]

        # Step 2: scale and offset

        # Updating scale and offset
        sca_new = (T_ant["hot_load"] - T_ant["ambient"]) / (
            Th_iter[i, :] - Ta_iter[i, :]
        )
        off_new = Ta_iter[i, :] - T_ant["ambient"]

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
        for k, v in T_cal_iter.items():
            v[i, :] = (T_raw[k] - Tamb_internal) * sca[i, :] + Tamb_internal - off[i, :]

        # Step 4: computing NWP
        tu, tc, ts = NWP_fit(
            fmask,
            gamma_rec,
            gamma_ant["open"],
            gamma_ant["short"],
            T_cal_iter["open"][i, :],
            T_cal_iter["short"][i, :],
            T_ant["open"],
            T_ant["short"],
            wterms,
        )

        TU[i] = tu(fmask)
        TC[i] = tc(fmask)
        TS[i] = ts(fmask)

    return (np.poly1d(p_sca), np.poly1d(p_off), tu, tc, ts)


def get_linear_coefficients(gamma_ant, gamma_rec, sca, off, TU, TC, TS, T_load=300):
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
    TU, TC, TS : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    T_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    K = get_K(gamma_rec, gamma_ant)

    return get_linear_coefficients_from_K(K, sca, off, TU, TC, TS, T_load)


def get_linear_coefficients_from_K(K, sca, off, TU, TC, TS, T_load=300):
    # Noise wave contribution
    noise_wave_terms = TU * K[1] + TC * K[2] + TS * K[3]

    return sca / K[0], (T_load - off - noise_wave_terms - T_load * sca) / K[0]


def calibrated_antenna_temperature(
    temp_raw, gamma_ant, gamma_rec, sca, off, TU, TC, TS, T_load=300
):
    """
    Use Monsalve (2017) Eq. 7 to determine calibrated (or "true") temperature
    from an uncalibrated temperature.

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
    TU, TC, TS : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    T_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    a, b = get_linear_coefficients(gamma_ant, gamma_rec, sca, off, TU, TC, TS, T_load)

    return temp_raw * a + b
    # K = get_K(gamma_rec, gamma_ant)
    #
    # # # S11 quantities
    # # Fd = np.sqrt(1 - np.abs(gamma_rec) ** 2) / (1 - gamma_ant * gamma_rec)
    # # PHId = np.angle(gamma_ant * Fd)
    # # G = 1 - np.abs(gamma_rec) ** 2
    # # K1d = (1 - np.abs(gamma_ant) ** 2) * np.abs(Fd) ** 2 / G
    # # K2d = (np.abs(gamma_ant) ** 2) * (np.abs(Fd) ** 2) / G
    # # K3d = (np.abs(gamma_ant) * np.abs(Fd) / G) * np.cos(PHId)
    # # K4d = (np.abs(gamma_ant) * np.abs(Fd) / G) * np.sin(PHId)
    #
    # # Applying scale and offset to raw spectrum
    # # Gives the LHS of Eq. 7
    # temp_corrected = (temp_raw - T_load) * sca + T_load - off
    #
    # # Noise wave contribution
    # noise_wave_terms = TU * K[1] + TC * K[2] + TS * K[3]
    #
    # print(sca/K[0], (T_load-off - noise_wave_terms - T_load*sca)/K[0])
    #
    # # Antenna temperature
    # return (temp_corrected - noise_wave_terms) / K[0]


def uncalibrated_antenna_temperature(
    temp, gamma_ant, gamma_rec, sca, off, TU, TC, TS, T_load=300
):
    """
    Use Monsalve (2017) Eq. 7 to determine calibrated (or "true") temperature
    from an uncalibrated temperature.

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
    TU, TC, TS : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    T_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    a, b = get_linear_coefficients(gamma_ant, gamma_rec, sca, off, TU, TC, TS, T_load)
    return (temp - b) / a

    # K = get_K(gamma_rec, gamma_ant)
    #
    # # # S11 quantities
    # # Fd = np.sqrt(1 - np.abs(rl) ** 2) / (1 - rd * rl)
    # # PHId = np.angle(rd * Fd)
    # # G = 1 - np.abs(rl) ** 2
    # # K1d = (1 - np.abs(rd) ** 2) * np.abs(Fd) ** 2 / G
    # # K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
    # # K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
    # # K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)
    #
    # # Noise wave contribution
    # noise_wave_terms = TU * K[1] + TC * K[2] + TS * K[3]
    #
    # # Scaled and offset spectrum
    # # This is the full RHS of Eq. 7
    # Tde_corrected = temp * K[0] + noise_wave_terms
    #
    # # Removing scale and offset
    # return T_load + (Tde_corrected - T_load + off) / sca
