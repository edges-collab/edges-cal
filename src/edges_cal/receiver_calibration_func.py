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


def NWP_fit(f_norm, rl, ro, rs, Toe, Tse, To, Ts, wterms):
    """
    Fit noise-wave polynomial parameters.

    Parameters
    ----------
    f_norm : array_like
        Normalized frequencies (arbitrarily normalised, but standard assumption is
        that the centre is zero, and the scale is such that the range is (-1, 1))
    rl
    ro
    rs
    Toe
    Tse
    To
    Ts
    wterms : int
        The number of polynomial terms to use for each of the noise-wave functions.

    Returns
    -------
    Tunc, Tcos, Tsin : array_like
        The solutions to each of T_unc, T_cos and T_sin as functions of frequency.
    """
    # S11 quantities
    Fo = np.sqrt(1 - np.abs(rl) ** 2) / (1 - ro * rl)
    Fs = np.sqrt(1 - np.abs(rl) ** 2) / (1 - rs * rl)
    PHIo = np.angle(ro * Fo)
    PHIs = np.angle(rs * Fs)
    G = 1 - np.abs(rl) ** 2
    K1o = (1 - np.abs(ro) ** 2) * (np.abs(Fo) ** 2) / G
    K1s = (1 - np.abs(rs) ** 2) * (np.abs(Fs) ** 2) / G

    K2o = (np.abs(ro) ** 2) * (np.abs(Fo) ** 2) / G
    K2s = (np.abs(rs) ** 2) * (np.abs(Fs) ** 2) / G

    K3o = (np.abs(ro) * np.abs(Fo) / G) * np.cos(PHIo)
    K3s = (np.abs(rs) * np.abs(Fs) / G) * np.cos(PHIs)
    K4o = (np.abs(ro) * np.abs(Fo) / G) * np.sin(PHIo)
    K4s = (np.abs(rs) * np.abs(Fs) / G) * np.sin(PHIs)

    # Matrices A and b
    A = np.zeros((3 * wterms, 2 * len(f_norm)))
    for i in range(wterms):
        A[i, :] = np.append(K2o * f_norm ** i, K2s * f_norm ** i)
        A[i + 1 * wterms, :] = np.append(K3o * f_norm ** i, K3s * f_norm ** i)
        A[i + 2 * wterms, :] = np.append(K4o * f_norm ** i, K4s * f_norm ** i)
    b = np.append((Toe - To * K1o), (Tse - Ts * K1s))

    # Transposing matrices so 'frequency' dimension is along columns
    M = A.T
    ydata = np.reshape(b, (-1, 1))

    # Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
    Q1, R1 = sp.linalg.qr(M, mode="economic")
    param = sp.linalg.solve(R1, np.dot(Q1.T, ydata))

    # Evaluating TU, TC, and TS
    TU = np.zeros(len(f_norm))
    TC = np.zeros(len(f_norm))
    TS = np.zeros(len(f_norm))

    for i in range(wterms):
        TU = TU + param[i, 0] * f_norm ** i
        TC = TC + param[i + 1 * wterms, 0] * f_norm ** i
        TS = TS + param[i + 2 * wterms, 0] * f_norm ** i

    return TU, TC, TS


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
    F = get_F(gamma_rec, gamma_ant)
    alpha = get_alpha(gamma_rec, gamma_ant)

    terms = [
        temp_ant * (1 - np.abs(gamma_ant) ** 2) * np.abs(F) ** 2,
        temp_unc * np.abs(gamma_ant) ** 2 * np.abs(F) ** 2,
        np.abs(gamma_ant) * np.abs(F) * temp_cos * np.cos(alpha),
        np.abs(gamma_ant) * np.abs(F) * temp_sin * np.sin(alpha),
        (offset - temp_load),
        scale * temp_noise_source * (1 - np.abs(gamma_rec) ** 2),
    ]

    if return_terms:
        return terms
    else:
        return sum(terms[:5]) / terms[5]


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
    # Get F and alpha for each load (Eqs. 3 and 4)
    F = {k: get_F(gamma_rec, v) for k, v in gamma_ant.items()}
    alpha = {k: get_alpha(gamma_rec, v) for k, v in gamma_ant.items()}

    # The denominator of each term in Eq. 7
    G = 1 - np.abs(gamma_rec) ** 2

    K1, K2, K3, K4 = {}, {}, {}, {}
    for k, gamma_a in gamma_ant.items():
        K1[k] = (1 - np.abs(gamma_a) ** 2) * np.abs(F[k]) ** 2 / G
        K2[k] = (np.abs(gamma_a) ** 2) * (np.abs(F[k]) ** 2) / G
        K3[k] = (np.abs(gamma_a) * np.abs(F[k]) / G) * np.cos(alpha[k])
        K4[k] = (np.abs(gamma_a) * np.abs(F[k]) / G) * np.sin(alpha[k])

    # Initializing arrays
    niter = 4
    Ta_iter = np.zeros((niter, len(f_norm)))
    Th_iter = np.zeros((niter, len(f_norm)))

    sca = np.zeros((niter, len(f_norm)))
    off = np.zeros((niter, len(f_norm)))

    # Calibrated temperature iterations
    T_cal_iter = {k: np.zeros((niter, len(f_norm))) for k in T_ant}

    TU = np.zeros((niter, len(f_norm)))
    TC = np.zeros((niter, len(f_norm)))
    TS = np.zeros((niter, len(f_norm)))

    # Calibration loop
    for i in range(niter):
        # Step 1: approximate physical temperature
        if i == 0:
            Ta_iter[i, :] = T_raw["ambient"] / K1["ambient"]
            Th_iter[i, :] = T_raw["hot_load"] / K1["hot_load"]

        if i > 0:
            NWPa = (
                TU[i - 1, :] * K2["ambient"]
                + TC[i - 1, :] * K3["ambient"]
                + TS[i - 1, :] * K4["ambient"]
            )
            NWPh = (
                TU[i - 1, :] * K2["hot_load"]
                + TC[i - 1, :] * K3["hot_load"]
                + TS[i - 1, :] * K4["hot_load"]
            )

            Ta_iter[i, :] = (T_cal_iter["ambient"][i - 1, :] - NWPa) / K1["ambient"]
            Th_iter[i, :] = (T_cal_iter["hot_load"][i - 1, :] - NWPh) / K1["hot_load"]

        # Step 2: scale and offset

        # Updating scale and offset
        sca_new = (T_ant["hot_load"] - T_ant["ambient"]) / (
            Th_iter[i, :] - Ta_iter[i, :]
        )
        off_new = Ta_iter[i, :] - T_ant["ambient"]

        if i == 0:
            sca_raw = sca_new
            off_raw = off_new
        if i > 0:
            sca_raw = sca[i - 1, :] * sca_new
            off_raw = off[i - 1, :] + off_new

        # Modeling scale
        p_sca = np.polyfit(f_norm, sca_raw, cterms - 1)
        m_sca = np.polyval(p_sca, f_norm)
        sca[i, :] = m_sca

        # Modeling offset
        p_off = np.polyfit(f_norm, off_raw, cterms - 1)
        m_off = np.polyval(p_off, f_norm)
        off[i, :] = m_off

        # Step 3: corrected "uncalibrated spectrum" of cable
        for k, v in T_cal_iter.items():
            v[i, :] = (T_raw[k] - Tamb_internal) * sca[i, :] + Tamb_internal - off[i, :]

        # Step 4: computing NWP
        TU[i, :], TC[i, :], TS[i, :] = NWP_fit(
            f_norm,
            gamma_rec,
            gamma_ant["open"],
            gamma_ant["short"],
            T_cal_iter["open"][i, :],
            T_cal_iter["short"][i, :],
            T_ant["open"],
            T_ant["short"],
            wterms,
        )

    return sca[-1, :], off[-1, :], TU[-1, :], TC[-1, :], TS[-1, :]


def calibrated_antenna_temperature(
    Tde, rd, rl, sca, off, TU, TC, TS, Tamb_internal=300
):
    """
    Function for equation (7)
    rd - refelection coefficient of the load
    rl - reflection coefficient of the receiver
    Td - temperature of the device under test
    TU ,Tc,Ts - noise wave parameters
    Tamb_internal - noise temperature of the load
    """

    # S11 quantities
    Fd = np.sqrt(1 - np.abs(rl) ** 2) / (1 - rd * rl)
    PHId = np.angle(rd * Fd)
    G = 1 - np.abs(rl) ** 2
    K1d = (1 - np.abs(rd) ** 2) * np.abs(Fd) ** 2 / G
    K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
    K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
    K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)

    # Applying scale and offset to raw spectrum
    Tde_corrected = (Tde - Tamb_internal) * sca + Tamb_internal - off

    # Noise wave contribution
    NWPd = TU * K2d + TC * K3d + TS * K4d

    # Antenna temperature
    Td = (Tde_corrected - NWPd) / K1d

    return Td


def uncalibrated_antenna_temperature(
    Td, rd, rl, sca, off, TU, TC, TS, Tamb_internal=300
):
    """
    Function for equation (7)
    rd - refelection coefficient of the load
    rl - reflection coefficient of the receiver
    Td - temperature of the device under test
    TU ,Tc,Ts - noise wave parameters
    Tamb_internal - noise temperature of the load
    """
    # S11 quantities
    Fd = np.sqrt(1 - np.abs(rl) ** 2) / (1 - rd * rl)
    PHId = np.angle(rd * Fd)
    G = 1 - np.abs(rl) ** 2
    K1d = (1 - np.abs(rd) ** 2) * np.abs(Fd) ** 2 / G
    K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
    K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
    K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)

    # Noise wave contribution
    NWPd = TU * K2d + TC * K3d + TS * K4d

    # Scaled and offset spectrum
    Tde_corrected = Td * K1d + NWPd

    # Removing scale and offset
    Tde = Tamb_internal + (Tde_corrected - Tamb_internal + off) / sca

    return Tde
