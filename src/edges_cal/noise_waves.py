"""Functions for calibrating the receiver."""
from __future__ import annotations

import numpy as np

from . import modelling as mdl


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

    return terms if return_terms else sum(terms[:5]) / terms[5]


def noise_wave_param_fit(
    freq: np.ndarray,
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
    freq : array_like
        Frequencies at which the data was taken.
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
    if np.any(np.isnan(freq)):
        raise ValueError("Some frequencies are NaN")
    if np.any(np.isnan(gamma_rec)):
        raise ValueError("Some receiver reflection coefficients are NaN")
    if np.any(np.isnan(gamma_open)):
        raise ValueError("Some open reflection coefficients are NaN")
    if np.any(np.isnan(gamma_short)):
        raise ValueError("Some short reflection coefficients are NaN")
    if np.any(np.isnan(temp_raw_open)):
        raise ValueError("Some open raw temperatures are NaN")
    if np.any(np.isnan(temp_raw_short)):
        raise ValueError("Some short raw temperatures are NaN")
    if np.any(np.isnan(temp_thermistor_open)):
        raise ValueError("Some open thermistor temperatures are NaN")
    if np.any(np.isnan(temp_thermistor_short)):
        raise ValueError("Some short thermistor temperatures are NaN")

    Kopen = get_K(gamma_rec, gamma_open)
    Kshort = get_K(gamma_rec, gamma_short)

    tr = mdl.ScaleTransform(scale=freq[len(freq) // 2])

    models = {
        name: mdl.Polynomial(n_terms=wterms, transform=tr)
        for name in ["tunc", "tcos", "tsin"]
    }

    extra_basis = {
        "tunc": np.concatenate((Kopen[1], Kshort[1])),
        "tcos": np.concatenate((Kopen[2], Kshort[2])),
        "tsin": np.concatenate((Kopen[3], Kshort[3])),
    }

    model = mdl.CompositeModel(models=models, extra_basis=extra_basis).at(
        x=np.concatenate((freq, freq))
    )

    fit = model.fit(
        ydata=np.concatenate(
            (
                (temp_raw_open - temp_thermistor_open * Kopen[0]),
                (temp_raw_short - temp_thermistor_short * Kshort[0]),
            )
        )
    ).fit

    return fit.model["tunc"], fit.model["tcos"], fit.model["tsin"]


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

    K2 = fgant**2 * gain
    K1 = f_ratio**2 / gain - K2
    K3 = fgant * np.cos(alpha)
    K4 = fgant * np.sin(alpha)

    return K1, K2, K3, K4


def get_calibration_quantities_iterative(
    freq: np.ndarray,
    temp_raw: dict,
    gamma_rec: np.ndarray,
    gamma_ant: dict,
    temp_ant: dict,
    cterms: int,
    wterms: int,
    temp_amb_internal: float = 300,
    niter: int = 4,
):
    """
    Derive calibration parameters using the scheme laid out in Monsalve (2017).

    All equation numbers and symbol names come from M17 (arxiv:1602.08065).

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
    mask = np.all([np.isfinite(temp) for temp in temp_raw.values()], axis=0)

    fmask = freq[mask]
    gamma_ant = {key: value[mask] for key, value in gamma_ant.items()}
    temp_raw = {key: value[mask] for key, value in temp_raw.items()}
    temp_ant = {
        key: (value[mask] if hasattr(value, "__len__") else value)
        for key, value in temp_ant.items()
    }
    gamma_rec = gamma_rec[mask]
    temp_ant_hot = temp_ant["hot_load"]

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
    nf = len(fmask)
    ta_iter = np.zeros(nf)
    th_iter = np.zeros(nf)

    sca, off, tunc, tcos, tsin = (
        np.ones(nf),
        np.zeros(nf),
        np.zeros(nf),
        np.zeros(nf),
        np.zeros(nf),
    )

    tr = mdl.ScaleTransform(scale=freq[len(freq) // 2])
    sca_mdl = mdl.Polynomial(n_terms=cterms, transform=tr).at(x=fmask)
    off_mdl = mdl.Polynomial(n_terms=cterms, transform=tr).at(x=fmask)

    temp_cal_iter = {}
    # Calibration loop
    for i in range(niter):
        # Step 1: approximate physical temperature
        if i == 0:
            ta_iter = temp_raw["ambient"] / K1["ambient"]
            th_iter = temp_raw["hot_load"] / K1["hot_load"]
        else:
            for load, arry in zip(["ambient", "hot_load"], (ta_iter, th_iter)):
                noise_wave_param = tunc * K2[load] + tcos * K3[load] + tsin * K4[load]
                arry[:] = (temp_cal_iter[load] - noise_wave_param) / K1[load]

        # Step 2: scale and offset

        # Updating scale and offset
        sca_new = (temp_ant_hot - temp_ant["ambient"]) / (th_iter - ta_iter)
        off_new = ta_iter - temp_ant["ambient"]

        if i == 0:
            sca_raw = sca_new
            off_raw = off_new
        else:
            sca_raw = sca * sca_new
            off_raw = off + off_new

        # Model scale
        p_sca = sca_mdl.fit(ydata=sca_raw).fit
        sca = p_sca(fmask)

        # Model offset
        p_off = off_mdl.fit(ydata=off_raw).fit
        off = p_off(fmask)

        # Step 3: corrected "uncalibrated spectrum" of cable
        temp_cal_iter = {
            k: (v - temp_amb_internal) * sca + temp_amb_internal - off
            for k, v in temp_raw.items()
        }

        # Step 4: computing NWP
        tu, tc, ts = noise_wave_param_fit(
            fmask,
            gamma_rec,
            gamma_ant["open"],
            gamma_ant["short"],
            temp_cal_iter["open"],
            temp_cal_iter["short"],
            temp_ant["open"],
            temp_ant["short"],
            wterms,
        )

        tunc = tu(fmask)
        tcos = tc(fmask)
        tsin = ts(fmask)

        yield (
            p_sca,
            p_off,
            tu,
            tc,
            ts,
        )


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


def decalibrate_antenna_temperature(
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
