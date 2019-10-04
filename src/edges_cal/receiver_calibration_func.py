# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 14:02:33 2018

@author: Nivedita
"""
import numpy as np
import scipy as sp
import scipy.io as sio


def load_level1_MAT(file_name):
    """
    This function loads the antenna temperature and date/time from MAT files produced by
    the MATLAB function acq2level1.m

    Parameters
    ----------
    file_name: str
        path and name of MAT file
    plot: bool
        flag for plotting spectrum data.

    Returns
    -------
    ds: 2D Spectra array
    dd: Nx6 date/time array

    Examples
    --------
    >>> ds, dd = load_level1_MAT('/file.MAT', plot='yes')
    """
    # loading data and extracting main array
    d = sio.loadmat(file_name)
    if "ta" in d.keys():
        return d["ta"]
    elif "ant_temp" in d.keys():
        return d["ant_temp"]


def temperature_thermistor(resistance, coeffs="oven_industries_TR136_170", kelvin=True):
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


class AverageCal:
    """
    Load and average (in time) calibration data (ambient, hot, open,
    shorted, simulators, etc.) in MAT format.

    It also returns the average physical temperature of the corresponding calibrator,
    measured with an Oven Industries TR136-170 thermistor.

    Parameters
    ----------
    spectrum_files: string, or list of strings,
        Paths and names of spectrum files to process
    resistance_file: string, or list,
        Path and name of resistance file to process
    start_percent: float, optional
        percentage of initial data to dismiss, for both Spectra and resistance

    Returns
    -------
    av_ta: average spectrum at raw frequency resolution, starting at 0 Hz
    av_temp: average physical temperature

    Examples
    --------
    >>> spec_file1 = '/file1.mat'
    >>> spec_file2 = '/file2.mat'
    >>> spec_files = [spec_file1, spec_file2]
    >>> res_file = 'res_file.txt'
    >>> av_ta, av_temp = AverageCal(spec_files, res_file, start_percentage=10)
    """

    def __init__(self, spectrum_files, resistance_file, start_percent=0):
        self.start_percent = start_percent
        self.ave_spectrum = self.read_spectrum(spectrum_files)
        self.thermistor_temp = self.read_thermistor_temp(resistance_file)

    @property
    def temp_ave(self):
        """Average thermistor temperature"""
        return np.mean(self.indexed_temp)

    @property
    def start_index(self):
        """Starting index for useful measurements"""
        return int((self.start_percent / 100) * len(self.thermistor_temp))

    @property
    def indexed_temp(self):
        """A view of the thermistor temperatures beginning at the start index."""
        return self.thermistor_temp[self.start_index :]

    def read_spectrum(self, spectrum_files, start_percent=None):
        start_percent = start_percent or self.start_percent
        for i, fl in enumerate(spectrum_files):
            tai = load_level1_MAT(fl)
            if i == 0:
                ta = tai
            else:
                ta = np.concatenate((ta, tai), axis=1)

        index_start_spectra = int((start_percent / 100) * len(ta[0, :]))
        ta_sel = ta[:, index_start_spectra::]
        av_ta = np.mean(ta_sel, axis=1)
        return av_ta

    @staticmethod
    def read_thermistor_temp(resistance_file):
        if isinstance(resistance_file, list):
            if len(resistance_file) == 0:
                raise ValueError("Empty list of resistance files")

            resistance = np.genfromtxt(resistance_file[0])
            for fl in resistance_file[1:]:
                resistance = np.concatenate((resistance, np.genfromtxt(fl)), axis=0)
        else:
            resistance = np.genfromtxt(resistance_file)

        return temperature_thermistor(resistance)


def frequency_edges(f_low, f_high):
    """
    Return the raw EDGES frequency array, in MHz.

    Parameters
    ----------
    f_low: float
        low-end limit of frequency range, in MHz
    f_high: float
        high-end limit of frequency range, in MHz

    Returns
    -------
    freqs: 1D-array
        full frequency array from 0 to 200 MHz, at raw resolution
    index_flow: int
        index of flow
    index_fhigh: int
        index of fhigh

    Examples
    --------
    >>> freqs, index_flow, index_fhigh = frequency_edges(90, 190)
    """
    # Full frequency vector
    nchannels = 16384 * 2
    max_freq = 200.0
    fstep = max_freq / nchannels
    freqs = np.arange(0, max_freq, fstep)

    # Indices of frequency limits
    if f_low < 0 or f_low >= max(freqs) or f_high < 0 or f_high >= max(freqs):
        raise ValueError("Limits are 0 MHz and " + str(max(freqs)) + " MHz")

    lower_index = np.where(freqs >= f_low)[0] - 1
    upper_index = np.where(freqs >= f_high)[0]

    return freqs, lower_index, upper_index


def NWP_fit(flow, fhigh, f, rl, ro, rs, Toe, Tse, To, Ts, wterms):
    # S11 quantities
    fn = (f - ((fhigh - flow) / 2 + flow)) / ((fhigh - flow) / 2)
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
    A = np.zeros((3 * wterms, 2 * len(fn)))
    for i in range(wterms):
        A[i, :] = np.append(K2o * fn ** i, K2s * fn ** i)
        A[i + 1 * wterms, :] = np.append(K3o * fn ** i, K3s * fn ** i)
        A[i + 2 * wterms, :] = np.append(K4o * fn ** i, K4s * fn ** i)
    b = np.append((Toe - To * K1o), (Tse - Ts * K1s))

    # Transposing matrices so 'frequency' dimension is along columns
    M = A.T
    ydata = np.reshape(b, (-1, 1))
    # print('Test Printouts')
    # print(M) #toubleshooting - D
    # print(ydata)

    # Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
    Q1, R1 = sp.linalg.qr(M, mode="economic")
    param = sp.linalg.solve(R1, np.dot(Q1.T, ydata))

    # Evaluating TU, TC, and TS
    TU = np.zeros(len(fn))
    TC = np.zeros(len(fn))
    TS = np.zeros(len(fn))

    for i in range(wterms):
        TU = TU + param[i, 0] * fn ** i
        TC = TC + param[i + 1 * wterms, 0] * fn ** i
        TS = TS + param[i + 2 * wterms, 0] * fn ** i

    return TU, TC, TS


def get_F(gamma_rec, gamma_ant):
    """Get the F parameter for a given receiver and antenna"""
    return np.sqrt(1 - np.abs(gamma_rec) ** 2) / (1 - gamma_ant * gamma_rec)


def get_alpha(gamma_rec, gamma_ant):
    """Get the alpha parameter for a given receiver and antenna"""
    return np.angle(gamma_ant * get_F(gamma_rec, gamma_ant))


def power_ratio(
    freqs,
    temp_ant,
    gamma_ant,
    gamma_rec,
    c1_poly,
    c2_poly,
    temp_unc_poly,
    temp_cos_poly,
    temp_sin_poly,
    temp_noise_source,
    temp_load,
    ref_freq=75.0,
):
    """
    Compute the ratio of raw powers from the three-position switch.

    Computes (as a model)

    .. math :: Q_P = (P_ant - P_L)/(P_NS - P_L)

    Parameters
    ----------
    freqs : array_like
        Frequencies of the Spectra.
    temp_ant : array_like, shape (NFREQS,)
        Temperature of the antenna, or simulator.
    gamma_ant : array_like, shape (NFREQS,)
        S11 of the antenna (or simulator)
    gamma_rec : array_like, shape (NFREQS,)
        S11 of the receiver.
    c1_poly : :class:`np.poly1d`
        A polynomial representing the C_1 term.
    c2_poly : :class:`np.poly1d`
        A polynomial representing the C_2 term
    temp_unc_poly : :class:`np.poly1d`
        A polynomial representing the uncorrelated noise-wave parameter
    temp_cos_poly : :class:`np.poly1d`
        A polynomial representing the cosine noise-wave parameter
    temp_sin_poly : :class:`np.poly1d`
        A polynomial representing the sine noise-wave parameter
    temp_noise_source : array_like, shape (NFREQS,)
        Temperature of the internal noise source
    temp_load : array_like, shape (NFREQS,)
        Temperature of the internal load
    ref_freq : float, optional
        A reference frequency around which to expand the polynomials.

    Returns
    -------
    array_like : the quantity Q_P as a function of frequency.
    """
    F = get_F(gamma_rec, gamma_ant)
    alpha = get_alpha(gamma_rec, gamma_ant)

    f = freqs / ref_freq

    return (
        temp_ant * (1 - np.abs(gamma_ant) ** 2) * np.abs(F) ** 2
        + temp_unc_poly(f) * np.abs(gamma_ant) ** 2 * np.abs(F) ** 2
        + np.abs(gamma_ant)
        * np.abs(F)
        * (temp_cos_poly(f) * np.cos(alpha) + temp_sin_poly(f) * np.sin(alpha))
        + (c2_poly(f) - temp_load)
    ) / (c1_poly(f) * temp_noise_source * (1 - np.abs(gamma_rec) ** 2))


def calibration_quantities(
    flow,
    fhigh,
    f,
    Tae,
    The,
    Toe,
    Tse,
    rl,
    ra,
    rh,
    ro,
    rs,
    Ta,
    Th,
    To,
    Ts,
    cterms,
    wterms,
    Tamb_internal=300,
):
    # S11 quantities
    Fa = np.sqrt(1 - np.abs(rl) ** 2) / (1 - ra * rl)
    Fh = np.sqrt(1 - np.abs(rl) ** 2) / (1 - rh * rl)

    PHIa = np.angle(ra * Fa)
    PHIh = np.angle(rh * Fh)

    G = 1 - np.abs(rl) ** 2

    K1a = (1 - np.abs(ra) ** 2) * np.abs(Fa) ** 2 / G
    K1h = (1 - np.abs(rh) ** 2) * np.abs(Fh) ** 2 / G

    K2a = (np.abs(ra) ** 2) * (np.abs(Fa) ** 2) / G
    K2h = (np.abs(rh) ** 2) * (np.abs(Fh) ** 2) / G

    K3a = (np.abs(ra) * np.abs(Fa) / G) * np.cos(PHIa)
    K3h = (np.abs(rh) * np.abs(Fh) / G) * np.cos(PHIh)

    K4a = (np.abs(ra) * np.abs(Fa) / G) * np.sin(PHIa)
    K4h = (np.abs(rh) * np.abs(Fh) / G) * np.sin(PHIh)

    # Initializing arrays
    niter = 4
    Ta_iter = np.zeros((niter, len(f)))
    Th_iter = np.zeros((niter, len(f)))

    sca = np.zeros((niter, len(f)))
    off = np.zeros((niter, len(f)))

    Tae_iter = np.zeros((niter, len(f)))
    The_iter = np.zeros((niter, len(f)))
    Toe_iter = np.zeros((niter, len(f)))
    Tse_iter = np.zeros((niter, len(f)))

    TU = np.zeros((niter, len(f)))
    TC = np.zeros((niter, len(f)))
    TS = np.zeros((niter, len(f)))
    fn = (f - ((fhigh - flow) / 2 + flow)) / ((fhigh - flow) / 2)

    # Calibration loop
    for i in range(niter):
        # Step 1: approximate physical temperature
        if i == 0:
            Ta_iter[i, :] = Tae / K1a
            Th_iter[i, :] = The / K1h

        if i > 0:
            NWPa = TU[i - 1, :] * K2a + TC[i - 1, :] * K3a + TS[i - 1, :] * K4a
            NWPh = TU[i - 1, :] * K2h + TC[i - 1, :] * K3h + TS[i - 1, :] * K4h

            Ta_iter[i, :] = (Tae_iter[i - 1, :] - NWPa) / K1a
            Th_iter[i, :] = (The_iter[i - 1, :] - NWPh) / K1h

        # Step 2: scale and offset

        # Updating scale and offset
        sca_new = (Th - Ta) / (Th_iter[i, :] - Ta_iter[i, :])
        off_new = Ta_iter[i, :] - Ta

        if i == 0:
            sca_raw = sca_new
            off_raw = off_new
        if i > 0:
            sca_raw = sca[i - 1, :] * sca_new
            off_raw = off[i - 1, :] + off_new

        # Modeling scale
        p_sca = np.polyfit(fn, sca_raw, cterms - 1)
        m_sca = np.polyval(p_sca, fn)
        sca[i, :] = m_sca

        # Modeling offset
        p_off = np.polyfit(fn, off_raw, cterms - 1)
        m_off = np.polyval(p_off, fn)
        off[i, :] = m_off

        # Step 3: corrected "uncalibrated spectrum" of cable

        Tae_iter[i, :] = (Tae - Tamb_internal) * sca[i, :] + Tamb_internal - off[i, :]
        The_iter[i, :] = (The - Tamb_internal) * sca[i, :] + Tamb_internal - off[i, :]
        Toe_iter[i, :] = (Toe - Tamb_internal) * sca[i, :] + Tamb_internal - off[i, :]
        Tse_iter[i, :] = (Tse - Tamb_internal) * sca[i, :] + Tamb_internal - off[i, :]

        # Step 4: computing NWP
        TU[i, :], TC[i, :], TS[i, :] = NWP_fit(
            flow, fhigh, f, rl, ro, rs, Toe_iter[i, :], Tse_iter[i, :], To, Ts, wterms
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
