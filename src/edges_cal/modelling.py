# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 20:48:35 2018

@author: Nivedita
"""
import functools

import numpy as np
import scipy as sp

_MODELS = {}

F_CENTER = 75.0


def flexible_model(func):
    functools.wraps(func)
    _MODELS[func.__name__] = func
    return func


@flexible_model
def explog(x, *par, f_center=F_CENTER):
    y = x / f_center
    return 2.725 + par[0] * y ** sum(
        [p * np.log(y) ** i for i, p in enumerate(par[1:])]
    )


@flexible_model
def physical5(x, a, b, c, d, e, f_center=F_CENTER):
    return np.log(
        a
        * (x / f_center) ** (-2.5 + b + c * np.log(x / f_center))
        * np.exp(-d * (x / f_center) ** -2)
        + e * (x / f_center) ** -2
    )


@flexible_model
def physicallin5(x, a, b, c, d, e, f_center=F_CENTER):
    return (
        a * (x / f_center) ** -2.5
        + b * (x / f_center) ** -2.5 * np.log(x / f_center)
        + c * (x / f_center) ** -2.5 * (np.log(x / f_center)) ** 2
        + d * (x / f_center) ** -4.5
        + e * (x / f_center) ** -2
    )


@flexible_model
def loglog(x, *par, f_center=F_CENTER):
    """
    Log-Log foreground model, with arbitrary number of parameters

    Parameters
    ----------
    x : array_like
        Frequencies.
    p : iterable
        Co-efficients of the polynomial, from p0 to pn.
    f_center : float
        Central / reference frequency.
    """
    return sum([p * (x / f_center) ** i for i, p in enumerate(par)])


polynomial = loglog


@flexible_model
def linlog(x, *par, f_center=F_CENTER):
    """
    Lin-Log foreground model, with arbitrary number of parameters

    Parameters
    ----------
    x : array_like
        Frequencies.
    p : iterable
        Co-efficients of the polynomial, from p0 to pn.
    f_center : float
        Central / reference frequency.
    """
    return sum([p * np.log(x / f_center) ** i for i, p in enumerate(par)])


@flexible_model
def edges_polynomial(x, *par, f_center=F_CENTER):
    return sum([p * (x / f_center) ** (-2.5 + i) for i, p in enumerate(par)])


def model_evaluate(model, par, xdata, center=False, **kwargs):
    """
    This function evaluates 'polynomial' or 'fourier' models at array 'xdata',
    using parameters 'par'.
    It is a direct complement to the function 'fit_polynomial_fourier'.
    If P is the total number of parameters, the 'polynomial' model is: model = a0 + a1*xdata +
    a2*xdata**2 + ... + (aP-1)*xdata**(P-1).
    The 'fourier' model is: model = a0 + (a1*np.cos(1*xdata) + a2*np.sin(1*xdata)) + ... + ((
    aP-2)*np.cos(((P-1)/2)*xdata) + (aP-1)*np.sin(((P-1)/2)*xdata)).

    Parameters
    ----------
    model: str {'polynomial', 'fourier'}
        Model to evaluate
    par: 1D array
        Parameters, in increasing order, i.e., [a0, a1, ... , aP-1]
    xdata: 1D array
        Independent variable

    Returns
    -------
    1D-array with model

    Examples
    --------
    >>> model = model_evaluate('fourier', [0, 1], np.arange(10))
    >>> [0,1,2,3,4,5,6,7,8,9]
    """
    if type(model) == str:
        model = _MODELS[model]

    if model not in _MODELS.values():
        raise ValueError("the model passed was not a registered flexible_model")

    if not center:
        kwargs["f_center"] = 1

    return model(xdata, *par, **kwargs)


def fit_polynomial_fourier(
    model_type,
    xdata,
    ydata,
    nterms,
    Weights=1,
    plot=False,
    fr=150,
    df=10,
    zr=8,
    dz=2,
    z_alpha=0,
    anastasia_model_number=0,
    jordan_model_number=0,
):
    """
    This function computes a Least-Squares fit to data using the QR decomposition method.
    Two models are supported: 'polynomial', and 'fourier'.

    If P is the total number of parameters (P = nterms), the 'polynomial' model is::

        ydata = a0 + a1*xdata + a2*xdata**2 + ... + (aP-1)*xdata**(P-1).

    The 'fourier' model is::

        ydata = a0 + (a1*np.cos(1*xdata) + a2*np.sin(1*xdata)) + ... + ((aP-2)*np.cos(((
        P-1)/2)*xdata) + (aP-1)*np.sin(((P-1)/2)*xdata)).

    Parameters
    ----------
    model_type: 'polynomial', 'EDGES_polynomial', or 'fourier'
    xdata: 1D array of independent measurements, of length N, properly normalized to optimize the
    fit
    ydata: 1D array of dependent measurements, of length N
    nterms: total number of fit coefficients for baseline
    W: matrix of weights, expressed as the inverse of a covariance matrix. It doesn't have to be
    normalized to anything in particular. Relative weights are OK.
    plot: flag to plot measurements along with fit, and residuals. Use plot='yes' for plotting

    Returns
    -------
    param: 1D-array
        Fit parameters, in increasing order, i.e., [a0, a1, ... , aP-1]
    model: 1D-array of length N,
        Model evaluated at fit parameters
    rms: RMS of residuals
    cov: covariance matrix of fit parameters, organized following 'param' array

    Examples
    --------
    >>> param, model, rms, cov = fit_polynomial_fourier('fourier', (f_MHz-150)/50,
    measured_spectrum, 11)
    """

    # initializing "design" matrix
    AT = np.zeros((nterms, len(xdata)))

    # assigning basis functions
    if model_type == "polynomial":
        for i in range(nterms):
            AT[i, :] = xdata ** i

    if model_type == "fourier":
        AT[0, :] = np.ones(len(xdata))
        for i in range(int((nterms - 1) / 2)):
            AT[2 * i + 1, :] = np.cos((i + 1) * xdata)
            AT[2 * i + 2, :] = np.sin((i + 1) * xdata)

    if model_type.startswith("EDGES_polynomial"):
        for i in range(nterms):
            AT[i, :] = xdata ** (-2.5 + i)

    # Physical model from Memo 172
    elif model_type.startswith("Physical_model"):
        if nterms >= 3:
            AT = np.zeros((nterms, len(xdata)))
            AT[0, :] = xdata ** (-2.5)
            AT[1, :] = np.log(xdata) * xdata ** (-2.5)
            AT[2, :] = (np.log(xdata)) ** 2 * xdata ** (-2.5)

            if nterms >= 4:
                AT[3, :] = xdata ** (-4.5)
                if nterms == 5:
                    AT[4, :] = xdata ** (-2)
        else:
            raise ValueError("For the Physical model it has to be 4 or 5 terms.")

    # Applying General Least Squares Formalism, and Solving using QR decomposition
    # See: http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/2804/pdf/imm2804.pdf

    # if no weights are given
    if np.isscalar(Weights):
        W = np.eye(len(xdata))

    # if a vector is given
    elif np.ndim(Weights) == 1:
        W = np.diag(Weights)

    # if a matrix is given
    elif np.ndim(Weights) == 2:
        W = Weights

    # sqrt of weight matrix
    sqrtW = np.sqrt(W)

    # transposing matrices so 'frequency' dimension is along columns
    A = AT.T
    ydata = np.reshape(ydata, (-1, 1))

    # A and ydata "tilde"
    WA = np.dot(sqrtW, A)
    Wydata = np.dot(sqrtW, ydata)

    # solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
    Q1, R1 = sp.linalg.qr(WA, mode="economic")  # returns
    param = sp.linalg.solve(R1, np.dot(Q1.T, Wydata))

    model = np.dot(A, param)
    error = ydata - model
    DF = len(xdata) - len(param) - 1
    wMSE = (1 / DF) * np.dot(error.T, np.dot(W, error))
    wRMS = np.sqrt(np.dot(error.T, np.dot(W, error)) / np.sum(np.diag(W)))
    inv_pre_cov = np.linalg.inv(np.dot(R1.T, R1))
    cov = wMSE * inv_pre_cov

    # back to input format
    ydata = ydata.flatten()
    model = model.flatten()
    param = param.flatten()

    return param, model, wRMS, cov, wMSE  # wMSE = reduced chi square
