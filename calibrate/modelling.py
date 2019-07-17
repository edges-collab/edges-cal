# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 20:48:35 2018

@author: Nivedita
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def model_evaluate(model_type, par, xdata, fr=150, df=10, zr=8, dz=2, z_alpha=0,
                   anastasia_model_number=0, jordan_model_number=0):
    """
    Last modification: May 24, 2015.

    This function evaluates 'polynomial' or 'fourier' models at array 'xdata', using parameters 'par'.
    It is a direct complement to the function 'fit_polynomial_fourier'.
    If P is the total number of parameters, the 'polynomial' model is: model = a0 + a1*xdata + a2*xdata**2 + ... + (aP-1)*xdata**(P-1).
    The 'fourier' model is: model = a0 + (a1*np.cos(1*xdata) + a2*np.sin(1*xdata)) + ... + ((aP-2)*np.cos(((P-1)/2)*xdata) + (aP-1)*np.sin(((P-1)/2)*xdata)).

    Parameters
    ----------
    model_type: str {'polynomial', 'EDGES_polynomial', 'fourier'}
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
    >>> model = model_evaluate('fourier', par_array, fn)
    """

    if model_type == 'polynomial':
        summ = 0
        for i in range(len(par)):
            summ = summ + par[i] * xdata ** i
    elif model_type == 'fourier':
        summ = par[0]

        n_cos_sin = int((len(par) - 1) / 2)
        for i in range(n_cos_sin):
            icos = 2 * i + 1
            isin = 2 * i + 2
            summ = summ + par[icos] * np.cos((i + 1) * xdata) + par[isin] * np.sin((i + 1) * xdata)
    elif model_type == 'EDGES_polynomial':
        summ = 0
        for i in range(len(par)):
            summ = summ + par[i] * xdata ** (-2.5 + i)
    elif model_type.startswith('EDGES_polynomial_plus'):
        summ = 0
        # Here is the difference with the case above. The last parameters is the
        # amplitude of the Gaussian/Tanh.
        for i in range(len(par) - 1):
            summ = summ + par[i] * xdata ** (-2.5 + i)
    elif model_type == 'Physical_model':
        # Physical model from Memo 172
        summ = 0
        basis = np.zeros((5, len(xdata)))
        basis[0, :] = xdata ** (-2.5)
        basis[1, :] = np.log(xdata) * xdata ** (-2.5)
        basis[2, :] = (np.log(xdata)) ** 2 * xdata ** (-2.5)
        basis[3, :] = xdata ** (-4.5)
        basis[4, :] = xdata ** (-2)

        for i in range(len(par)):
            summ = summ + par[i] * basis[i, :]

    elif model_type.startswith("Physical_model_plus"):
        # Physical model from Memo 172
        summ = 0
        basis = np.zeros((5, len(xdata)))
        basis[0, :] = xdata ** (-2.5)
        basis[1, :] = np.log(xdata) * xdata ** (-2.5)
        basis[2, :] = (np.log(xdata)) ** 2 * xdata ** (-2.5)
        basis[3, :] = xdata ** (-4.5)
        basis[4, :] = xdata ** (-2)

        # Here is the difference with the case above. The last parameters is the
        # amplitude of the Gaussian/Tanh.
        for i in range(len(par) - 1):
            summ = summ + par[i] * basis[i, :]
    else:
        summ = 0

    if model_type.endswith('plus_gaussian_frequency'):
        gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_frequency', fr=fr, df=df)
        summ = summ + par[-1] * gaussian_function

    elif model_type.endswith('plus_gaussian_redshift'):
        gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_redshift', zr=zr, dz=dz,
                                              z_alpha=z_alpha)
        summ += par[-1] * gaussian_function

    elif model_type.endswith('plus_tanh'):
        tanh_function, xHI, z = model_eor(xdata, T21=1, zr=zr, dz=dz)
        summ = summ + par[-1] * tanh_function

    elif model_type.endswith('plus_anastasia'):
        # xdata: frequency in MHz, model_in_K: it is in K
        model_in_K = model_eor_anastasia(anastasia_model_number, xdata)
        summ = summ + par[-1] * model_in_K

    elif model_type.endswith('plus_jordan'):
        model_in_K = model_eor_jordan(jordan_model_number, xdata)  # xdata: frequency in MHz, model_in_K: it is in K
        summ = summ + par[-1] * model_in_K

    return summ


def fit_polynomial_fourier(model_type, xdata, ydata, nterms, Weights=1, plot=False,
                           fr=150, df=10, zr=8, dz=2, z_alpha=0,
                           anastasia_model_number=0, jordan_model_number=0):
    """
    This function computes a Least-Squares fit to data using the QR decomposition method.
    Two models are supported: 'polynomial', and 'fourier'.

    If P is the total number of parameters (P = nterms), the 'polynomial' model is::

        ydata = a0 + a1*xdata + a2*xdata**2 + ... + (aP-1)*xdata**(P-1).

    The 'fourier' model is::

        ydata = a0 + (a1*np.cos(1*xdata) + a2*np.sin(1*xdata)) + ... + ((aP-2)*np.cos(((P-1)/2)*xdata) + (aP-1)*np.sin(((P-1)/2)*xdata)).

    Parameters
    ----------
    model_type: 'polynomial', 'EDGES_polynomial', or 'fourier'
    xdata: 1D array of independent measurements, of length N, properly normalized to optimize the fit
    ydata: 1D array of dependent measurements, of length N
    nterms: total number of fit coefficients for baseline
    W: matrix of weights, expressed as the inverse of a covariance matrix. It doesn't have to be normalized to anything in particular. Relative weights are OK.
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
    >>> param, model, rms, cov = fit_polynomial_fourier('fourier', (f_MHz-150)/50, measured_spectrum, 11)
    """

    # initializing "design" matrix
    AT = np.zeros((nterms, len(xdata)))

    # assigning basis functions
    if model_type == 'polynomial':
        for i in range(nterms):
            AT[i, :] = xdata ** i

    if model_type == 'fourier':
        AT[0, :] = np.ones(len(xdata))
        for i in range(int((nterms - 1) / 2)):
            AT[2 * i + 1, :] = np.cos((i + 1) * xdata)
            AT[2 * i + 2, :] = np.sin((i + 1) * xdata)

    if model_type.startswith('EDGES_polynomial'):
        for i in range(nterms):
            AT[i, :] = xdata ** (-2.5 + i)

    # Physical model from Memo 172
    elif model_type.startswith('Physical_model'):
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
            raise ValueError('For the Physical model it has to be 4 or 5 terms.')

    # nterms ONLY includes the number of parameters for the baseline.

    # Gaussian in frequency
    if model_type.endswith('plus_gaussian_frequency'):
        gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_frequency', fr=fr, df=df)
        AT = np.append(AT, gaussian_function.reshape(1, -1), axis=0)

    # Gaussian in redshift
    elif model_type.endswith('gaussian_redshift'):
        gaussian_function, xHI, z = model_eor(xdata, T21=1, model_type='gaussian_redshift', zr=zr, dz=dz,
                                              z_alpha=z_alpha, dz_accuracy_skewed_gaussian=0.0025)
        AT = np.append(AT, gaussian_function.reshape(1, -1), axis=0)

    # Tanh
    elif model_type.endswith("tanh"):
        tanh_function, xHI, z = model_eor(xdata, T21=1, zr=zr, dz=dz)
        AT = np.append(AT, tanh_function.reshape(1, -1), axis=0)

    elif model_type.endswith("anastasia"):
        model_in_K = model_eor_anastasia(anastasia_model_number,
                                         xdata)  # xdata: frequency in MHz, model_in_K: it is in K
        AT = np.append(AT, model_in_K.reshape(1, -1), axis=0)

    elif model_type.endswith("jordan"):
        model_in_K = model_eor_jordan(jordan_model_number, xdata)  # xdata: frequency in MHz, model_in_K: it is in K
        AT = np.append(AT, model_in_K.reshape(1, -1), axis=0)

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
    Q1, R1 = sp.linalg.qr(WA, mode='economic')  # returns
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

    # plotting ?
    if plot:
        fhigh = 180
        flow = 50
        plt.close()
        plt.figure(1, facecolor='w')
        plt.subplot(2, 1, 1)
        plt.plot(xdata * (fhigh - flow) / 2 + (flow + (fhigh - flow) / 2), ydata, 'b')
        plt.plot(xdata * (fhigh - flow) / 2 + (flow + (fhigh - flow) / 2), model, 'r.')
        plt.ylabel(r'$T_{ant} (K)$')
        plt.xlabel('Frequency(MHz)')
        plt.grid()
        plt.ticklabel_format(useOffset=False)

        plt.subplot(2, 1, 2)
        plt.plot(xdata * (fhigh - flow) / 2 + (flow + (fhigh - flow) / 2), ydata - model, 'b')
        plt.ylabel(r'$Residues (K)$')
        plt.xlabel('Frequency(MHz)')
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.show()

    return param, model, wRMS, cov, wMSE  # wMSE = reduced chi square
