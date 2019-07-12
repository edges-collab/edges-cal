# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 14:02:33 2018

@author: Nivedita
"""

import numpy as np
import scipy as sp
import time  as tt 
import datetime as dt
import multiprocessing as mp
import reflection_coefficient as rc
import numpy.linalg as nla

import scipy.io as sio
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.special as scs

import matplotlib.pyplot as plt

import astropy.units as apu
import astropy.time as apt
#import astropy.coordinates as apc

#import ephem as eph	# to install, at the bash terminal type $ conda install ephem
#import emcee as ec
#import healpy as hp
import pickle

from os import listdir, makedirs, system
from os.path import exists
from os.path import expanduser
from astropy.io import fits
from scipy import stats

import h5py
from modelling import *

def level1_MAT(file_name, plot='no'):
	"""
	Last modification: May 24, 2015.

	This function loads the antenna temperature and date/time from MAT files produced by the MATLAB function acq2level1.m

	Definition:
	ds, dd = level1_MAT(file_name, plot='no')

	Input parameters:
	file_name: path and name of MAT file
	plot: flag for plotting spectrum data. Use plot='yes' for plotting

	Output parameters:
	ds: 2D spectra array
	dd: Nx6 date/time array

	Usage:
	ds, dd = level1_MAT('/file.MAT', plot='yes')
	"""


	# loading data and extracting main array
	d = sio.loadmat(file_name)
	darray = d['ta']

	# extracting spectra and date/time
	ds = darray
	#dd = darray[0,1]

	# plotting ?
	if plot == 'yes':
		plt.imshow(ds, aspect = 'auto', vmin = 0, vmax = 2000)
		plt.xlabel('frequency channels')
		plt.ylabel('trace')
		plt.colorbar()
		plt.show()

	return ds
 
def temperature_thermistor_oven_industries_TR136_170(R, unit):

	# Steinhart-Hart coefficients
	a1 = 1.03514e-3
	a2 = 2.33825e-4
	a3 = 7.92467e-8


	# TK in Kelvin
	TK = 1/(a1 + a2*np.log(R) + a3*(np.log(R))**3)

	# Kelvin or Celsius
	if unit == 'K':
		T = TK
	if unit == 'C':
		T = TK - 273.15

	return T

def average_calibration_spectrum(spectrum_files, resistance_file, start_percent=0, plot='no'):
    """
    Last modification: May 24, 2015.

    This function loads and averages (in time) calibration data (ambient, hot, open, shorted, simulators, etc.) in MAT format produced by the "acq2level1.m" MATLAB program. It also returns the average physical temperature of the corresponding calibrator, measured with an Oven Industries TR136-170 thermistor.

    Definition:
    av_ta, av_temp = average_calibration_spectrum(spectrum_files, resistance_file, start_percentage=0, plot='no')

    Input parameters:
    spectrum_files: string, or list of strings, with the paths and names of spectrum files to process
    resistance_file: string, or list, with the path and name of resistance file to process
    start_percent: percentage of initial data to dismiss, for both, spectra and resistance
    plot: flag for plotting representative data cuts. Use plot='yes' for plotting

    Output parameters:
    av_ta: average spectrum at raw frequency resolution, starting at 0 Hz
    av_temp: average physical temperature

    Usage:
    spec_file1 = '/file1.mat'
    spec_file2 = '/file2.mat'
    spec_files = [spec_file1, spec_file2]
    res_file = 'res_file.txt'
    av_ta, av_temp = average_calibration_spectrum(spec_files, res_file, start_percentage=10, plot='yes')
    """



    # spectra
    for i in range(len(spectrum_files)):
        tai = level1_MAT(spectrum_files[i], plot='no')
        if i == 0:
            ta = tai
        elif i > 0:
            ta = np.concatenate((ta, tai), axis=1)

    index_start_spectra = int((start_percent/100)*len(ta[0,:]))
    ta_sel = ta[:,index_start_spectra::]
    av_ta = np.mean(ta_sel, axis=1)



    # temperature
    if isinstance(resistance_file, list):
        for i in range(len(resistance_file)):
            if i == 0:
                R = np.genfromtxt(resistance_file[i])
            else:	
                R = np.concatenate((R, np.genfromtxt(resistance_file[i])), axis=0)
    else:
        R = np.genfromtxt(resistance_file)


    temp = temperature_thermistor_oven_industries_TR136_170(R, 'K')
    index_start_temp = int((start_percent/100)*len(temp))
    temp_sel = temp[index_start_temp::]
    av_temp = np.average(temp_sel)
     



    # plot
    if plot == 'yes':
        plt.close()
        plt.subplot(2,2,1)
        plt.plot(ta[:,3e4],'r')
        plt.plot([index_start_spectra, index_start_spectra],[min(ta[:,3e4])-5, max(ta[:,3e4])+5], 'k--')
        plt.ylabel('spectral temperature')
        plt.ylim([min(ta[:,3e4])-5, max(ta[:,3e4])+5])
  
            

        plt.subplot(2,2,2)
        plt.plot(ta_sel[:,3e4],'r')
        plt.ylim([min(ta[:,3e4])-5, max(ta[:,3e4])+5])

        plt.subplot(2,2,3)
        plt.plot(temp,'r')
        plt.plot([index_start_temp, index_start_temp],[min(temp)-5, max(temp)+5], 'k--')
        plt.xlabel('sample')
        plt.ylabel('physical temperature')
        plt.ylim([min(temp)-5, max(temp)+5])

        plt.subplot(2,2,4)
        plt.plot(temp_sel,'r')
        plt.xlabel('sample')
        plt.ylim([min(temp)-5, max(temp)+5])

    return av_ta, av_temp,temp,temp_sel
 

def frequency_edges(flow, fhigh):
	"""
	Last modification: May 24, 2015.

	This function returns the raw EDGES frequency array, in MHz.

	Definition:
	freqs, index_flow, index_fhigh = frequency_edges(flow, fhigh)

	Input parameters:
	flow: low-end limit of frequency range, in MHz
	fhigh: high-end limit of frequency range, in MHz

	Output parameters:
	freqs: full frequency array from 0 to 200 MHz, at raw resolution
	index_flow: index of flow
	index_fhigh: index of fhigh

	Usage:
	freqs, index_flow, index_fhigh = frequency_edges(90, 190)
	"""

	# Full frequency vector
	nchannels = 16384*2
	max_freq = 200.0
	fstep = max_freq/nchannels
	freqs = np.arange(0, max_freq, fstep)

	# Indices of frequency limits
	if (flow < 0) or (flow >= max(freqs)) or (fhigh < 0) or (fhigh >= max(freqs)):
		print('ERROR. Limits are 0 MHz and ' + str(max(freqs)) + ' MHz')
	else:
		for i in range(len(freqs)-1):
			if (freqs[i] <= flow) and (freqs[i+1] >= flow):
				index_flow = i
			if (freqs[i] <= fhigh) and (freqs[i+1] >= fhigh):
				index_fhigh = i

		return freqs, index_flow, index_fhigh
  
  
'''
to obtain Noise wave parameters TU, TC, TS
ro -  reflection coefficient of ?
rs -  reflection coefficient of?
'''
def NWP_fit(flow,fhigh, f, rl, ro, rs, Toe, Tse, To, Ts, wterms):
    # S11 quantities
    fn=(f - ((fhigh-flow)/2 + flow))/((fhigh-flow)/2)
    Fo = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - ro*rl ) 
    Fs = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rs*rl )
    PHIo = np.angle( ro*Fo )
    PHIs = np.angle( rs*Fs )
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
    A = np.zeros((3 * wterms, 2*len(fn)))
    for i in range(wterms):
        A[i, :] = np.append(K2o * fn ** i, K2s * fn ** i)
        A[i + 1 * wterms, :] = np.append(K3o * fn ** i, K3s * fn ** i)
        A[i + 2 * wterms, :] = np.append(K4o * fn ** i, K4s * fn ** i)
    b = np.append( (Toe - To*K1o), (Tse - Ts*K1s) )

	# Transposing matrices so 'frequency' dimension is along columns
    M = A.T
    ydata = np.reshape(b, (-1,1))
    #print('Test Printouts')
    #print(M) #toubleshooting - D
    #print(ydata)

	# Solving system using 'short' QR decomposition (see R. Butt, Num. Anal. Using MATLAB)
    Q1, R1 = sp.linalg.qr(M, mode='economic')
    param  = sp.linalg.solve(R1, np.dot(Q1.T, ydata))	



	# Evaluating TU, TC, and TS
    TU = np.zeros(len(fn))
    TC = np.zeros(len(fn))
    TS = np.zeros(len(fn))

    for i in range(wterms):
        TU = TU + param[i, 0] * fn ** i
        TC = TC + param[i+1*wterms, 0] * fn ** i
        TS = TS + param[i+2*wterms, 0] * fn ** i



	# Parameters
    #print('length',len(param)) #Troubleshooting - D
    #print('intlength',int(len(param)/3))
    #print()
    #print(param.shape)
    #print(param[0: (int(len(param)/3)),0])
    
    pU = param[0:(int(len(param) / 3)), 0].T
    pC = param[(int(len(param)/3)):(int(2 * len(param) / 3)), 0].T
    pS = param[(int(2 * len(param)/3)):(int(len(param))), 0].T



    return TU, TC, TS 
'''
uses the calibration load data in equation(7)
ra,rh - ref coeff of ambienta nd hot loads resp
Tae , The, Toe,Tse - uncalibrated temperature spectra of the ambient & hot loads, open & short cables
Ta, Th, To, Ts - physical temperature of the 2 loads & 2 cables

'''
def calibration_quantities(flow,fhigh, f, Tae, The, Toe, Tse, rl, ra, rh, ro, rs, Ta, Th, To, Ts, Tamb_internal, cterms, wterms):



	# S11 quantities 
    Fa = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - ra*rl ) 
    Fh = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rh*rl )

    PHIa = np.angle( ra*Fa )
    PHIh = np.angle( rh*Fh )

    G = 1 - np.abs(rl) ** 2

    K1a = (1 - np.abs(ra) **2) * np.abs(Fa) ** 2 / G
    K1h = (1 - np.abs(rh) **2) * np.abs(Fh) ** 2 / G

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
    fn = (f - ((fhigh-flow)/2 + flow))/((fhigh-flow)/2)

	# Calibration loop
    for i in range(niter):

        #print(i)

		# Step 1: approximate physical temperature
        if i == 0:
            Ta_iter[i,:] = Tae / K1a
            Th_iter[i,:] = The / K1h

        if i > 0:		
            NWPa = TU[i-1,:]*K2a + TC[i-1,:]*K3a + TS[i-1,:]*K4a
            NWPh = TU[i-1,:]*K2h + TC[i-1,:]*K3h + TS[i-1,:]*K4h			

            Ta_iter[i,:] = (Tae_iter[i-1,:] - NWPa) / K1a
            Th_iter[i,:] = (The_iter[i-1,:] - NWPh) / K1h	


        # Step 2: scale and offset

        # Updating scale and offset
        sca_new  = (Th - Ta) / (Th_iter[i,:] - Ta_iter[i,:])
        off_new  = Ta_iter[i,:] - Ta

        if i == 0:
            sca_raw = sca_new
            off_raw = off_new
        if i > 0:
            sca_raw = sca[i-1,:] * sca_new
            off_raw = off[i-1,:] + off_new

        # Modeling scale
        p_sca    = np.polyfit(fn, sca_raw, cterms-1)
        m_sca    = np.polyval(p_sca, fn)
        sca[i,:] = m_sca

        # Modeling offset
        p_off    = np.polyfit(fn, off_raw, cterms-1)
        m_off    = np.polyval(p_off, fn)		
        off[i,:] = m_off




        # Step 3: corrected "uncalibrated spectrum" of cable
        Tamb_internal = 300  # same as used for 3-pos switch computation. BUT RESULTS DON'T CHANGE IF ANOTHER VALUE IS USED

        Tae_iter[i,:] = (Tae - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]
        The_iter[i,:] = (The - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]
        Toe_iter[i,:] = (Toe - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]
        Tse_iter[i,:] = (Tse - Tamb_internal) * sca[i,:] + Tamb_internal - off[i,:]



        # Step 4: computing NWP
        TU[i,:], TC[i,:], TS[i,:] = NWP_fit(flow,fhigh, f, rl, ro, rs, Toe_iter[i,:], Tse_iter[i,:], To, Ts, wterms)

    return sca[-1,:], off[-1,:], TU[-1,:], TC[-1,:], TS[-1,:]
 
'''
Function for equation (7)
rd - refelection coefficient of the load
rl - reflection coefficient of the receiver
Td - temperature of the device under test
TU ,Tc,Ts - noise wave parameters
Tamb_internal - noise temperature of the load
''' 

def calibrated_antenna_temperature(Tde, rd, rl, sca, off, TU, TC, TS, Tamb_internal=300):

    # S11 quantities
    Fd = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rd*rl )	
    PHId = np.angle( rd*Fd )
    G = 1 - np.abs(rl) ** 2
    K1d = (1 - np.abs(rd) **2) * np.abs(Fd) ** 2 / G
    K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
    K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
    K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)


    # Applying scale and offset to raw spectrum
    Tde_corrected = (Tde - Tamb_internal)*sca + Tamb_internal - off

    # Noise wave contribution
    NWPd = TU*K2d + TC*K3d + TS*K4d

    # Antenna temperature
    Td = (Tde_corrected - NWPd) / K1d

    return Td
 
 
 
'''
Function for equation (7)
rd - refelection coefficient of the load
rl - reflection coefficient of the receiver
Td - temperature of the device under test
TU ,Tc,Ts - noise wave parameters
Tamb_internal - noise temperature of the load
''' 
def uncalibrated_antenna_temperature(Td, rd, rl, sca, off, TU, TC, TS, Tamb_internal=300):

    # S11 quantities
    Fd = np.sqrt( 1 - np.abs(rl) ** 2 ) / ( 1 - rd*rl )
    PHId = np.angle( rd*Fd )
    G = 1 - np.abs(rl) ** 2
    K1d = (1 - np.abs(rd) **2) * np.abs(Fd) ** 2 / G
    K2d = (np.abs(rd) ** 2) * (np.abs(Fd) ** 2) / G
    K3d = (np.abs(rd) * np.abs(Fd) / G) * np.cos(PHId)
    K4d = (np.abs(rd) * np.abs(Fd) / G) * np.sin(PHId)	


    # Noise wave contribution
    NWPd = TU*K2d + TC*K3d + TS*K4d	

    # Scaled and offset spectrum 
    Tde_corrected = Td*K1d + NWPd

    # Removing scale and offset
    Tde = Tamb_internal + (Tde_corrected - Tamb_internal + off) / sca

    return Tde
 

def models_calibration_physical_temperature(band, calibration_temperature, f, s_parameters=np.zeros(1), MC_temp=np.zeros(6)):

    # Choosing the band
    if band == 'high_band_2015':

        # Paths
        path_root    = home_folder + ''

        if calibration_temperature == 25:
            path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_march_2015/LNA/models/25degC/temp/'
            path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_march_2015/LNA/models/25degC/s11/'

        if calibration_temperature == 35:
            path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_march_2015/LNA/models/35degC/temp/'
            path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_march_2015/LNA/models/35degC/s11/'



        # Normalized frequency
        fn = (f - 140)/60			


    if band == 'high_band_2017':

        # Paths
        path_root    = home_folder + ''

        if calibration_temperature == 25:
            path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_high_band_2017_january/LNA/models/25degC/temp/'
            path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_high_band_2017_january/LNA/models/25degC/s11/'

        # Normalized frequency
        fn = (f - 140)/60	


    if band == 'low_band_2015':

        # Paths
        path_root    = home_folder + ''
        path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_august_2015/LNA/models/25degC/temp/'
        path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_august_2015/LNA/models/25degC/s11/'	

        # Normalized frequency
        fn = (f - 75)/25



    if band == 'low_band_2016':

        if calibration_temperature == 25:
            ## Paths
            #path_root    = home_folder + ''
            #path_temp    = path_root + '/DATA/EDGES/receiver_calibration/calibration_june_2016/LNA/models/25degC/temp/'
            #path_s11     = path_root + '/DATA/EDGES/receiver_calibration/calibration_june_2016/LNA/models/25degC/s11/'

            # Paths
            path_root    = home_folder + ''
            path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_september_2016/LNA/models/25degC/temp/'
            path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_september_2016/LNA/models/25degC/s11/'

        if calibration_temperature == 15:

            # Paths
            path_root    = home_folder + ''
            path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_15C_november_2016/LNA/models/15degC/temp/'
            path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_15C_november_2016/LNA/models/15degC/s11/'


        if calibration_temperature == 35:

            # Paths
            path_root    = home_folder + ''
            path_temp    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_35C_november_2016/LNA/models/35degC/temp/'
            path_s11     = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_35C_november_2016/LNA/models/35degC/s11/'






        # Normalized frequency
        fn = (f - 75)/25




    # Physical temperatures
    phys_temp = np.genfromtxt(path_temp + 'physical_temperatures.txt')
    Ta  = phys_temp[0] * np.ones(len(fn))
    Th  = phys_temp[1] * np.ones(len(fn))
    To  = phys_temp[2] * np.ones(len(fn))
    Ts  = phys_temp[3] * np.ones(len(fn))
    Ts1 = phys_temp[4] * np.ones(len(fn))
    Ts2 = phys_temp[5] * np.ones(len(fn))



	# MC realizations of physical temperatures
    STD_temp = 0.1
    if MC_temp[0] > 0:
        Ta  = (Ta  + MC_temp[0] * STD_temp * np.random.normal(0,1)) 

    if MC_temp[1] > 0:	
        Th  = (Th  + MC_temp[1] * STD_temp * np.random.normal(0,1)) #* np.ones(len(fn))	

    if MC_temp[2] > 0:	
        To  = (To  + MC_temp[2] * STD_temp * np.random.normal(0,1)) #* np.ones(len(fn))

    if MC_temp[3] > 0:
        Ts  = (Ts  + MC_temp[3] * STD_temp * np.random.normal(0,1)) #* np.ones(len(fn))

    if MC_temp[4] > 0:
        Ts1 = (Ts1 + MC_temp[4] * STD_temp * np.random.normal(0,1)) #* np.ones(len(fn))

    if MC_temp[5] > 0:
        Ts2 = (Ts2 + MC_temp[5] * STD_temp * np.random.normal(0,1)) #* np.ones(len(fn))




    # S-parameters of hot load device
    if len(s_parameters) == 1:
        out = models_calibration_s11(band, calibration_temperature, f)
        rh        = out[2]
        s11_sr    = out[5]
        s12s21_sr = out[6]
        s22_sr    = out[7]

    if len(s_parameters) == 4:
        rh        = s_parameters[0]
        s11_sr    = s_parameters[1]
        s12s21_sr = s_parameters[2]
        s22_sr    = s_parameters[3]



    # Temperature of hot device

    # reflection coefficient of termination
    rht = rc.gamma_de_embed(s11_sr, s12s21_sr, s22_sr, rh)

	# inverting the direction of the s-parameters,
	# since the port labels have to be inverted to match those of Pozar eqn 10.25
    s11_sr_rev = s22_sr
    s22_sr_rev = s11_sr

    # absolute value of S_21
    abs_s21 = np.sqrt(np.abs(s12s21_sr))

    # available power gain
    G = ( abs_s21**2 ) * ( 1-np.abs(rht)**2 ) / ( (np.abs(1-s11_sr_rev*rht))**2 * (1-(np.abs(rh))**2) )

    # temperature
    Thd  = G*Th + (1-G)*Ta



    # Output
    output = np.array([Ta, Thd, To, Ts, Ts1, Ts2])

    return output
 
def models_calibration_s11(band, calibration_temperature, fe, receiver_reflection='actual', MC_s11_noise=np.zeros(20), MC_s11_syst=np.zeros(20), systematic_s11='uncorrelated', plots='no', plot_flag=''):


	
    if band == 'low_band_2015':

        # Paths
        path_root       = home_folder + ''
        path_s11        = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_august_2015/LNA/S11/corrected/'			
        path_par_s11    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_august_2015/LNA/models/25degC/s11/'
        path_plot_save  = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_august_2015/LNA/models/25degC/plots/'

        # Data
        calfile = 's11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt'
        s11             = np.genfromtxt(path_s11 + calfile)
        print(calfile)
        fs11            = s11[:, 0]
        fs11n           = (fs11-75)/25
        fen             = (fe-75)/25

        # Loading raw data				
        s11_LNA_raw       = s11[:, 1]  + 1j*s11[:, 2]
        s11_amb_raw       = s11[:, 3]  + 1j*s11[:, 4]
        s11_hot_raw       = s11[:, 5]  + 1j*s11[:, 6]
        s11_open_raw      = s11[:, 7]  + 1j*s11[:, 8]
        s11_shorted_raw   = s11[:, 9]  + 1j*s11[:, 10]
        s11_sr_raw        = s11[:, 11] + 1j*s11[:, 12]
        s12s21_sr_raw     = s11[:, 13] + 1j*s11[:, 14]
        s22_sr_raw        = s11[:, 15] + 1j*s11[:, 16]
        s11_simu1_raw     = s11[:, 17] + 1j*s11[:, 18]
        s11_simu2_raw     = s11[:, 19] + 1j*s11[:, 20]





    if band == 'low_band_2016':



        if calibration_temperature == 25:

            ## Paths
            #path_root       = home_folder + ''
            #path_s11        = path_root + '/DATA/EDGES/receiver_calibration/calibration_june_2016/LNA/S11/corrected/'			
            #path_par_s11    = path_root + '/DATA/EDGES/receiver_calibration/calibration_june_2016/LNA/models/25degC/s11/'
            #path_plot_save  = path_root + '/DATA/EDGES/receiver_calibration/calibration_june_2016/LNA/models/25degC/plots/'

            ## Data
            #calfile = 's11_calibration_low_band_LNA25degC_2016-08-25-20-17-13_25Credo_best.txt'			


            # Paths
            path_root       = home_folder + ''
            path_s11        = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_september_2016/LNA/S11/corrected/'			
            path_par_s11    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_september_2016/LNA/models/25degC/s11/'
            path_plot_save  = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_september_2016/LNA/models/25degC/plots/'

            # Data (both files produce same results)
            calfile = 's11_calibration_low_band_LNA25degC_2016-10-31-13-02-58.txt' 
            #calfile = 's11_calibration_low_band_LNA25degC_2016-10-31-17-50-22.txt'



        if calibration_temperature == 15:

            # Paths
            path_root       = home_folder + ''
            path_s11        = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_15C_november_2016/LNA/S11/corrected/'			
            path_par_s11    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_15C_november_2016/LNA/models/15degC/s11/'
            path_plot_save  = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_15C_november_2016/LNA/models/15degC/plots/'

            # Data
            calfile = 's11_calibration_low_band_LNA15degC_2016-12-14-23-01-53.txt'



        if calibration_temperature == 35:

            # Paths
            path_root       = home_folder + ''
            path_s11        = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_35C_november_2016/LNA/S11/corrected/'			
            path_par_s11    = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_35C_november_2016/LNA/models/35degC/s11/'
            path_plot_save  = path_root + '/DATA/EDGES/calibration/receiver_calibration/calibration_35C_november_2016/LNA/models/35degC/plots/'

            # Data
            calfile = 's11_calibration_low_band_LNA35degC_2016-12-14-23-35-37.txt'








        s11             = np.genfromtxt(path_s11 + calfile)
        print(calfile)
        fs11            = s11[:, 0]
        fs11n           = (fs11-75)/25
        fen             = (fe-75)/25

        # Loading raw data				
        s11_LNA_raw       = s11[:, 1]  + 1j*s11[:, 2]
        s11_amb_raw       = s11[:, 3]  + 1j*s11[:, 4]
        s11_hot_raw       = s11[:, 5]  + 1j*s11[:, 6]
        s11_open_raw      = s11[:, 7]  + 1j*s11[:, 8]
        s11_shorted_raw   = s11[:, 9]  + 1j*s11[:, 10]
        s11_sr_raw        = s11[:, 11] + 1j*s11[:, 12]
        s12s21_sr_raw     = s11[:, 13] + 1j*s11[:, 14]
        s22_sr_raw        = s11[:, 15] + 1j*s11[:, 16]
        s11_simu1_raw     = s11[:, 17] + 1j*s11[:, 18]
        s11_simu2_raw     = s11[:, 19] + 1j*s11[:, 20]
        #s11_simu2_raw     = s11[:, 21] + 1j*s11[:, 22]







    # LNA
    s11_LNA_mag_raw     = np.abs(s11_LNA_raw)
    s11_LNA_ang_raw     = np.unwrap(np.angle(s11_LNA_raw))

    # Ambient
    s11_amb_mag_raw     = np.abs(s11_amb_raw)
    s11_amb_ang_raw     = np.unwrap(np.angle(s11_amb_raw))

    # Hot
    s11_hot_mag_raw     = np.abs(s11_hot_raw)
    s11_hot_ang_raw     = np.unwrap(np.angle(s11_hot_raw))

    # Open
    s11_open_mag_raw    = np.abs(s11_open_raw)
    s11_open_ang_raw    = np.unwrap(np.angle(s11_open_raw))

    # Shorted
    s11_shorted_mag_raw = np.abs(s11_shorted_raw)
    s11_shorted_ang_raw = np.unwrap(np.angle(s11_shorted_raw))

    # sr-s11
    s11_sr_mag_raw      = np.abs(s11_sr_raw)
    s11_sr_ang_raw      = np.unwrap(np.angle(s11_sr_raw))

    # sr-s12s21
    s12s21_sr_mag_raw   = np.abs(s12s21_sr_raw)
    s12s21_sr_ang_raw   = np.unwrap(np.angle(s12s21_sr_raw))

    # sr-s22
    s22_sr_mag_raw      = np.abs(s22_sr_raw)
    s22_sr_ang_raw      = np.unwrap(np.angle(s22_sr_raw))

    # Simu1
    s11_simu1_mag_raw   = np.abs(s11_simu1_raw)
    s11_simu1_ang_raw   = np.unwrap(np.angle(s11_simu1_raw))

    # Simu2
    s11_simu2_mag_raw   = np.abs(s11_simu2_raw)
    s11_simu2_ang_raw   = np.unwrap(np.angle(s11_simu2_raw))		




    # Loading S11 parameters
    par_s11_LNA_mag     = np.genfromtxt(path_par_s11 + 'par_s11_LNA_mag.txt')
    par_s11_LNA_ang     = np.genfromtxt(path_par_s11 + 'par_s11_LNA_ang.txt')	
    par_s11_amb_mag     = np.genfromtxt(path_par_s11 + 'par_s11_amb_mag.txt')
    par_s11_amb_ang     = np.genfromtxt(path_par_s11 + 'par_s11_amb_ang.txt')
    par_s11_hot_mag     = np.genfromtxt(path_par_s11 + 'par_s11_hot_mag.txt')
    par_s11_hot_ang     = np.genfromtxt(path_par_s11 + 'par_s11_hot_ang.txt')
    par_s11_open_mag    = np.genfromtxt(path_par_s11 + 'par_s11_open_mag.txt')
    par_s11_open_ang    = np.genfromtxt(path_par_s11 + 'par_s11_open_ang.txt')
    par_s11_shorted_mag = np.genfromtxt(path_par_s11 + 'par_s11_shorted_mag.txt')
    par_s11_shorted_ang = np.genfromtxt(path_par_s11 + 'par_s11_shorted_ang.txt')

    par_s11_sr_mag      = np.genfromtxt(path_par_s11 + 'par_s11_sr_mag.txt')
    par_s11_sr_ang      = np.genfromtxt(path_par_s11 + 'par_s11_sr_ang.txt')
    par_s12s21_sr_mag   = np.genfromtxt(path_par_s11 + 'par_s12s21_sr_mag.txt')
    par_s12s21_sr_ang   = np.genfromtxt(path_par_s11 + 'par_s12s21_sr_ang.txt')
    par_s22_sr_mag      = np.genfromtxt(path_par_s11 + 'par_s22_sr_mag.txt')
    par_s22_sr_ang      = np.genfromtxt(path_par_s11 + 'par_s22_sr_ang.txt')

    par_s11_simu1_mag   = np.genfromtxt(path_par_s11 + 'par_s11_simu1_mag.txt')
    par_s11_simu1_ang   = np.genfromtxt(path_par_s11 + 'par_s11_simu1_ang.txt')
    par_s11_simu2_mag   = np.genfromtxt(path_par_s11 + 'par_s11_simu2_mag.txt')
    par_s11_simu2_ang   = np.genfromtxt(path_par_s11 + 'par_s11_simu2_ang.txt')
	
    #par_s11_simu2_mag   = np.genfromtxt(path_par_s11 + 'par_s11_simu2r2_mag.txt')
    #par_s11_simu2_ang   = np.genfromtxt(path_par_s11 + 'par_s11_simu2r2_ang.txt')	


    # Load noise RMS
    RMS_s11             = np.genfromtxt(path_par_s11 + 'RMS_s11.txt')
    RMS_s11_LNA_mag     = RMS_s11[0]
    RMS_s11_LNA_ang     = RMS_s11[1]
    RMS_s11_amb_mag     = RMS_s11[2]
    RMS_s11_amb_ang     = RMS_s11[3]
    RMS_s11_hot_mag     = RMS_s11[4]
    RMS_s11_hot_ang     = RMS_s11[5]
    RMS_s11_open_mag    = RMS_s11[6]
    RMS_s11_open_ang    = RMS_s11[7]
    RMS_s11_shorted_mag = RMS_s11[8]
    RMS_s11_shorted_ang = RMS_s11[9]
    RMS_s11_sr_mag      = RMS_s11[10]
    RMS_s11_sr_ang      = RMS_s11[11]
    RMS_s12s21_sr_mag   = RMS_s11[12]
    RMS_s12s21_sr_ang   = RMS_s11[13]
    RMS_s22_sr_mag      = RMS_s11[14]
    RMS_s22_sr_ang      = RMS_s11[15]
    RMS_s11_simu1_mag   = RMS_s11[16]
    RMS_s11_simu1_ang   = RMS_s11[17]
    RMS_s11_simu2_mag   = RMS_s11[18]
    RMS_s11_simu2_ang   = RMS_s11[19]		








	

    if (band == 'low_band_2015') or (band == 'low_band_2016'):

        # Evaluating S11 models at raw frequency
        s11_LNA_mag_raw_model     = model_evaluate('polynomial', par_s11_LNA_mag,     fs11n)
        s11_LNA_ang_raw_model     = model_evaluate('polynomial', par_s11_LNA_ang,     fs11n)

        s11_amb_mag_raw_model     = model_evaluate('fourier',    par_s11_amb_mag,     fs11n)
        s11_amb_ang_raw_model     = model_evaluate('fourier',    par_s11_amb_ang,     fs11n)
        s11_hot_mag_raw_model     = model_evaluate('fourier',    par_s11_hot_mag,     fs11n)
        s11_hot_ang_raw_model     = model_evaluate('fourier',    par_s11_hot_ang,     fs11n)

        s11_open_mag_raw_model    = model_evaluate('fourier',    par_s11_open_mag,    fs11n)
        s11_open_ang_raw_model    = model_evaluate('fourier',    par_s11_open_ang,    fs11n)
        s11_shorted_mag_raw_model = model_evaluate('fourier',    par_s11_shorted_mag, fs11n)
        s11_shorted_ang_raw_model = model_evaluate('fourier',    par_s11_shorted_ang, fs11n)

        s11_sr_mag_raw_model      = model_evaluate('polynomial', par_s11_sr_mag,      fs11n)
        s11_sr_ang_raw_model      = model_evaluate('polynomial', par_s11_sr_ang,      fs11n)
        s12s21_sr_mag_raw_model   = model_evaluate('polynomial', par_s12s21_sr_mag,   fs11n)
        s12s21_sr_ang_raw_model   = model_evaluate('polynomial', par_s12s21_sr_ang,   fs11n)
        s22_sr_mag_raw_model      = model_evaluate('polynomial', par_s22_sr_mag,      fs11n)
        s22_sr_ang_raw_model      = model_evaluate('polynomial', par_s22_sr_ang,      fs11n)
        
        s11_simu1_mag_raw_model   = model_evaluate('polynomial', par_s11_simu1_mag,   fs11n)
        s11_simu1_ang_raw_model   = model_evaluate('polynomial', par_s11_simu1_ang,   fs11n)
        s11_simu2_mag_raw_model   = model_evaluate('polynomial', par_s11_simu2_mag,   fs11n)
        s11_simu2_ang_raw_model   = model_evaluate('polynomial', par_s11_simu2_ang,   fs11n)	


        # Evaluating S11 models at EDGES frequency
        s11_LNA_mag     = model_evaluate('polynomial', par_s11_LNA_mag,     fen)
        s11_LNA_ang     = model_evaluate('polynomial', par_s11_LNA_ang,     fen)

        s11_amb_mag     = model_evaluate('fourier', par_s11_amb_mag,     fen)
        s11_amb_ang     = model_evaluate('fourier', par_s11_amb_ang,     fen)
        s11_hot_mag     = model_evaluate('fourier', par_s11_hot_mag,     fen)
        s11_hot_ang     = model_evaluate('fourier', par_s11_hot_ang,     fen)

        s11_open_mag    = model_evaluate('fourier',    par_s11_open_mag,    fen)
        s11_open_ang    = model_evaluate('fourier',    par_s11_open_ang,    fen)
        s11_shorted_mag = model_evaluate('fourier',    par_s11_shorted_mag, fen)
        s11_shorted_ang = model_evaluate('fourier',    par_s11_shorted_ang, fen)

        s11_sr_mag      = model_evaluate('polynomial', par_s11_sr_mag,      fen)
        s11_sr_ang      = model_evaluate('polynomial', par_s11_sr_ang,      fen)
        s12s21_sr_mag   = model_evaluate('polynomial', par_s12s21_sr_mag,   fen)
        s12s21_sr_ang   = model_evaluate('polynomial', par_s12s21_sr_ang,   fen)	
        s22_sr_mag      = model_evaluate('polynomial', par_s22_sr_mag,      fen)
        s22_sr_ang      = model_evaluate('polynomial', par_s22_sr_ang,      fen)	

        s11_simu1_mag   = model_evaluate('polynomial', par_s11_simu1_mag,   fen)
        s11_simu1_ang   = model_evaluate('polynomial', par_s11_simu1_ang,   fen)
        s11_simu2_mag   = model_evaluate('polynomial', par_s11_simu2_mag,   fen)
        s11_simu2_ang   = model_evaluate('polynomial', par_s11_simu2_ang,   fen)



    # Adding noise
    if MC_s11_noise[0] > 0: 
        noise        = MC_s11_noise[0] * RMS_s11_LNA_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_LNA_mag  = s11_LNA_mag + noise

    if MC_s11_noise[1] > 0: 
        noise        = MC_s11_noise[1] * RMS_s11_LNA_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_LNA_ang  = s11_LNA_ang + noise

    if MC_s11_noise[2] > 0: 
        noise        = MC_s11_noise[2] * RMS_s11_amb_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_amb_mag  = s11_amb_mag + noise		

    if MC_s11_noise[3] > 0: 
        noise        = MC_s11_noise[3] * RMS_s11_amb_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_amb_ang  = s11_amb_ang + noise

    if MC_s11_noise[4] > 0: 
        noise        = MC_s11_noise[4] * RMS_s11_hot_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_hot_mag  = s11_hot_mag + noise		

    if MC_s11_noise[5] > 0: 
        noise        = MC_s11_noise[5] * RMS_s11_hot_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_hot_ang  = s11_hot_ang + noise

    if MC_s11_noise[6] > 0: 
        noise        = MC_s11_noise[6] * RMS_s11_open_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_open_mag  = s11_open_mag + noise		

    if MC_s11_noise[7] > 0: 
        noise        = MC_s11_noise[7] * RMS_s11_open_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_open_ang  = s11_open_ang + noise

    if MC_s11_noise[8] > 0: 
        noise        = MC_s11_noise[8] * RMS_s11_shorted_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_shorted_mag  = s11_shorted_mag + noise		

    if MC_s11_noise[9] > 0: 
        noise        = MC_s11_noise[9] * RMS_s11_shorted_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_shorted_ang  = s11_shorted_ang + noise

    if MC_s11_noise[10] > 0: 
        noise        = MC_s11_noise[10] * RMS_s11_sr_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_sr_mag   = s11_sr_mag + noise		

    if MC_s11_noise[11] > 0: 
        noise        = MC_s11_noise[11] * RMS_s11_sr_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_sr_ang   = s11_sr_ang + noise		

    if MC_s11_noise[12] > 0: 
        noise         = MC_s11_noise[12] * RMS_s12s21_sr_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s12s21_sr_mag = s12s21_sr_mag + noise		

    if MC_s11_noise[13] > 0: 
        noise         = MC_s11_noise[13] * RMS_s12s21_sr_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s12s21_sr_ang = s12s21_sr_ang + noise

    if MC_s11_noise[14] > 0: 
        noise        = MC_s11_noise[14] * RMS_s22_sr_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s22_sr_mag   = s22_sr_mag + noise		

    if MC_s11_noise[15] > 0: 
        noise        = MC_s11_noise[15] * RMS_s22_sr_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s22_sr_ang   = s22_sr_ang + noise

    if MC_s11_noise[16] > 0: 
        noise         = MC_s11_noise[16] * RMS_s11_simu1_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_simu1_mag = s11_simu1_mag + noise		

    if MC_s11_noise[17] > 0: 
        noise         = MC_s11_noise[17] * RMS_s11_simu1_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_simu1_ang = s11_simu1_ang + noise

    if MC_s11_noise[18] > 0: 
        noise         = MC_s11_noise[18] * RMS_s11_simu2_mag * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_simu2_mag = s11_simu2_mag + noise		

    if MC_s11_noise[19] > 0: 
        noise         = MC_s11_noise[19] * RMS_s11_simu2_ang * np.random.normal(np.zeros(len(fen)), np.ones(len(fen)))
        s11_simu2_ang = s11_simu2_ang + noise








    # Systematic S11

    # Standard deviation of flat errors in frequency
    sigma_mag         = np.abs( 10**(-15.005/20) - 10**(-15/20) )
    sigma_mag_s21     = np.abs( 10**(-0/20) - 10**(-0.001/20) )
    sigma_phase_1mag  = 0.015 # deg at mag=1, which produces 6\sigma = 0.5deg at -15 dB


    if systematic_s11 == 'correlated':

        # flat errors in frequency
        error_mag      = np.random.normal(0, sigma_mag)
        norm_error_ang = np.random.normal(0, 1)



        if MC_s11_syst[0] > 0:
            s11_LNA_mag        = s11_LNA_mag       +   MC_s11_syst[0] * error_mag

        if MC_s11_syst[1] > 0:
            s11_LNA_ang        = s11_LNA_ang       +   MC_s11_syst[1] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_LNA_mag)				

        if MC_s11_syst[2] > 0:
            s11_amb_mag        = s11_amb_mag       +   MC_s11_syst[2] * error_mag

        if MC_s11_syst[3] > 0:
            s11_amb_ang        = s11_amb_ang       +   MC_s11_syst[3] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_amb_mag)				

        if MC_s11_syst[4] > 0:
            s11_hot_mag        = s11_hot_mag       +   MC_s11_syst[4] * error_mag

        if MC_s11_syst[5] > 0:
            s11_hot_ang        = s11_hot_ang       +   MC_s11_syst[5] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_hot_mag)				

        if MC_s11_syst[6] > 0:
            s11_open_mag       = s11_open_mag      +   MC_s11_syst[6] * error_mag

        if MC_s11_syst[7] > 0:
            s11_open_ang       = s11_open_ang      +   MC_s11_syst[7] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_open_mag)				

        if MC_s11_syst[8] > 0:
            s11_shorted_mag    = s11_shorted_mag   +   MC_s11_syst[8] * error_mag

        if MC_s11_syst[9] > 0:
            s11_shorted_ang    = s11_shorted_ang   +   MC_s11_syst[9] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_shorted_mag)



        if MC_s11_syst[10] > 0:
            s11_sr_mag         = s11_sr_mag        +   MC_s11_syst[10] * error_mag			

        if MC_s11_syst[11] > 0:
            s11_sr_ang         = s11_sr_ang        +   MC_s11_syst[11] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_sr_mag)				

        if MC_s11_syst[12] > 0:
            s12s21_sr_mag      = s12s21_sr_mag     +   MC_s11_syst[12] * np.random.normal(0, 2*sigma_mag_s21)       # not correlated. This uncertainty value is technically (2 * A * dA), where A is |S21|. But also |S21| ~ 1

        if MC_s11_syst[13] > 0:
            s12s21_sr_ang      = s12s21_sr_ang     +   MC_s11_syst[13] * (np.pi/180) * np.random.normal(0, 2*sigma_phase_1mag)   # not correlated, This uncertainty value is technically (2 * dP) where dP is the uncertainty in the phase of S21

        if MC_s11_syst[14] > 0:
            s22_sr_mag         = s22_sr_mag        +   MC_s11_syst[14] * error_mag			

        if MC_s11_syst[15] > 0:
            s22_sr_ang         = s22_sr_ang        +   MC_s11_syst[15] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s22_sr_mag)



        if MC_s11_syst[16] > 0:
            s11_simu1_mag      = s11_simu1_mag     +   MC_s11_syst[16] * error_mag

        if MC_s11_syst[17] > 0:
            s11_simu1_ang      = s11_simu1_ang     +   MC_s11_syst[17] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_simu1_mag)

        if MC_s11_syst[18] > 0:
            s11_simu2_mag      = s11_simu2_mag     +   MC_s11_syst[18] * error_mag

        if MC_s11_syst[19] > 0:
            s11_simu2_ang      = s11_simu2_ang     +   MC_s11_syst[19] * norm_error_ang * (np.pi/180) * (sigma_phase_1mag/s11_simu2_mag)			






    elif systematic_s11 == 'uncorrelated':


        if MC_s11_syst[0] > 0:
            s11_LNA_mag       = s11_LNA_mag       +   MC_s11_syst[0] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[1] > 0:
            s11_LNA_ang       = s11_LNA_ang       +   MC_s11_syst[1] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_LNA_mag)				

        if MC_s11_syst[2] > 0:
            s11_amb_mag       = s11_amb_mag       +   MC_s11_syst[2] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[3] > 0:
            s11_amb_ang       = s11_amb_ang       +   MC_s11_syst[3] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_amb_mag)				

        if MC_s11_syst[4] > 0:
            s11_hot_mag       = s11_hot_mag       +   MC_s11_syst[4] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[5] > 0:
            s11_hot_ang       = s11_hot_ang       +   MC_s11_syst[5] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_hot_mag)				

        if MC_s11_syst[6] > 0:
            s11_open_mag      = s11_open_mag      +   MC_s11_syst[6] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[7] > 0:
            s11_open_ang      = s11_open_ang      +   MC_s11_syst[7] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_open_mag)				

        if MC_s11_syst[8] > 0:
            s11_shorted_mag   = s11_shorted_mag   +   MC_s11_syst[8] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[9] > 0:
            s11_shorted_ang   = s11_shorted_ang   +   MC_s11_syst[9] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_shorted_mag)



        if MC_s11_syst[10] > 0:
            s11_sr_mag        = s11_sr_mag        +   MC_s11_syst[10] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[11] > 0:
            s11_sr_ang        = s11_sr_ang        +   MC_s11_syst[11] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_sr_mag)				

        if MC_s11_syst[12] > 0:
            s12s21_sr_mag     = s12s21_sr_mag     +   MC_s11_syst[12] * np.random.normal(0, 1) * 2*sigma_mag_s21        # not correlated. This uncertainty value is technically (2 * A * dA), where A is |S21|. But also |S21| ~ 1

        if MC_s11_syst[13] > 0:
            s12s21_sr_ang     = s12s21_sr_ang     +   MC_s11_syst[13] * np.random.normal(0, 1) * (np.pi/180) * 2 * sigma_phase_1mag    # not correlated, This uncertainty value is technically (2 * dP) where dP is the uncertainty in the phase of S21

        if MC_s11_syst[14] > 0:
            s22_sr_mag        = s22_sr_mag        +   MC_s11_syst[14] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[15] > 0:
            s22_sr_ang        = s22_sr_ang        +   MC_s11_syst[15] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s22_sr_mag)



        if MC_s11_syst[16] > 0:
            s11_simu1_mag     = s11_simu1_mag     +   MC_s11_syst[16] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[17] > 0:
            s11_simu1_ang     = s11_simu1_ang     +   MC_s11_syst[17] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_simu1_mag)

        if MC_s11_syst[18] > 0:
            s11_simu2_mag     = s11_simu2_mag     +   MC_s11_syst[18] * np.random.normal(0, 1) * sigma_mag

        if MC_s11_syst[19] > 0:
            s11_simu2_ang     = s11_simu2_ang     +   MC_s11_syst[19] * np.random.normal(0, 1) * (np.pi/180) * (sigma_phase_1mag/s11_simu2_mag)	







    # Producing complex S11
    rl        = s11_LNA_mag     * (np.cos(s11_LNA_ang)     + 1j*np.sin(s11_LNA_ang))
    ra        = s11_amb_mag     * (np.cos(s11_amb_ang)     + 1j*np.sin(s11_amb_ang))
    rh        = s11_hot_mag     * (np.cos(s11_hot_ang)     + 1j*np.sin(s11_hot_ang))
    ro        = s11_open_mag    * (np.cos(s11_open_ang)    + 1j*np.sin(s11_open_ang))
    rs        = s11_shorted_mag * (np.cos(s11_shorted_ang) + 1j*np.sin(s11_shorted_ang))

    s11_sr    = s11_sr_mag      * (np.cos(s11_sr_ang)      + 1j*np.sin(s11_sr_ang))
    s12s21_sr = s12s21_sr_mag   * (np.cos(s12s21_sr_ang)   + 1j*np.sin(s12s21_sr_ang))
    s22_sr    = s22_sr_mag      * (np.cos(s22_sr_ang)      + 1j*np.sin(s22_sr_ang))

    rs1       = s11_simu1_mag   * (np.cos(s11_simu1_ang)   + 1j*np.sin(s11_simu1_ang))
    rs2       = s11_simu2_mag   * (np.cos(s11_simu2_ang)   + 1j*np.sin(s11_simu2_ang))


    output = np.array([rl, ra, rh, ro, rs, s11_sr, s12s21_sr, s22_sr, rs1, rs2])


    return output

