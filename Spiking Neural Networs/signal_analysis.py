import numpy as np
import pandas as pd
import scipy 
import os 
import sys
from spectrum import *
from scipy.signal import hilbert, chirp 

def hilbert_phase(x):
    sig=x-x.mean()
    std = sig.std()
    sig/=std
    analytic_sig = hilbert(sig)
    instantaneous_phase = np.angle(analytic_sig)
    amplitude_envelope  = np.abs(analytic_sig)
    return np.mod(instantaneous_phase,2*np.pi), (amplitude_envelope+x.mean())*std

def bandpass_filter(xs, norder, f_range, fs=1e4): 
    sos = scipy.signal.butter(N=norder, Wn=f_range, btype="bandpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, xs)

def lowpass_filter(xs, norder, f_range, fs=1e4):
    sos = scipy.signal.butter(N=norder, Wn=f_range, btype="lowpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, xs)
    
def bandpass_filter_and_hilbert_transform(sig, fs, f0, df, norder):
    scale = sig.mean() 
    f_range = (f0-df, f0+df)
    sig_band = bandpass_filter(sig, norder, f_range, fs)
    phase, envelope= hilbert_phase(sig_band)
    return sig_band+scale, envelope+scale,  phase

def get_plv(phi1,phi2):
    dphi = np.mod(phi1-phi2,2*np.pi)
    return np.abs(np.sum(np.exp(1j*dphi)))/len(dphi)



