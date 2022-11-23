"""
Library for the signal processing project.
Utilitarian functions for the project.
"""
import scipy.io.wavfile as wf

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def compute_dirac_impulse():
    """
    Returns the dirac impulse.
    """
    return sig.unit_impulse(100)


def compute_l_filter(n, d, size):
    """
    Given a size, returns the L filter.
    """
    t = list(np.zeros(size))
    return sig.lfilter(n, d, t)


def compute_frequency_response(n, d, sampling_freq):
    """
    Given the filter parameters and a sampling freq,
    returns the frequency response.
    """
    w, h = sig.freqz(n, d, worN=2048, fs=sampling_freq)
    return w, h


def compute_time_vector(samples, sampling_freq):
    """
    Given a number of samples and a sampling frequency,
    returns the time vector.
    """
    time_inter = samples / sampling_freq
    return np.linspace(0, time_inter, samples)


def compute_omega(freq):
    """
    Given a freq, returns the omega.
    """
    return 2.0 * np.pi * freq


def do_a_sinus(freq, amp, time_vector):
    """
    Given a freq, an amplitude and a time vector of samples,
    returns the sinusoidal signal.
    """
    w = 2.0 * np.pi * freq
    return amp * np.sin(w * time_vector)


def compute_freq_from_sinus(N_dft, sinus, sampling_freq):
    """
    Given a sinus and a sampling freq, returns the frequency vector.
    """
    sample = len(sinus)
    time_inter = sample / sampling_freq
    extended_sinus = np.concatenate((sinus, [0] * (N_dft - len(sinus))))
    freqz = sig.freqz(extended_sinus, worN=N_dft)
    t = np.linspace(0, time_inter, N_dft)
    return t, 20 * np.log10(abs(freqz[1]))


def plot_show_save(sines, time_vector, save):
    """
    Given an array of sine waves and a save boolean,
    plots the sine waves and saves them if 'save' is True.
    """
    for i in range(len(sines)):
        plt.plot(time_vector, sines[i])
    if save:
        plt.savefig('plot.png')
    plt.show()


def superimpose_signals(signals):
    """
    Given an array of signals, returns the superimposed signal.
    """
    superimposed_signal = np.zeros(len(signals[0]))
    for i in range(len(signals)):
        superimposed_signal += signals[i]
    return superimposed_signal


def create_bandpass_filter(freq, sampling_freq):
    """
    Given a freq and a sampling freq, returns the bandstop filter.
    """
    bandpass_theta = 2.0 * np.pi * freq / sampling_freq
    procent_freq = round(freq * 0.01)
    rhos = np.linspace(1, 0, 1000)
    rho = 0
    boolean = False
    for r in rhos:
        n = [1, -np.cos(bandpass_theta), 1]
        d = [1, -2 * np.cos(bandpass_theta) * r, r ** 2]
        w, h = sig.freqz(n, d, worN=2048, fs=sampling_freq)
        for i in range(len(w)):
            amp_low = 0
            amp_mid = 0
            amp_high = 0
            if round(w[i], 1) == freq - procent_freq:
                idx_low = i
                amp = h[idx_low]
                amp_low = amp.real
            if round(w[i], 1) == freq:
                idx_mid = i
                amp = h[idx_mid]
                amp_mid = amp.real
            if round(w[i], 1) == freq + procent_freq:
                idx_high = i
                amp = h[idx_high]
                amp_high = amp.real
            if (amp_mid - amp_low) <= 0.01 and (amp_mid - amp_high) <= 0.01:
                rho = r
                boolean = True
                break
        if boolean:
            break
    k = 1 / (1 - rho * np.cos(bandpass_theta))
    n = [k, -2 * k * np.cos(bandpass_theta), k]
    d = [1, -2 * rho * np.cos(bandpass_theta), rho ** 2]
    w, h = sig.freqz(n, d, worN=2048, fs=sampling_freq)
    return n, d, w, h, rho, k


def compute_bandpass_transfert_function(n, d):
    """
    Given the filter parameters, returns the transfert function.
    """
    return sig.TransferFunction(n, d)

def create_bandstop_filter(freq, sampling_freq):
    """
    Given a freq and a sampling freq, returns the bandstop filter.
    """
    bandstop_theta = 2.0 * np.pi * freq / sampling_freq
    procent_freq = round(freq * 0.01)
    rhos = np.linspace(1, 0, 1000)
    rho = 0
    boolean = False
    for r in rhos:
        n = [1, -2 * r * np.cos(bandstop_theta), r ** 2]
        d = [1, -2 * np.cos(bandstop_theta), 1]
        w, h = sig.freqz(n, d, worN=2048, fs=sampling_freq)
        for i in range(len(w)):
            amp_low = 0
            amp_mid = 0
            amp_high = 0
            if round(w[i], 1) == freq - procent_freq:
                idx_low = i
                amp = h[idx_low]
                amp_low = amp.real
            if round(w[i], 1) == freq:
                idx_mid = i
                amp = h[idx_mid]
                amp_mid = amp.real
            if round(w[i], 1) == freq + procent_freq:
                idx_high = i
                amp = h[idx_high]
                amp_high = amp.real
            if (amp_mid - amp_low) <= 0.01 and (amp_mid - amp_high) <= 0.01:
                rho = r
                boolean = True
                break
        if boolean:
            break
    k = 1 / (1 - rho * np.cos(bandstop_theta))
    n = [k, -2 * k * np.cos(bandstop_theta), k]
    d = [1, -2 * rho * np.cos(bandstop_theta), rho ** 2]
    w, h = sig.freqz(n, d, worN=2048, fs=sampling_freq)
    return n, d, w, h, rho, k

def normalise(s):
    """
    Given a signal, returns the normalised signal.
    """
    return s / max(abs(s))

## 2 - Anti-aliasing filter synthesis
def create_filter_cheby(wp, ws, gpass, gstop, fs):
    """
    Given the filter parameters, returns the filter.
    """
    n, wn = sig.cheb1ord(wp, ws, gpass, gstop, fs=fs)
    b, a = sig.cheby1(n, gpass, wn, btype='lowpass', analog=False, output='ba', fs=fs)
    return b, a

def create_filter_cauer(wp, ws, gpass, gstop, fs):
    """
    Given the filter parameters, returns the filter.
    """
