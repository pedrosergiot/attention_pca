__all__ = ["load_data","filt_data","get_energy"]

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


# load data
def load_data(filename):
    # load dataset
    exames_dict = loadmat(filename)

    target = exames_dict['target']
    exames = exames_dict['exames']

    # samples as the first dimension
    exames = np.transpose(exames, (2, 0, 1))

    target = target[0]
    return exames, target

# filter the data (filtfilt with a butterworth filter)
def filt_data(data, fs, half_band, order, f1, f2):
    nyq = 0.5 * fs  # nyquist frequency

    a1, b1 = butter(order, [(f1 - half_band) / nyq, (f1 + half_band) / nyq], btype='bandpass')
    a2, b2 = butter(order, [(f2 - half_band) / nyq, (f2 + half_band) / nyq], btype='bandpass')

    dataf1 = filtfilt(a1, b1, data, axis=1)
    dataf2 = filtfilt(a2, b2, data, axis=1)

    return dataf1, dataf2

# calculates the energy of the filtered data
def get_energy(data):
    data = np.square(data)
    return np.sum(data, axis=1)