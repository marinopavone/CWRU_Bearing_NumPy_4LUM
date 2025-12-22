import matplotlib.pyplot as plt
import numpy as np
import math

import numpy as np


def custom_fft(signal, fs=12000):
    N = len(signal)
    # FFT
    spectrum_full = np.fft.fft(signal)
    # Frequency axis
    freqs_full = np.fft.fftfreq(N, d=1 / fs)
    # Keep only positive half
    half = N // 2
    freqs = freqs_full[:half]
    # spectrum = spectrum_full[:half]
    spectrum = np.abs(spectrum_full[:half]) / N  # modulo normalizzato

    # return freqs, spectrum
    return  spectrum


def split_signal_into_segments(signal, w):
    n = len(signal)
    M = n // w                     # how many full segments
    trimmed = signal[:M * w]       # remove leftover tail
    segments = trimmed.reshape(M, w)
    return segments


def split_and_fft(signal, w, fs=12000):
    """
    Split in segments of length w and apply custom FFT to each.
    Returns a matrix (M × w/2) where M = n // w.
    """
    n = len(signal)
    M = n // w
    trimmed = signal[:M * w]
    segments = trimmed.reshape(M, w)

    # Apply FFT to each segment
    fft_list = []

    for seg in segments:
        spectrum = custom_fft(seg, fs)
        fft_list.append(spectrum)

    return np.vstack(fft_list)


import pywt
def split_and_wavelet(signal, w, Fs=12000):
    """
    Split in segments of length w and apply custom FFT to each.
    Returns a matrix (M × w/2) where M = n // w.
    """
    n = len(signal)
    M = n // w
    trimmed = signal[:M * w]
    segments = trimmed.reshape(M, w)

    max_level = 3
    wavelet_list = []

    for seg in segments:
        coeffs = pywt.wavedec(seg, 'db4', level=max_level)
        wavelet_list.append(  np.concatenate(coeffs) )

    return wavelet_list

def average_over_window(arr, aw):
    rows, cols = arr.shape
    M = cols // aw                     # number of full blocks
    trimmed = arr[:, :M * aw]         # remove leftover columns
    reshaped = trimmed.reshape(rows, M, aw)
    out = reshaped.mean(axis=2)       # average inside each block

    return out

import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

def stratified_train_test_split(
    df,
    label_col="Fault",
    test_size=0.2,
    random_state=42
):
    """
    Stratified random split preserving class proportions.

    Parameters:
        df (DataFrame): Input dataframe
        label_col (str): Column containing class labels
        test_size (float): Fraction of samples per class in test set
        random_state (int): Seed for reproducibility

    Returns:
        train_df, test_df
    """
    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for label, group in df.groupby(label_col):
        n_samples = len(group)
        n_test = int(np.round(n_samples * test_size))

        # Shuffle indices within the class
        indices = rng.permutation(group.index)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        test_parts.append(df.loc[test_idx])
        train_parts.append(df.loc[train_idx])

    train_df = (
        pd.concat(train_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    test_df = (
        pd.concat(test_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return train_df, test_df


def buil_x_y(df):
    x_spectrums = []
    x_rpm = []
    y_lab = []
    for index, elem in df.iterrows():
        c_RPM = elem["RPM"]
        for segment in elem["Freq_data"]:
            x_spectrums.append(segment)
            x_rpm.append(np.array([c_RPM]))
            y_lab.append(elem["Classification_label"])
    x_spectrums = np.array(x_spectrums)
    x_spectrums_compressed = average_over_window(x_spectrums, aw=3)
    return x_rpm, x_spectrums, y_lab