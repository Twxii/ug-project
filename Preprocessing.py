import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft2, ifft2
import colorednoise as cn

filename = 'RawAudio/output.wav'
original_signal, sr = librosa.load(filename)

def plot_signals(original_signal, augmented_signal):
    fig, ax = plt.subplots(nrows=2, sharex=True)

    librosa.display.waveshow(original_signal, sr=sr, ax=ax[0], color="blue")
    ax[0].set(title='Original Audio')
    ax[0].label_outer()

    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1], color="blue")
    ax[1].set(title='Augmented Audio')
    ax[1].label_outer()

    plt.show()

## Normalise signal between -1 and 1
def normalisation(signal):
    augmented_signal = librosa.util.normalize(signal)
    return augmented_signal

def generate_white_noise(signal):
    noise = np.random.normal(0, signal.std(), signal.size)
    return noise
    
def apply_white_noise(signal, noise_factor):
    noise = generate_white_noise(signal)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal

def generate_pink_noise(signal):
    nrows = signal.size
    ncols = 16

    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values

def apply_pink_noise(signal, noise_factor):
    augmented_signal = 0

    return augmented_signal

normalised_signal = normalisation(original_signal)

pink_noise_signal = apply_pink_noise(normalised_signal, 0.1)

#plot_signals(original_signal, pink_noise_signal)
plot_signals(original_signal, generate_pink_noise(original_signal))