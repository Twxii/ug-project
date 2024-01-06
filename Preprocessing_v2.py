import librosa
import matplotlib.pyplot as plt
import colorednoise as cn

filename = 'RawAudio/output.wav'
original_signal, sr = librosa.load(filename)

def plot_signals(signal_one, signal_two):
    fig, ax = plt.subplots(nrows=2, sharex=True)

    librosa.display.waveshow(signal_one, sr=sr, ax=ax[0], color="blue")
    ax[0].set(title='Original Audio')
    ax[0].label_outer()

    librosa.display.waveshow(signal_two, sr=sr, ax=ax[1], color="blue")
    ax[1].set(title='Augmented Audio')
    ax[1].label_outer()

    plt.show()

def normalisation(signal):
    """Apply normalisation to signal.
    Uses librosa library.

    Parameters:
    -----------
    signal : np.ndarray
        The signal to apply normalisation to.

    Returns:
    --------
    augmented_signal : np.ndarray
        The signal with normalisation applied.
    """
    augmented_signal = librosa.util.normalize(signal)
    return augmented_signal

def apply_noise(colour, signal):
    """Apply some colours of noise to signal.
    Uses colorednoise library.

    Parameters:
    -----------
    colour : str
        Type of noise as string.
    signal : np.ndarray
        The signal to apply noise to.

    Returns:
    --------
    augmented_signal : np.ndarray
        The signal with noise applied.
    """
    if colour == "white":
        exponent = 0
    elif colour == "pink":
        exponent = 1
    elif colour == "brown":
        exponent = 2
    elif colour == "blue":
        -1
    else:
        raise Exception("noise colour not supported")
    
    samples = signal.size
    noise_factor = 0.05
    noise = cn.powerlaw_psd_gaussian(exponent, samples)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal



normalised_signal = normalisation(original_signal)

plus_white_noise = apply_noise("white", normalised_signal)

plus_pink_noise = apply_noise("pink", plus_white_noise)

plus_brown_noise = apply_noise("brown", plus_pink_noise)

blue_noise = cn.powerlaw_psd_gaussian(-1, original_signal.size)

plot_signals(normalised_signal, blue_noise)

plot_signals(normalised_signal, plus_brown_noise)