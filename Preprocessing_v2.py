import librosa
import matplotlib.pyplot as plt
import colorednoise as cn
import numpy as np
import glob

def plot_signals_wave(signal_one, signal_two):
    """Plot and show two signals as waveforms on top of each other for 
        comparison.

    Uses librosa library - https://github.com/librosa/librosa.
    Uses matplotlib library - https://github.com/matplotlib/matplotlib.

    Parameters:
    -----------
    signal_one : np.ndarray
        First signal to be shown
    signal_two : np.ndarray
        Second signal to be shown
    """
    fig, ax = plt.subplots(nrows=2, sharex=True)

    librosa.display.waveshow(signal_one, ax=ax[0], color="blue")
    ax[0].set(title="Signal One")
    ax[0].label_outer()

    librosa.display.waveshow(signal_two, ax=ax[1], color="blue")
    ax[1].set(title="Signal Two")
    ax[1].label_outer()

    plt.show()

def normalisation(signal):
    """Apply normalisation to signal.

    Uses librosa library - https://github.com/librosa/librosa.

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

    Uses colorednoise library - https://github.com/felixpatzelt/colorednoise.

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
        exponent = -1
    elif colour == "violet":
        exponent = -2
    else:
        raise Exception("noise colour not supported")
    
    samples = signal.size
    noise_factor = 0.05
    noise = cn.powerlaw_psd_gaussian(exponent, samples)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal

def apply_all_preprocessing(path):
    """Apply all preprocessing steps to all .wav files in the specified 
        directory and subdirectories.
        Outputs a mel-frequency-spectrogram into the same directory the audio 
        file was taken from.

        Uses librosa library - https://github.com/librosa/librosa.
        Uses matplotlib library - https://github.com/matplotlib/matplotlib.

        Parameters:
        -----------
        path : str
            path of root directory
    """
    for file in glob.iglob(path + "/**/*.wav", recursive=True):
        output_filename = file.replace("\\", "-").removesuffix(".wav")
        path = file.removesuffix("output.wav")

        original_signal, sr = librosa.load(file)
        normalised_signal = normalisation(original_signal)
        plus_white_noise = apply_noise("white", normalised_signal)
        plus_pink_noise = apply_noise("pink", plus_white_noise)
        plus_brown_noise = apply_noise("brown", plus_pink_noise)

        S = librosa.feature.melspectrogram(y=plus_brown_noise, sr=sr, 
                                            n_mels=128, fmax=8000)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', 
                                sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_axis_off()
        plt.gca().collections[-1].colorbar.remove()

        #plt.savefig(path + output_filename + ".png", bbox_inches='0')
        #print(path + output_filename + ".png")
        plt.show()


##Test code
#test_file = "RawAudio/output.wav"
#test_signal, sr = librosa.load(test_file)

#normalised_signal = normalisation(test_signal)

#plus_white_noise = apply_noise("white", normalised_signal)

#plus_pink_noise = apply_noise("pink", plus_white_noise)

#plus_brown_noise = apply_noise("brown", plus_pink_noise)

#plus_blue_noise = apply_noise("blue", plus_brown_noise)

#plus_violet_noise = apply_noise("violet", plus_blue_noise)

#plot_signals_wave(test_signal, normalised_signal)

#plot_signals_wave(normalised_signal, plus_violet_noise)

apply_all_preprocessing("AcousticSignalLabel")