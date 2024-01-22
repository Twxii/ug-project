import librosa
import matplotlib.pyplot as plt
import colorednoise as cn
import numpy as np
import glob
import os
import time
import shutil

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

def apply_noise(colour, signal, noise_factor):
    """Apply some colours of noise to signal.

    Uses colorednoise library - https://github.com/felixpatzelt/colorednoise.

    Parameters:
    -----------
    colour : str
        Type of noise as string.
    signal : np.ndarray
        The signal to apply noise to.
    noise_factor : int
        Amount of noise to be applied to signal.

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
    noise = cn.powerlaw_psd_gaussian(exponent, samples)
    augmented_signal = signal + noise * noise_factor

    return augmented_signal

def windowing(signal, sample_rate, window_length, overlap):
    '''Splits signal into windows of specified lenth with overlap between windows.

    Parameters:
    -----------
    signal : np.ndarray
        Signal to split into windows.
    sample_rate : int > 0
        Samplerate of provided signal.
    window_length : int > 0
        Length of window in milliseconds
    overlap : int > 0
        Overlap between windows as percentage decimal

    Returns:
    --------
    windowed_signal : 2d array
        2d array of each window of the signal
    '''
    window_length_in_samples = int(window_length * sample_rate / 1000)
    overlap_in_samples = int(window_length_in_samples * overlap)
    window_length_in_samples_plus_overlap = int(window_length_in_samples + overlap_in_samples * 2)
    windowed_signal = [signal[i : i + window_length_in_samples_plus_overlap] for i in range(0, signal.size, overlap_in_samples)]

    return windowed_signal

def plot_mel_spectrogram(signal, sample_rate, fig=None, ax=None):
    '''Plots mel spectrogram from signal given.
    
        Uses librosa libray - https://github.com/librosa/librosa.
        Uses matplotlib library - https://github.com/matplotlib/matplotlib.

        Parameters:
        -----------
        signal : np.ndarray
            Signal to plot spectrogram of.
        sample_rate : int > 0
            Samplerate of provided signal.
        fig : matplotlib figure, optional
            The figure to use for plotting.
        ax : matplotlib axis, optional
            The axis to use for plotting.

        Returns:
        ax : matplotlib axis
            The axis containing the mel spectrogram.
    '''
    start_time = time.time()

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000, ax=ax)
    
    if ax.get_images():
        ax.get_images()[0].set_array(S_dB)
    else:
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_axis_off()
        plt.gca().collections[-1].colorbar.remove()

    print(f"spectrogram complete in {time.time() - start_time} seconds")

    return ax

def apply_all_preprocessing(path):
    '''Apply all preprocessing steps to all .wav files in the specified 
        directory and subdirectories.
        Outputs a mel-frequency-spectrogram into the same directory the audio 
        file was taken from.

        Uses librosa library - https://github.com/librosa/librosa.
        Uses matplotlib library - https://github.com/matplotlib/matplotlib.

        Parameters:
        -----------
        path : str
            path of root directory.
    '''
    start_time = time.time()

    for file in glob.iglob(path + "/**/*.wav", recursive=True):
        output_filename = os.path.dirname(file).replace("\\", "-").removesuffix("output.wav")
        output_path = os.path.join(os.path.dirname(file), "Augmented")

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        complete_flag_path = os.path.join(output_path, "complete")
        
        if not os.path.exists(complete_flag_path):
            noise_colours = ["white", "pink", "brown", "blue"]

            original_signal, sr = librosa.load(file)
            normalised_signal = normalisation(original_signal)
            windows = windowing(normalised_signal, sr, 400, 0.5)

            plt.figure()

            for count, window in enumerate(windows):
                plt.clf() # Clear figure
                plot_mel_spectrogram(window, sr)
                save_time = time.time()
                plt.savefig(os.path.join(output_path, f"{output_filename}-normalised-{count}.png"), bbox_inches="tight", pad_inches=0)
                plt.close()
                print(f"saving one spectrogram complete in {time.time() - save_time} seconds")
            
            print(f"{output_filename}-normalised complete at {time.time() - start_time} seconds")

            for colour in noise_colours:
                plus_noise = apply_noise(colour, normalised_signal, 0.05)
                windows = windowing(plus_noise, sr, 400, 0.5)

                for count, window in enumerate(windows):
                    plt.clf() # Clear the figure
                    plot_mel_spectrogram(window, sr)
                    save_time = time.time()
                    plt.savefig(os.path.join(output_path, f"{output_filename}-{colour}-{count}.png"), bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"saving one spectrogram complete in {time.time() - save_time} seconds")
                
                print(f"{output_filename}-{colour} complete at {time.time() - start_time} seconds")

            print()
            open(complete_flag_path, "x").close()

def delete_augmented_dir(path):
    '''Deletes all Augmented directories from path, used to clear all files 
    created by this program.

    Parameters:
    -----------
    path : str
        path of root directory.
    '''
    dirs = glob.glob(path + "/**/Augmented", recursive=True)
    for dir in dirs:
        shutil.rmtree(dir)
        print("Removed " + dir)
    print("Removing complete")

##Test code
#test_file = "RawAudio/output.wav"
#test_signal, sr = librosa.load(test_file)

#normalised_signal = normalisation(test_signal)

#plot_signals_wave(normalised_signal, plus_violet_noise)
        
#windowing(normalised_signal, sr, 400, 0.5)
    
delete_augmented_dir("AcousticSignalLabel")

#apply_all_preprocessing("AcousticSignalLabel")