import librosa
import matplotlib.pyplot as plt
import colorednoise as cn
import numpy as np
import glob
import os
import time
import shutil
from multiprocessing import Pool
import re
import random

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

def apply_noise(colour, signal, noise_factor, noise_variation):
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
    noise_variation : float > 0
        Amount of variation to add to noise_factor. e.g., +- 0.01

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
    noise_factor = noise_factor + random.uniform(-noise_variation, noise_variation)
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
    windowed_signal = [signal[i : i + window_length_in_samples] for i in range(0, signal.size, overlap_in_samples)]
    
    x = False
    while x == False:
        i = len(windowed_signal) - 1
        if len(windowed_signal[i]) < window_length_in_samples:
            windowed_signal.pop(i)
        else:
            x = True

    return windowed_signal

def window_labels(file, window_length, overlap):
    output_filename = os.path.dirname(file).replace("\\", "_")
    output_path = os.path.join(os.path.dirname(file), "Augmented")
    window_length_seconds = window_length / 1000
    
    windowed_labels = []
    activities = []

    input_label_file = open(os.path.join(file))
    for line in input_label_file:
        activity = re.sub(r"[0-9].", "", line).lstrip().rstrip("\n ").replace(".\t", "")
        activity_times = re.sub(r"[A-Za-z]", "", line).rstrip()
        activity_start_time = float(activity_times[:8])
        activity_end_time = float(activity_times[-8:])

        activities.append([activity, activity_start_time, activity_end_time])

    file_length = float(activities[-1][2])

    current_time = 0

    while current_time + window_length_seconds <= file_length:
        interval_end = round(current_time + window_length / 1000, 2)

        windowed_labels.append(["", current_time, interval_end])

        current_time = round(current_time + window_length_seconds - (window_length_seconds * overlap), 2)

    for window in windowed_labels:
        for activity in activities:
            if window[1] < activity[2] and window[0] == "":
                window[0] = activity[0]
            elif window[1] < activity[2] and window[2] > activity[1] and window[0] != activity[0]:
                window[0] = [window[0], activity[0]]     

    #print(windowed_labels)
    
    #print(os.path.join(output_path, f"{output_filename}_{count}_labels.txt"))

    return 0

def count_classes(file):
    '''Counts activity classes in the given file.

    Parameters:
    -----------
    file : str
        path of file.

    Returns:
    --------
    labels : 2d list
        2d list in the format [["activity0", count], ["activity1", count], ...]
    '''
    labels = []
    input_label_file = open(os.path.join(file))
    for line in input_label_file:
        label = re.sub(r"[0-9].", "", line).lstrip().rstrip("\n ").replace(".\t", "")
        if not any(label in sublist for sublist in labels):
            labels.append(([label, 1]))
        else:
            for x in labels:
                if x[0] == label:
                    x[1] += 1
    #print(labels)
    #print(os.path.join(output_path, f"{output_filename}_label_count.txt"))
    return labels

def join_label_lists(listOne, listTwo):
    '''Takes two of the label lists and adds listTwo to listOne.

    Parameters:
    -----------
    listOne : list
        2d label list
    listTwo : list
        2d label list

    Returns:
    --------
    listOne : list
        2d label list
    '''
    for x in listTwo:
        if not any(x[0] in sublist for sublist in listOne):
            listOne.append([x[0], x[1]])
        else:
            for y in listOne:
                if y[0] == x[0]:
                    y[1] = y[1] + x[1]

    return listOne

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
    #385 x 128
    fig, ax = plt.subplots()

    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000, ax=ax)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_axis_off()
    plt.gca().collections[-1].colorbar.remove()

    return ax

def process_file(file):
    '''Apply all preprocessing steps to file provided.
        Outputs a mel-frequency-spectrogram into the same a new "Augmented" 
        directory from the location the audio file was taken from.

        Uses librosa library - https://github.com/librosa/librosa.
        Uses matplotlib library - https://github.com/matplotlib/matplotlib.

        Parameters:
        -----------
        file : iterator
            file to be processed.
    '''
    start_time = time.time()

    output_filename = os.path.dirname(file).replace("\\", "_").removesuffix("output.wav")
    output_path = os.path.join(os.path.dirname(file), "Augmented")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    complete_flag_path = os.path.join(output_path, "complete")
    
    if not os.path.exists(complete_flag_path):
        noise_colours = ["white", "pink", "brown", "blue"]

        original_signal, sr = librosa.load(file, sr=44100)
        normalised_signal = normalisation(original_signal)
        windows = windowing(normalised_signal, sr, 400, 0.5)

        plt.figure()

        for count, window in enumerate(windows):
            plt.clf() # Clear figure
            plot_mel_spectrogram(window, sr)
            plt.savefig(os.path.join(output_path, f"{output_filename}_normalised_{count}.png"), bbox_inches="tight", pad_inches=0)
            plt.close()
        
        #print(f"{output_filename}_normalised complete at {time.time() - start_time} seconds")
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"{output_filename}_normalised complete at {current_time}")

        for colour in noise_colours:
            plus_noise = apply_noise(colour, normalised_signal, 0.05, 0.01)
            windows = windowing(plus_noise, sr, 400, 0.5)

            for count, window in enumerate(windows):
                plt.clf() # Clear the figure
                plot_mel_spectrogram(window, sr)
                plt.savefig(os.path.join(output_path, f"{output_filename}_{colour}_{count}.png"), bbox_inches="tight", pad_inches=0)
                plt.close()
            
            #print(f"{output_filename}_{colour} complete at {time.time() - start_time} seconds")
            current_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"{output_filename}_{colour} complete at {current_time}")
        
        open(complete_flag_path, "x").close()

def apply_all_preprocessing(path, pool_size):
    '''Applies all preprocessing steps to all .wav files in the given path. Uses
        multiprocessing for faster completion.

        Parameters:
        -----------
        path : str
            Path of parent directory to be processed.
        pool_size : int
            Number of "cpu cores" to use.
    '''
    with Pool(pool_size) as pool:
        pool.map(process_file, glob.iglob(os.path.join(path, "**/*.wav"), recursive=True))

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

def count_all_classes(path):
    '''
    '''
    label_list = []
    for file in glob.iglob(os.path.join(path, "**/*.txt"), recursive=True):
        label_list = join_label_lists(label_list, count_classes(file))
    print(label_list)

##Test code
if __name__ == "__main__":
    start_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Start time: {start_time}")
    #test_file = "RawAudio/output.wav"
    #test_signal, sr = librosa.load(test_file, sr=44100)

    #plot_mel_spectrogram(test_signal, sr)

    #normalised_signal = normalisation(test_signal)

    #plot_signals_wave(normalised_signal, plus_violet_noise)
        
    #windowing(normalised_signal, sr, 400, 0.5)

    #delete_augmented_dir("AcousticSignalLabel")

    #apply_all_preprocessing("AcousticSignalLabel\Series1", 8)

    #count_all_classes("AcousticSignalLabel")

    window_labels("AcousticSignalLabel\Series1\A\A10\Labels_A10.txt", 400, 0.5)

    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Start time was: {start_time}")
    print(f"Complete at: {current_time}")