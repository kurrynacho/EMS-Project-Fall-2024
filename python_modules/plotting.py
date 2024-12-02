import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import periodogram
from scipy.signal import find_peaks


import preprocessing

def plot_data(data):
    plt.figure(figsize=(20,10))
    plt.plot(data['date'], data['count'])
    plt.xlabel('date')
    plt.show()

def plot_predictions(data, predictions_dict, length):
    plt.figure(figsize=(20,10))
    plt.plot(data['date'][-length:], data['count'][-length:])
    plt.xlabel('date')
    for model in predictions_dict:
        plt.plot(data['date'][-len(predictions_dict[model]):], predictions_dict[model], label=model +'forecast') #forecast
    plt.show()


def plot_smoothed_data(data, window_size=180):
    # Ensure the input is a DataFrame
    trend = preprocessing.get_trend(data, window_size)
    
    # Plot the original data and the extracted trend
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['count'], label='Original Data', alpha=0.7)
    plt.plot(data['date'], trend, label=f'Moving Average (Window={window_size})', color='red', linewidth=2)
    plt.title('Original Data and Trend')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_trend_residue(data, window_size=180, type='mult'):
    # Ensure the input is a DataFrame
    trend = preprocessing.get_trend(data, window_size)
    
    # Plot the original data and the extracted trend
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], trend, label=f'Moving Average (Window={window_size})', color='red', linewidth=2)
    if type =='mult':
        plt.plot(data['date'], data['count']/trend, label=f'De-trended', color='green', linewidth=2)
    if type =='add':
        plt.plot(data['date'], data['count']-trend, label=f'De-trended', color='green', linewidth=2)
    plt.plot(data['date'], trend, label=f'Moving Average (Window={window_size})', color='red', linewidth=2)

    plt.title('Trend and Detrended')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def fftplot(data, frequency_scaler=2):
    # Perform Fourier Transform
    N = len(data['count'])  # Number of samples
    T = 1.0               # Sampling interval (adjust as per your data)
    frequencies = fftfreq(N, T)[:N // frequency_scaler]  # Positive frequencies
    fft_values = fft(data['count'])[:N //frequency_scaler ]  # FFT values

    # Plot Power Spectrum
    plt.plot(frequencies, np.abs(fft_values))
    plt.title("Frequency Domain (Spectral Analysis)")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.ylim(0,100)
    plt.show()

def pgram_plot(data):
    # Sampling frequency (fs): 1/T where T is the sampling interval
    sampling_frequency = 1  # Adjust this based on your data
    # Compute the periodogram
    frequencies, power_spectrum = periodogram(data['count'], fs=sampling_frequency)
    # Plot the periodogram
    plt.semilogy(frequencies, power_spectrum)  # semilogy for better visualization
    plt.title("Periodogram (Spectral Analysis)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.show()
    
    # Find peaks in the power spectrum
    peaks, _ = find_peaks(power_spectrum, height=0.1)  # Adjust height threshold as needed

    # Extract dominant frequencies, powers, and periods
    dominant_frequencies = frequencies[peaks]
    dominant_powers = power_spectrum[peaks]
    dominant_periods = 1 / dominant_frequencies

    # Remove frequencies too close to zero
    threshold = 1/400  # Set a threshold to exclude low frequencies
    valid_indices = dominant_frequencies > threshold
    dominant_frequencies = dominant_frequencies[valid_indices]
    dominant_powers = dominant_powers[valid_indices]
    dominant_periods = dominant_periods[valid_indices]

    # Sort by power (descending order)
    sorted_indices = np.argsort(dominant_powers)[::-1]
    dominant_frequencies = dominant_frequencies[sorted_indices]
    dominant_powers = dominant_powers[sorted_indices]
    dominant_periods = dominant_periods[sorted_indices]

    # Print sorted dominant frequencies, periods, and powers
    print("Sorted Dominant Frequencies, Periods, and Powers (Excluding Low Frequencies):")
    for freq, period, pwr in zip(dominant_frequencies, dominant_periods, dominant_powers):
        print(f"Frequency: {freq:.6f} cycles/day, Period: {period:.2f} days, Power: {pwr:.2f}")

    