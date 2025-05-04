import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
import streamlit as st

st.title('FFT Plot')

file = st.file_uploader('Upload a .WAV file to get started',
                        type='.wav')

if file:
    fs, data = wavfile.read(file)

    # Check if data is stereo or mono
    if data.ndim > 1:
        # Extract the first channel and normalize
        a = data.T[0]
        b = np.array([(ele / 2**8.) * 2 - 1 for ele in a])  # Normalize to [-1, 1)
    else:  # Mono
        b = np.array([(ele / 2**8.) * 2 - 1 for ele in data])  # Normalize to [-1, 1)

    # Compute the FFT
    c = fft(b)
    d = len(c) // 2  # Only consider the first half (real signals have symmetric FFT)

    # Compute frequency bins
    frequencies = np.linspace(0, fs / 2, d - 1)
    magnitudes = np.abs(c[:d - 1])

    # Plot area chart (log-log)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("FFT of the Uploaded WAV File")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, which="both", ls="--", lw=0.5)

    # Fill the area under the curve
    ax.fill_between(frequencies, magnitudes, color='c', alpha=0.5)

    # Display the plot in Streamlit
    st.pyplot(fig)
