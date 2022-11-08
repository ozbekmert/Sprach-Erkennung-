import numpy as np
import recognizer.tools as tl
import recognizer.feature_extraction as fe
from scipy.io import wavfile
import matplotlib.pyplot as plt


def compute_features():
    # Hier wird audio file gelesen.
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, audio_data = wavfile.read(audio_file)

    w = 0.025
    h = 0.01

    abs_spec = fe.compute_features(audio_file, w, h)
       
    frequency_bins = np.linspace(0, sampling_rate/2, len(abs_spec[0]))
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05], 'height_ratios': [1, 2]}, figsize=(8, 4))
    length = audio_data.shape[0] / sampling_rate
    time = np.linspace(0., length, audio_data.shape[0])
    im = ax[0, 0].plot(time, tl.normalize2d_array(audio_data), label="Audio Data")
    ax[0, 0].set_xlabel('Time (seconds)')
    ax[0, 0].set_ylabel('Amplitude')
    im = ax[1, 0].imshow(20*np.log10(np.transpose(abs_spec, axes=None)),
                         aspect='auto', origin='lower', extent=[0, length, 0,   frequency_bins[-1]])
    plt.colorbar(im, cax=ax[1, 1], orientation='vertical', boundaries=np.linspace(-40, 100, 20))
    ax[1, 0].set_xlabel('Time Seconds')
    ax[1, 0].set_ylabel('Frequency in Hz')
    ax[1, 1].set_ylabel('Magnitude in dB 20log()')
    plt.figure()

    frequency_bins = np.linspace(0, sampling_rate/2, len(abs_spec[0]))
    plt.imshow(20*np.log10(np.transpose(abs_spec, axes=None)),
               aspect='auto', origin='lower', extent=[0, length, 0, frequency_bins[-1]])
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    plt.close('all')
    compute_features()
