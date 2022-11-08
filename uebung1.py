import numpy as np
import recognizer.tools as tl
import recognizer.feature_extraction as fe
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


def compute_features():
    # Hier wird audio file gelesen.
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    sampling_rate, audio_data = wavfile.read(audio_file)
    w = 0.4
    h = 0.25
    
    x = fe.make_frames(audio_data, sampling_rate, w, h)

    plt.figure(0)
    plt.title('Audio Data')
    length = audio_data.shape[0] / sampling_rate
    time = np.linspace(0., length, audio_data.shape[0])
    plt.plot(time, audio_data, label="Audio Data")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.figure(1)
    plt.title('Ohne multiplikation mit hamming ')
    for i in range(4):
        plt.subplot(4, 1, i+1)
        length = x[i].shape[0] / sampling_rate
        time = np.linspace(0., length, x[i].shape[0])
        text = str(i) + 'th frame'
        plt.plot(time, tl.normalize2d_array(x[i]), label=text)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

    plt.show()


if __name__ == "__main__":
    compute_features()
