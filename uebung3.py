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
    n_filters = 24
    fbank = fe.get_mel_filters(sampling_rate, w, n_filters, 0, 8000)
    plt.figure()
    plt.plot(fbank.T)
    plt.xlabel('Frequency in Hz', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    
    # mas = fe.apply_mel_filters(abs_spec, fbank)
    mas = fe.compute_features(audio_file, w, h, 'FBANK', n_filters, 0, 8000)
    length = audio_data.shape[0] / sampling_rate
    
    plt.figure()
    plt.imshow(mas, aspect='auto', interpolation='none', origin='lower', extent=[0 , length, 0,  len(mas)])
    plt.xlabel('Time in Seconds', fontsize=14)
    plt.ylabel('Mel Filter Index', fontsize=14)

    plt.show()


if __name__ == "__main__":
    plt.close('all')
    compute_features()  # -*- coding: utf-8 -*-
