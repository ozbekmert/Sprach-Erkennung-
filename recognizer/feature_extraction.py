import recognizer.tools as tl
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import math


def make_frames(audio_data, sampling_rate, window_size, hop_size):

    # Nächsthöhere 2er Potenz wird bereits von next_pow2 berechnet
    
    rounded_window_size = int(math.pow(2, tl.next_pow2_samples(window_size, sampling_rate)))
    
    # Diskrete hop_size
    hop_size_samples = tl.sec_to_samples(hop_size, sampling_rate)
    
    number_of_frames = tl.get_num_frames(len(audio_data), rounded_window_size, hop_size_samples)

    frames_2d = np.zeros((number_of_frames, rounded_window_size))
    
    frame_overlap = rounded_window_size - hop_size_samples
    
    audio_data_index = 0
    for i in range(number_of_frames):        
        if i == 0 and rounded_window_size < len(audio_data):
            # first frame
            audio_data_index += rounded_window_size
            frames_2d[i] = audio_data[i:audio_data_index]
            audio_data_index -= frame_overlap
        elif (audio_data_index + rounded_window_size) < len(audio_data):
            frames_2d[i] = audio_data[audio_data_index:(audio_data_index + rounded_window_size)]
            audio_data_index += hop_size_samples
        else:
            
            # Restliche Anzahl an samples (weniger als rounded_window_size)
            num_samples = len(audio_data) - audio_data_index
            
            # Zero padding ist nicht nötig, da array mit zeros initialisiert wurde
            frames_2d[i][:num_samples] = audio_data[audio_data_index: (audio_data_index + num_samples)]
            
    window_hamming = np.hamming(rounded_window_size)
        
    # Elementweise Multiplikation
    frames_2d = frames_2d * window_hamming
    return frames_2d


def compute_absolute_spectrum(frames):
    
    sample_lenght = len(frames[0])//2 + 1
  
    number_of_frames = len(frames)
    
    abs_spec_array = np.array([[0 for i in range(sample_lenght)]
                                for j in range(number_of_frames)], dtype=float)

    for i in range(len(frames)):
        fourier_transform = np.fft.rfft(frames[i])
        abs_spec_array[i] = np.abs(fourier_transform)

    return abs_spec_array


def compute_features(audio_file, window_size=0.025, hop_size=0.01, feature_type='STFT',
                     n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13):

    sampling_rate, audio_data = wavfile.read(audio_file)
    audio_data = tl.normalize2d_array(audio_data)
    x = make_frames(audio_data, sampling_rate, window_size, hop_size)

    if feature_type == 'STFT':
        abs_spec = compute_absolute_spectrum(x)
        return abs_spec
    # remove sampling ret val etc
    elif feature_type == 'MFCC':
        abs_spec = compute_absolute_spectrum(x)
                   
        fbank = get_mel_filters(sampling_rate,window_size, n_filters, fbank_fmin, fbank_fmax)
        mas = apply_mel_filters(abs_spec, fbank)
        cepstrum = compute_cepstrum(mas, num_ceps)
        return cepstrum
    elif feature_type == 'MFCC_D':
        abs_spec = compute_absolute_spectrum(x)
        fbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        mas = apply_mel_filters(abs_spec, fbank)
        y = compute_cepstrum(mas, num_ceps)
        delta_1 = get_delta(y)
        delt_app = append_delta(y, delta_1)
        return delt_app
    elif feature_type == 'MFCC_D_DD':
        abs_spec = compute_absolute_spectrum(x)
        fbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        mas = apply_mel_filters(abs_spec, fbank)
        y = compute_cepstrum(mas, num_ceps)
        delta_1 = get_delta(y)
        delta_2 = get_delta(delta_1)
        delt_app = append_delta(y, delta_1)
        delt_app2 = append_delta(delt_app, delta_2)
        return delt_app2
    elif feature_type == 'FBANK':
        abs_spec = compute_absolute_spectrum(x)
        fbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        mas = apply_mel_filters(abs_spec, fbank)
        
        return np.log(mas)


def compute_features_with_context(audio_file, window_size=25e-3,hop_size=10e-3, feature_type='STFT', n_filters=24,
                                  fbank_fmin=0,fbank_fmax=8000, num_ceps=13, left_context=4, right_context=4):
    return add_context(compute_features(audio_file, window_size, hop_size, feature_type, n_filters, fbank_fmin,
                                        fbank_fmax, num_ceps), left_context, right_context)


def get_mel_filters(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):
      
    lowmel = tl.hz2mel(f_min)
    highmel = tl.hz2mel(f_max)
    melpoints = np.linspace(lowmel, highmel, n_filters+2)  # the list of M+2 Mel-spaced frequencies
    hz_points = tl.mel2hz(melpoints)
    nfft = np.power(2, tl.next_pow2_samples(window_size_sec, sampling_rate))
    ratio = (int(nfft / 2) + 1)/hz_points[-1]
    bin_f = np.round(hz_points * ratio)
    fbank = np.zeros((n_filters, int(np.floor(nfft / 2)+1)))

    for m in range(1, n_filters+1):
        f_m_minus = int(bin_f[m - 1])
        f_m = int(bin_f[m])
        f_m_plus = int(bin_f[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = 2 * (k - bin_f[m - 1]) / ((bin_f[m + 1] - bin_f[m - 1]) * (bin_f[m] - bin_f[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = 2 * (bin_f[m + 1] - k) / ((bin_f[m + 1] - bin_f[m - 1]) * (bin_f[m + 1] - bin_f[m]))
    
    fbank = np.where(fbank == 0, 10**-10, fbank)
    return fbank


def apply_mel_filters(abs_spectrum, filterbank):
   
    s_mel = np.zeros([len(filterbank), len(abs_spectrum)], dtype=float)
    
    for m in range(0, len(filterbank)):
        for r in range(0, len(abs_spectrum)):
            for k in range(0, len(abs_spectrum[0])):
                s_mel[m, r] += filterbank[m, k]*abs_spectrum[r, k]
    
    return s_mel


def compute_cepstrum(mel_spectrum, num_ceps):
    mel_spec = mel_spectrum.T
    rahmen_count = len(mel_spec)  # 271
    xcep = np.zeros([rahmen_count, num_ceps], dtype=float)
    
    for r in range(0, rahmen_count):  # 0 -270
        x = np.log10(mel_spec[r])
        xcep[r] = dct(x, norm='ortho')[0:num_ceps]

    return xcep


def get_delta(x):
    delta_x_cep = np.zeros([len(x), len(x[0])], dtype=float)
    for r in range(0, len(x)): 
        for t_ in range(0, len(x[0])):
            if r == 0:
                delta_x_cep[r, t_] = x[1, t_]-x[0, t_]
            elif r == len(x)-1:
                delta_x_cep[r, t_] = x[r, t_]-x[r-1, t_]
            else:
                delta_x_cep[r, t_] = 0.5 * (x[r+1, t_] - x[r-1, t_])
    
    return delta_x_cep


def append_delta(x, delta): 
    x = np.concatenate((x, delta), axis=1)
    return x


def add_context(feats, left_context=4, right_context=4):
    first_dim = feats.shape[0]
    second_dim = feats.shape[1]
    third_dim = left_context + right_context + 1
    extended_feats = np.zeros((first_dim, second_dim, third_dim))
    for i in range(first_dim):
        for j in range(second_dim):
            for k in range(left_context):
                if i-left_context+k < 0:
                    extended_feats[i][j][k] = feats[0][j]
                else:
                    extended_feats[i][j][k] = feats[i-left_context+k][j]
            extended_feats[i][j][left_context] = feats[i][j]
            for k in range(right_context):
                if i+k+1 > first_dim - 1:
                    extended_feats[i][j][k] = feats[-1][j]
                else:
                    extended_feats[i][j][k] = feats[i+k+1][j]
    #print("shape with context: " + str(np.shape(extended_feats)))
    return extended_feats
