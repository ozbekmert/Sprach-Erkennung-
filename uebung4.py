import recognizer.feature_extraction as fe
import matplotlib.pyplot as plt
from scipy.io import wavfile

if __name__ == "__main__":
     plt.close('all')
     audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
     window_size = 0.025
     hop_size=0.01
     x1= fe.compute_features(audio_file, window_size, hop_size, feature_type='STFT', n_filters=24, fbank_fmin=0,fbank_fmax=8000)
     x2= fe.compute_features(audio_file, window_size, hop_size, feature_type='MFCC', n_filters=24, fbank_fmin=0,fbank_fmax=8000)
     x3= fe.compute_features(audio_file, window_size, hop_size, feature_type='MFCC_D', n_filters=24, fbank_fmin=0,fbank_fmax=8000)
     x4= fe.compute_features(audio_file, window_size, hop_size, feature_type='MFCC_D_DD', n_filters=24, fbank_fmin=0,fbank_fmax=8000)


     sampling_rate, audio_data = wavfile.read(audio_file)
     length = audio_data.shape[0] /sampling_rate


     plt.figure(1)
     plt.imshow(x2.T, interpolation='none', aspect='auto', origin='lower', extent=[0 , length, 0,  len(x2[0]) ] )
     plt.xlabel('Time in Seconds', fontsize=12)
     plt.ylabel('MFCC index', fontsize=12)
     plt.figure(2)
     plt.imshow(x3.T, interpolation='none', aspect='auto', origin='lower', extent=[0 , length, 0,  len(x3[0]) ] )
     plt.xlabel('Time in Seconds', fontsize=12)
     plt.ylabel('MFCC_D index', fontsize=12)
     plt.figure(3)
     im = plt.imshow(x4.T, interpolation='none', aspect='auto', origin='lower', extent=[0 , length, 0,  len(x4[0]) ] )
     plt.xlabel('Time in Seconds', fontsize=12)
     plt.ylabel('MFCC_D_DD index', fontsize=12)
     plt.colorbar(im)
     plt.show()
     
     
     
     
     
