import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
from keras.models import load_model
import numpy as np


if __name__ == "__main__":

    # die Parameter für die Merkmalsextraktion
    parameters = {'window_size': 25e-3,
        'hop_size': 10e-3,
        'feature_type': 'MFCC_D_DD',
        'n_filters': 24,
        'fbank_fmin': 0,
        'fbank_fmax': 8000,
        'num_ceps': 13,
        'left_context': 6,
        'right_context': 6}

    # das Hidden-Markov-Modell 
    hmm = HMM.HMM()


    # 1.) Test mit vorgegebenen Daten
    # die Zustandswahrscheinlichkeiten passend zum HMM aus UE6
    posteriors = np.load('data/TEST-MAN-AH-3O33951A.npy')

    # Transkription für die vorgegebenen Wahrscheinlichkeiten
    words = hmm.posteriors_to_transcription(posteriors)
    print('OUT: {}'.format(words))            # OUT: ['THREE', 'OH', 'THREE', 'THREE', 'NINE', 'FIVE', 'ONE']   



    # 2.) Test mit gegebenen Daten und Ihrer Merkmalsextraktion
    test_audio = 'data/TEST-MAN-AH-3O33951A.wav'

    # Laden eines Trainierten DNNs (z.B. aus UE6)
    model = load_model('exp/dnn.h5')
    
    posteriors = rec.wav_to_posteriors(model, test_audio, parameters)
    words = hmm.posteriors_to_transcription(posteriors)
    
    print('OUT: {}'.format(words))            # OUT: ['THREE', 'OH', 'THREE', 'THREE', 'NINE', 'FIVE', 'ONE']   
