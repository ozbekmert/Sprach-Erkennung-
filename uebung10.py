import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
from keras.models import load_model
import numpy as np
import argparse
import os
from collections import defaultdict
import recognizer.tools as tl
import random


def test_model(datadir, hmm, model_dir, parameters):
    test_dir_path = datadir + '/TEST/wav/'
    lab_dir_path = datadir + '/TEST/lab/'
    test_dir = os.listdir(test_dir_path)
    lab_dir = os.listdir(lab_dir_path)
    
    lab_dict = defaultdict(list)
    
    for i in range(len(test_dir)):
        test_dir[i] = os.path.join(test_dir_path, test_dir[i])
    
    for i in range(len(lab_dir)):
        lab_dir[i] = os.path.join(lab_dir_path, lab_dir[i])
        a = open(lab_dir[i]).read().split(" ")
        a = [x.lower() for x in a]
        lab_dict[lab_dir[i]] = a[:-1]
        
        
    model = load_model('exp/dnn.h5')
    N_ges = I_ges = S_ges = D_ges = 0
    i = 0
    mistake_files = []
    random.shuffle(test_dir)
    for audio_file in test_dir:
        posteriors = rec.wav_to_posteriors(model, audio_file, parameters)
        word_seq = hmm.posteriors_to_transcription(posteriors)    
        ref_seq = lab_dict[audio_file.replace('wav', 'lab')]
        N, D, I, S = tl.needlemann_wunsch(ref_seq, word_seq)
        if D != 0 or I != 0 or S != 0:
            mistake_files.append(audio_file)
            print('--' * 40)
            print('REF: ')
            print(ref_seq)
            print('OUT: ')
            print(word_seq)
            print("N: " + str(N) +" D: " + str(D) + " I: " + str(I) + "  S: " + str(S))
        N_ges += N
        D_ges += D
        I_ges += I
        S_ges += S
        
        i +=1
        if i == 200:
            break
        
        
    wer = (D_ges + I_ges + S_ges)/N_ges
    print('--' * 40)
    print('mistake_files: ')
    print(mistake_files)
    return wer

if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung10.py <data/dir>
    # e.g., python uebung11.py /media/public/TIDIGITS-ASE
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str, help='Data dir')
    args = parser.parse_args()

    # parameters for the feature extraction
    parameters = {'window_size': 25e-3,
        'hop_size': 10e-3,
        'feature_type': 'MFCC_D_DD',
        'n_filters': 24,
        'fbank_fmin': 0,
        'fbank_fmax': 8000,
        'num_ceps': 13,
        'left_context': 6,
        'right_context': 6}

    # default HMM
    hmm = HMM.HMM()

    # define a name for the model, e.g., 'dnn'
    model_name = 'dnn'
    # directory for the model
    model_dir = os.path.join('exp', model_name + '.h5')

    # test DNN
    wer = test_model(args.datadir, hmm, model_dir, parameters)
    print('--' * 40)
    print("Total WER: {}".format(wer))
