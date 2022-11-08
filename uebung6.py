import recognizer.tools as tl
import recognizer.dnn_recognizer as rec
import recognizer.hmm as HMM
import argparse
import os
from keras.models import load_model
import matplotlib.pyplot as plt


def train_model(datadir, hmm, model_dir, parameters, epochs):
    train_dir_x_path = datadir + '/TRAIN/wav/'
    train_dir_x = os.listdir(train_dir_x_path)
    for i in range(len(train_dir_x)):
        train_dir_x[i] = os.path.join(train_dir_x_path, train_dir_x[i])

    train_dir_y_path = datadir + '/TRAIN/TextGrid/'
    train_dir_y = os.listdir(train_dir_y_path)
    for j in range(len(train_dir_y)):
        train_dir_y[j] = os.path.join(train_dir_y_path, train_dir_y[j])

    input_shape = (parameters['num_ceps'] * 3, parameters['left_context'] + parameters['right_context'] + 1)
    #print(input_shape)
    output_shape = hmm.get_num_states()

    model = rec.dnn_model(input_shape, output_shape)
    # steps = len(train_dir_x)
    steps = 256
    rec.train_model(model, model_dir, train_dir_x, train_dir_y, hmm,
                    sampling_rate=16000, parameters=parameters, steps_per_epoch=steps, epochs=epochs)


if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /my_path/TIDIGITS-ASE
    # call:
    # python uebung6.py <data/dir>
    # e.g., python uebung6.py /my_path/TIDIGITS-ASE
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
                  'right_context': 6,
                  }

    # number of epoches (not too many for the beginning)
    epochs = 3

    # define a name for the model, e.g., 'dnn'
    model_name = 'dnn'
    # directory for the model
    model_dir = os.path.join('exp', model_name + '.h5')
    if not os.path.exists('exp'):
        os.makedirs('exp')

    # default HMM
    hmm = HMM.HMM()

    # train DNN
    train_model(args.datadir, hmm, model_dir, parameters, epochs)
    model = load_model(os.path.join('exp', model_name + '.h5'))

    # get posteriors for test file
    post = rec.wav_to_posteriors(model, 'data/TEST-MAN-AH-3O33951A.wav', parameters)
    plt.imshow(post.transpose(), origin='lower')
    plt.xlabel('Frames')
    plt.ylabel('HMM states')
    plt.colorbar()
    plt.show()