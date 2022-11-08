import math
import os

from keras.utils import np_utils
import keras
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import recognizer.tools as tl
import recognizer.feature_extraction as fe
import random
from sklearn.model_selection import train_test_split


def wav_to_posteriors(model, audio_file, parameters):
    features = fe.compute_features_with_context(audio_file=audio_file, **parameters)
    predictions = model.predict(features)
    return predictions


def generator(x_dirs, y_dirs, hmm, sampling_rate, parameters):
    window_size = parameters['window_size']
    hop_size = parameters['hop_size']
    rounded_window_size = int(math.pow(2, tl.next_pow2_samples(window_size, sampling_rate)))
    hop_size_samples = tl.sec_to_samples(hop_size, sampling_rate)
    i = 0
    while True:
        if i == len(x_dirs):

            i = 0
            xy = list(zip(x_dirs, y_dirs))

            random.shuffle(xy)

            x_dirs, y_dirs = zip(*xy)
        print(x_dirs[i])
        print(y_dirs[i])
        features = fe.compute_features_with_context(audio_file=x_dirs[i], **parameters)
        target = tl.praat_file_to_target(y_dirs[i], sampling_rate, rounded_window_size, hop_size_samples, hmm)

        i += 1

        yield features, target


def train_model(model, model_dir, x_dirs, y_dirs, hmm, sampling_rate, parameters, steps_per_epoch=10,
                epochs=10, viterbi_training=False):
    xy = list(zip(x_dirs, y_dirs))

    random.shuffle(xy)

    x_dirs, y_dirs = zip(*xy)
    x_train, x_val, y_train, y_val = train_test_split(x_dirs, y_dirs, test_size=0.33, shuffle=False)
    checkpoint_filepath = 'tmp/checkpoint'
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    model.fit(generator(x_train, y_train, hmm, sampling_rate, parameters),
              callbacks=[model_checkpoint_callback],
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              validation_data=generator(x_val, y_val, hmm, sampling_rate, parameters),
              validation_steps=10)
    model.load_weights(checkpoint_filepath)
    model.save(model_dir)


def dnn_model(input_shape, output_shape):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(output_shape, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
