from keras.datasets import fashion_mnist
from keras.utils import np_utils
from tensorflow import keras
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


warnings.filterwarnings('ignore', category=FutureWarning)


def get_data():
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    trainY = np_utils.to_categorical(trainY, 10)
    testY = np_utils.to_categorical(testY, 10)
    label_names = ["top", "trouser", "pullover", "dress", "coat","sandal", "shirt", "sneaker", "bag", "ankle boot"]
    data_dict = {"train" : {"image" : trainX,"label" : trainY}
        ,"test": {"image" : testX,"label" : testY},"labels": label_names}
    return data_dict


def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.summary()
    return model


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    label_names = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    modelpath = Path('fmnist_ff.h5')
    if not modelpath.exists():

        fig = plt.figure()
        for i in range(12):
            ax = fig.add_subplot(3, 4, i + 1)
            ax.imshow(x_train[i, :, :], cmap='gist_gray')
            ax.set_title(label_names[y_train[i]], fontsize=12)

        plt.subplots_adjust(hspace=0.4)
        # plt.show()

        data = get_data()
        model = get_model()

        epochs = 10
        batch_size = 20
        loss_hist = model.fit(data["train"]["image"], data["train"]["label"], epochs=epochs, batch_size=batch_size,
                             validation_data=(data["test"]["image"], data["test"]["label"]))
        model.save('fmnist_ff.h5')

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, epochs), loss_hist.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), loss_hist.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), loss_hist.history["accuracy"], label="train_accuracy")
        plt.plot(np.arange(0, epochs), loss_hist.history["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")
        plt.show()

    model = keras.models.load_model('fmnist_ff.h5')
    # model = keras.Sequential([model, keras.layers.Softmax(1)])

    testX = x_test.astype("float32") / 255.0
    predictions = model.predict(testX)

    fig = plt.figure()
    for i in range(1, 7):
        mod = 1600
        ax = fig.add_subplot(3, 4, i * 2 - 1)
        ax.imshow(x_test[i + mod, :, :], cmap='gist_gray')
        ax2 = fig.add_subplot(3, 4, i * 2)
        ax2.barh(label_names, predictions[i + mod])
        print(predictions[i + mod])

    plt.show()
