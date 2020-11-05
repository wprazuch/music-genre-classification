import numpy as np
import tensorflow as tf
import os

import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from ml_music import utils
from ml_music.models import common


opj = os.path.join

NUM_CLASSES = 10

gtzan_path = r'datasets\gtzan'
genres_dir = opj(gtzan_path, 'genres_original')


def main():

    data, labels = utils.load_data_by_classes(genres_dir)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    X_train, y_train = utils.preprocess_loaded_sequence_data(
        X_train, y_train, sequence_length=100000, new_encoding=True, num_classes=NUM_CLASSES)
    X_val, y_val = utils.preprocess_loaded_sequence_data(
        X_val, y_val, sequence_length=100000, new_encoding=False, num_classes=NUM_CLASSES)
    X_test, y_test = utils.preprocess_loaded_sequence_data(
        X_test, y_test, sequence_length=100000, new_encoding=False, num_classes=NUM_CLASSES)

    model = common.LSTM_XS(NUM_CLASSES, X_train.shape[1:])

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(
        learning_rate=0.0001), metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=24, batch_size=8)


if __name__ == '__main__':

    main()
