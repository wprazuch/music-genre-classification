import os
import pathlib
import pickle
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.image import resize
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

opj = os.path.join


def encode_labels(labels, new_encoding=False):
    if new_encoding:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        with open('label_encoder.pkl', 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open('label_encoder.pkl', 'rb') as handle:
                label_encoder = pickle.load(handle)
            labels = label_encoder.transform(labels)
        except:
            print("No existing label encoder found!")
            return None
    return labels


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(x.max()+1)[x]


def load_img_by_classes(input_path, no_images=None, image_size=None):
    data = []
    labels = []
    for class_dir in tqdm(os.listdir(input_path)):
        path = os.path.join(input_path, class_dir)
        if no_images is None:
            files = os.listdir(path)
        else:
            files = os.listdir(path)[:no_images]
        for file in files:
            img = load_img(pathlib.Path(path, file))
            img = img_to_array(img)
            if image_size is not None:
                img = resize(img, image_size)
            data.append(img)
            labels.append(class_dir)
    return data, labels


def preprocess_loaded_sequence_data(
        data, labels, sequence_length=None, new_encoding=True, num_classes=None):

    data = crop_sequences(data, sequence_length=sequence_length)
    data = np.array(data)

    data = tf.convert_to_tensor(data, dtype=tf.float32)

    labels = encode_labels(labels, new_encoding=new_encoding)

    labels = to_categorical(labels, num_classes)

    return data, labels


def preprocess_loaded_image_data(data, labels, image_size=(256, 256)):

    data = tf.convert_to_tensor(data, dtype=tf.float32)

    labels = encode_labels(labels, new_encoding=True)

    labels = to_categorical(labels)

    return data, labels


def load_data_by_classes(dataset_path):
    data = []
    labels = []
    classes = os.listdir(dataset_path)
    for cl in tqdm(classes):
        full_p = opj(dataset_path, cl)
        for file in os.listdir(full_p):
            for i in range(0, 20, 2):
                filepath = os.path.join(full_p, file)
                try:
                    wavedata, _ = librosa.load(filepath, sr=None, mono=True, offset=i, duration=2)
                except:
                    continue
                wavedata = wavedata[:, np.newaxis]
                data.append(wavedata)
                labels.append(cl)

    return data, labels


def crop_sequences(data, sequence_length=None):
    if sequence_length is None:
        min_size = min([len(item) for item in data])
        data = [item[:min_size] for item in data]
    else:
        data = [item[:sequence_length] for item in data]
    return data
