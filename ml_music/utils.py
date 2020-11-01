import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.preprocessing.image import load_img
import pathlib
from pathlib import Path
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder


def encode_labels(labels, new_encoding=False):
    if new_encoding:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        with open('label_encoder.pkl', 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open('label_encoder.pickle', 'rb') as handle:
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


def preprocess_loaded_data(data, labels, image_size=(256, 256)):

    data = tf.convert_to_tensor(data, dtype=tf.float32)

    labels = encode_labels(labels, new_encoding=True)

    labels = to_categorical(labels)

    return data, labels
