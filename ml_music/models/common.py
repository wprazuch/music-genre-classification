from tensorflow.keras import applications
import importlib

from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import layers, Model


def ResNet50V2(num_classes: int, image_size=(256, 256)):

    base_model = applications.ResNet50V2(
        weights=None, include_top=False, input_shape=(*image_size, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
