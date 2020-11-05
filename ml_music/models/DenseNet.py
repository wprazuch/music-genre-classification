from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout
import tensorflow as tf


class DenseBlock(layers.Layer):
    def __init__(self, no_neurons, dropout_rate=0.5, activation='relu'):

        super(DenseBlock, self).__init__()

        self.fc = layers.Dense(no_neurons, activation=activation)
        self.drop = layers.Dropout(dropout_rate)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.fc(inputs)
        x = self.drop(x, training=training)
        x = self.bn(x, training=training)

        return x


class DenseNet(Model):

    def __init__(
            self, input_shape, no_classes, layer_sizes=[512, 512, 512],
            dropout_rate=0.5, activation='relu'):
        super(DenseNet, self).__init__()

        self.inp = layers.Input(shape=(None, input_shape))

        self.dense_blocks = []

        for no_neurons in layer_sizes:
            d_block = DenseBlock(no_neurons=no_neurons,
                                 dropout_rate=dropout_rate, activation=activation)

            self.dense_blocks.append(d_block)

        self.out_dense = Dense(no_classes)

    def call(self, inputs, training=False):

        x = inputs

        for dense_block in self.dense_blocks:
            x = dense_block(x, training=training)

        x = self.out_dense(x)

        if not training:
            x = tf.nn.softmax(x)

        return x
