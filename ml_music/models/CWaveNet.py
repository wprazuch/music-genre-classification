from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout, MaxPooling1D, Conv1D, Flatten, Activation
import tensorflow as tf


class CWaveNet(Model):

    def __init__(self, inp_shape, kernel_size, activation='relu'):
        super(CWaveNet, self).__init__()

        self.kernel_size = kernel_size
        self.actication = 'relu'
        self.inp_shape = inp_shape

        self.conv1_1 = Conv1D(32, kernel_size, input_shape=self.inp_shape, activation='relu')
        self.conv1_2 = Conv1D(32, kernel_size, activation='relu')
        # model.add(BatchNormalization())
        self.mp1 = MaxPooling1D(pool_size=4)

        self.conv2_1 = Conv1D(32, kernel_size, activation='relu')
        # model.add(BatchNormalization())
        self.mp2 = MaxPooling1D(pool_size=4)

        self.flatten = Flatten()
        self.fc1 = Dense(100, activation='relu')
        self.drop1 = Dropout(0.5)
        # model.add(BatchNormalization())
        self.out = Dense(10, activation='softmax')

        self.out_act = Activation('softmax')

    def call(self, inputs, training=False):

        x = self.conv1_2(inputs)
        x = self.conv1_2(x)
        x = self.mp1(x)

        x = self.conv2_1(x)
        x = self.mp2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.drop1(x)
        x = self.out(x)
        x = self.out_act(x)

        return x
