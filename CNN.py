import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np


class cnn_model:
    def __init__(self, *args, **kwargs):
        self.model = Sequential()
        self.num_classes = kwargs['num_class']
        self.create_model()

    def create_model(self):
        self.model.add(
            Conv2D(
                32,
                kernel_size=[5, 5],
                activation="relu",
                input_shape=(32, 32, 1)
            )
        )
        self.model.add(
            MaxPooling2D(
                pool_size=[2, 2],
            )
        )
        self.model.add(
            Conv2D(
                64,
                kernel_size=[5, 5],
            )
        )
        self.model.add(
            Flatten()
        )
        self.model.add(
            Dense(
                64,
                activation="relu"
            )
        )
        self.model.add(
            Dense(
                self.num_classes, activation='softmax'
            )
        )
