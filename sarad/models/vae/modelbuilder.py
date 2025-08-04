from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Reshape
import tensorflow.keras.backend as K
import os.path
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Dense, Input, Lambda, Reshape
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Dropout, Input, Reshape, Conv2DTranspose, Layer
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU

class Sampling(Layer):
    """Sampling layer using (mean, log_var)"""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class ModelBuilder:
    def __init__(self):
        pass

    def build_model(self, X_train, n_layers=7, filters = 64, latent_dim = 64):
        ### build model
        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)
        x = inputs

        # ENCODER
        filters = filters
        n_layers = n_layers
        latent_dim = latent_dim

        for _ in range(n_layers):
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            filters *= 2  # grow deeper layers

        shape_before_flattening = K.int_shape(x)[1:]
        x_flat = Flatten()(x)

        # LATENT SPACE
        mean = Dense(latent_dim, name="z_mean")(x_flat)
        log_var = Dense(latent_dim, name="z_log_var")(x_flat)
        z = Sampling()([mean, log_var])

        # DECODER
        x = Dense(np.prod(shape_before_flattening))(z)
        x = Reshape(target_shape=shape_before_flattening)(x)

        filters //= 2
        for _ in range(n_layers):
            x = Conv2DTranspose(filters, (3, 3), strides=2, activation='relu', padding='same')(x)
            filters //= 2

        outputs = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
        return model


    def train_model(self, model, X_train, X_val, batch_size=8, epochs=35):
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        batch_size = batch_size
        epochs = epochs
        return model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )

