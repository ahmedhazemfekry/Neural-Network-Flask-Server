
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

import logging
logging.getLogger('tensorflow').disabled = True

def Train_1(train_data,train_labels, val_data, val_labels, output_size):
    ML_model = tf.keras.Sequential([
    # Adds a densely-connected layer with units to the model:
    layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu', input_shape=(160,)),
    # Drop Out Layers
    layers.Dropout(0.2),
    # Add another:
    layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    # Drop Out Layers
    layers.Dropout(0.1),
    # Add a softmax layer with output units:
    layers.Dense(output_size, activation='softmax')])

    ML_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    history = ML_model.fit(train_data, train_labels, callbacks=[es], epochs=600, batch_size=512,
              validation_data=(val_data, val_labels), verbose=2)
    ML_model.save("weights")
    return ML_model,history

def Train_2(train_data,train_labels, val_data, val_labels, output_size):
    ML_model = tf.keras.Sequential([
    # Adds a densely-connected layer with units to the model:
    layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu', input_shape=(400,)),
    # Drop Out Layers
    layers.Dropout(0.3),
    # Add another:
    layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    # Drop Out Layers
    layers.Dropout(0.2),
    # Add a softmax layer with output units:
    layers.Dense(output_size, activation='softmax')])

    ML_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    history = ML_model.fit(train_data, train_labels, callbacks=[es], epochs=600, batch_size=1024,
              validation_data=(val_data, val_labels), verbose=2)
    ML_model.save("weights")
    return ML_model,history

def Train_3(train_data,train_labels, val_data, val_labels, output_size):
    ML_model = tf.keras.Sequential([
    # Adds a densely-connected layer with units to the model:
    layers.Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu', input_shape=(400,)),
    # Drop Out Layers
    layers.Dropout(0.3),
    # Add another:
    layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    # Drop Out Layers
    layers.Dropout(0.2),
    # Add another:
    layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    # Drop Out Layers
    layers.Dropout(0.2),
    # Add a softmax layer with output units:
    layers.Dense(output_size, activation='softmax')])

    ML_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    history = ML_model.fit(train_data, train_labels, callbacks=[es], epochs=300, batch_size=512,
              validation_data=(val_data, val_labels), verbose=2)
    ML_model.save("weights")
    return ML_model,history
