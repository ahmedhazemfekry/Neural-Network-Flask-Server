
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

import logging
logging.getLogger('tensorflow').disabled = True

def Train(train_data,train_labels, val_data, val_labels, output_size):
    ML_model = tf.keras.Sequential([
    # Adds a densely-connected layer with units to the model:
    layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu', input_shape=(20,)),
    # Drop Out Layers
    layers.Dropout(0.3),
    # Add another:
    layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    # Drop Out Layers
    layers.Dropout(0.3),
    # Add a softmax layer with 43 output units:
    layers.Dense(output_size, activation='softmax')])

    ML_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    history = ML_model.fit(train_data, train_labels, callbacks=[es], epochs=300, batch_size=512,
              validation_data=(val_data, val_labels), verbose=2)
    ML_model.save("weights")
    return ML_model,history
