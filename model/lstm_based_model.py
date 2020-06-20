"""This module contains the LSTM based model for phonemes generation"""

import tensorflow as tf
from model.one_hot import OneHot


def lstm_based_model(vocab_size, batch_size):
    tf_inputs = tf.keras.Input(shape=(None,), batch_size=batch_size)
    one_hot = OneHot(vocab_size)(tf_inputs)
    rnn_layer1 = tf.keras.layers.LSTM(256,
                                      return_sequences=True,
                                      stateful=True)(one_hot)
    dropout1 = tf.keras.layers.Dropout(0.5)(rnn_layer1)
    rnn_layer2 = tf.keras.layers.LSTM(256,
                                      return_sequences=True,
                                      stateful=True)(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.5)(rnn_layer2)
    hidden_layer = tf.keras.layers.Dense(256, activation="relu")(dropout2)
    outputs = tf.keras.layers.Dense(vocab_size,
                                    activation="softmax")(hidden_layer)
    return tf.keras.Model(inputs=tf_inputs, outputs=outputs)
