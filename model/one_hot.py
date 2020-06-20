"""This module contains the One Hot class for One hot encoding."""

import tensorflow as tf


class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
        })
        return config
