"""This module regroups the training functions."""

import numpy as np
import tensorflow as tf
import argparse
import inputs_generation
import phoneme_codec.phoneme_decoding as phoneme_decoding
import phoneme_codec.phoneme_encoding as phoneme_encoding
from model.lstm_based_model import lstm_based_model


def gen_batch(inputs, targets, seq_len, batch_size, vocab_size, noise=0):
    # Size of each chunk
    chuck_size = (len(inputs) - 1) // batch_size
    # Numbef of sequence per chunk
    sequences_per_chunk = chuck_size // seq_len

    for s in range(0, sequences_per_chunk):
        batch_inputs = np.zeros((batch_size, seq_len))
        batch_targets = np.zeros((batch_size, seq_len))
        for b in range(0, batch_size):
            fr = (b * chuck_size) + (s * seq_len)
            to = fr + seq_len
            batch_inputs[b] = inputs[fr:to]
            batch_targets[b] = inputs[fr + 1:to + 1]
            if noise > 0:
                noise_indices = np.random.choice(seq_len, noise)
                batch_inputs[b][noise_indices] = np.random.randint(0,
                                                                   vocab_size)

        yield batch_inputs, batch_targets


@tf.function
def train_step(inputs, targets, model, loss_object, train_loss,
               train_accuracy, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(targets, predictions)


@tf.function
def val_step(val_inputs, val_targets, model, loss_object,
             val_loss, val_accuracy):
    with tf.GradientTape() as tape:
        val_predictions = model(val_inputs)
        loss = loss_object(val_targets, val_predictions)
    val_loss(loss)
    val_accuracy(val_targets, val_predictions)


@tf.function
def predict(inputs, model):
    predictions = model(inputs)
    return predictions


def train(model, batch_size, seq, inputs, targets,
          val_inputs, val_targets, vocab_size):
    """
    Train the model.

    Args:
        tf.keras.Model: the model to train
        int: the batch size
        int: the sequence length
        np.array: the inputs
        np.array: the targets
        np.array: the validation inputs
        np.array: the validation targets
    Returns:
        tf.keras.Model
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.\
        SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.\
        SparseCategoricalAccuracy(name='val_accuracy')

    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(20):

        for batch_inputs, batch_targets in gen_batch(inputs,
                                                     targets,
                                                     seq,
                                                     batch_size,
                                                     vocab_size):
            train_step(batch_inputs, batch_targets, model,
                       loss_object, train_loss, train_accuracy,
                       optimizer)

        for val_batch_inputs, val_batch_targets in gen_batch(val_inputs,
                                                             val_targets,
                                                             seq,
                                                             batch_size,
                                                             vocab_size):
            val_step(val_batch_inputs, val_batch_targets, model,
                     loss_object, train_loss, train_accuracy,
                     optimizer)

        accuracies.append(train_accuracy.result())
        losses.append(train_loss.result())
        val_accuracies.append(val_accuracy.result())
        val_losses.append(val_loss.result())

        model.reset_states()


def generate_verses(model, batch_size, int_to_phoneme, cmu_dict, text):
    model.reset_states()

    size_verses = 300

    verses = np.zeros((batch_size, size_verses, 1))
    sequences = np.zeros((batch_size, 100))
    for b in range(batch_size):
        rd = np.random.randint(0, len(inputs) - 100)
        sequences[b] = inputs[rd:rd+100]

    for i in range(size_verses+1):
        if i > 0:
            verses[:, i-1, :] = sequences
        softmax = predict(sequences)
        # Set the next sequences
        sequences = np.zeros((batch_size, 1))
        for b in range(batch_size):
            argsort = np.argsort(softmax[b][0])
            argsort = argsort[::-1]
            # Select one of the strongest 4 proposals
            sequences[b] = argsort[0]
    phoneme_decoding.decode(verses,
                            int_to_phoneme,
                            batch_size,
                            cmu_dict,
                            text)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_phonetic_text',
    default='./data/phonetic_rap_2.0.txt',
    help=".txt file containing the raw phonetic text for training"
)
parser.add_argument(
    '--val_phonetic_text',
    default='./data/val_phonetic_rap_2.0.txt',
    help='.txt file containing the raw phonetic text for validation'
)

parser.add_argument(
    '--raw_training_text',
    default='./data/rap_2.0.txt',
    help="Directory with processed dataset"
)

parser.add_argument(
    '--batch_size',
    default=32,
    help="Batch Size"
)

parser.add_argument(
    '--seq',
    default=500,
    help="Sequence length"
)

if __name__ == '__main__':

    args = parser.parse_args()

    cmu_dict = phoneme_encoding.create_CMU_encoding_dictionary()
    phonetic_text = phoneme_decoding.get_phonemes(args.train_phonetic_text)
    val_phonetic_text = phoneme_decoding.get_phonemes(args.val_phonetic_text)
    phoneme_to_int,\
        int_to_phoneme,\
        vocab_size = phoneme_decoding.get_codec_dictionaries(phonetic_text)
    inputs, targets = inputs_generation.get_inputs(phonetic_text,
                                                   phoneme_to_int)
    val_inputs, val_targets = inputs_generation.\
        get_inputs(val_phonetic_text, phoneme_to_int)
    BATCH_SIZE = args.batch_size
    SEQ = args.seq
    model = lstm_based_model(vocab_size, BATCH_SIZE)
    train(model, BATCH_SIZE, SEQ, inputs, targets,
          val_inputs, val_targets, vocab_size)
    generate_verses(model,
                    BATCH_SIZE,
                    int_to_phoneme,
                    cmu_dict,
                    args.raw_training_text)
