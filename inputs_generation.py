"""This module contains functions to generate the inputs."""

import numpy as np


def get_inputs(phonetic_text, phoneme_to_int):
    """
    Generates the inputs.

    Args:
        str: the phonetic text
        dict: the encoding dictionary
    Returns:
        np.array
    """
    encoded = np.array([phoneme_to_int[phoneme] for phoneme in phonetic_text])
    return encoded, encoded[1:]
