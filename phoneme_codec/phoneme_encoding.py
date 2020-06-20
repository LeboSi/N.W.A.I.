"""This module regroups the functions for phoneme encoding.

This module regroups the functions necessary for encoding a text into
phonemes using the Carnegie Mellon University Dictionary.
"""

import re
import unidecode
import cmudict
import difflib
from num2words import num2words


def replacements(text):
    """Replace characters and somes keys out-of-vocabulary words.

    Args:
        str: the input text
    Returns:
        str: the processed text
    """
    text = unidecode.unidecode(text)
    text = text.lower()
    for i in range(2000, 0, -1):
        text = text.replace(str(i), num2words(i))
    text = text.replace("nigga", "nigger")
    text = text.replace("2pac", "tupac")
    text = text.replace("A$AP", "ASAP")
    text = text.replace("é", "e")
    text = text.replace("è", "e")
    text = text.replace("ë", "e")
    text = text.replace("&", " and ")
    text = text.replace("*laughing*", "")
    text = text.replace("~", " ")
    text = text.replace(">", "")
    text = text.replace("<", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("-", " ")
    text = text.replace("$", "")
    text = text.replace(";", "")
    text = text.replace(":", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("'", "'")
    text = text.replace('"', '')
    text = text.replace(",", "")
    text = text.replace(".", " ")
    text = text.replace("--", " ")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("|", "")
    text = text.replace("+", " ")
    text = text.replace("*", "")
    text = text.replace(" \n", "\n")
    text = text.strip()
    return text


def tokenize(text):
    """Tokenize the text word by word.

    Args:
        str: input text
    Returns:
        list: list of the tokenized words and line breaks
    """
    return re.split("(\W)", text)


def create_CMU_encoding_dictionary():
    """
    Create a encoding CMU Dictionary.

    Returns:
        dict: CMU encoding Dictionary
    """
    return cmudict.dict()


def get_closest_phonetic_word(word, cmu_dict):
    """
    Return the closest phonetic word in the CMU dictionnary.

    Args:
        str: input word,
        dict: CMU Dictionary
    Returns:
        str: the closest phonetic word from the OOV input word in the CMU
        Dictionary
    """
    w_list = difflib.get_close_matches(word, cmu_dict.keys())
    for i in range(len(w_list)):
        new_w = w_list[i]
        phonetic_word = cmu_dict[new_w]
        if phonetic_word:
            phonetic_word = phonetic_word[0]
            break
    return phonetic_word


def generate_phonetic_text(words, cmu_dict):
    """
    Generate the phonetic text.

    Args:
        list: tokenized text
        dict: the CMU encoding Dictionary
    Returns:
        list: tokenized phonetic text
    """
    phonetic_text = []
    for word in words:
        if word == ' ' or word == '\n':
            phonetic_word = word
        else:
            phonetic_word = cmu_dict[word]
            if len(phonetic_word) >= 1:
                phonetic_word = phonetic_word[0]
            else:
                get_closest_phonetic_word(word, cmu_dict)
        for phoneme in phonetic_word:
            phonetic_text.append(phoneme)

    return phonetic_text
