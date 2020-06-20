import unidecode
import difflib
from phoneme_codec.phoneme_encoding import replacements, tokenize


def get_text(text_file="rap_2.0.txt"):
    """
    Get a text file.

    Args:
        str: the file's name containing the text
    Returns:
        str: the text
    """
    with open(text_file,
              "r",
              encoding="utf-8") as f:
        text = f.read()

    text = unidecode.unidecode(text)
    text = text.lower()


def get_phonemes(raw_phonetic_text_file, separator="#"):
    """
    Get a list of phonemes from a text file containing phonetic text.

    Args:
        str: the file's name containing the phonetic text
        str: the separator used in the text file to distinguish the phonemes
    Returns:
        list: the phonetic text
    """
    with open(raw_phonetic_text_file,
              "r",
              encoding="utf-8") as f:
        raw_phonetic_text = f.read()

    return raw_phonetic_text.split(separator)


def get_decoding_dictionary(cmu_dict):
    """
    Get a CMU decoding dictionary.

    Args:
        dict: the CMU dictionary
    Returns:
        dict: the CMU decoding dictionary
    """
    cmu_decode_dict = {}
    for key, value in cmu_dict.items():
        if len(value) >= 1:
            for e in value:
                cmu_decode_dict[str(e)].append(key)
    return cmu_decode_dict


def get_codec_dictionaries(phonetic_text):
    """
    Get encoding and decoding dictionaries for phonetic text.

    Args:
        list: the phonetic text
    Returns:
        dict: the encoding dictionary
        dict: the decoding dictionary
        int: the vocab length
    """
    vocab = set(phonetic_text)
    phoneme_to_int = {l: i for i, l in enumerate(vocab)}
    int_to_phoneme = {i: l for i, l in enumerate(vocab)}
    return phoneme_to_int, int_to_phoneme, len(vocab)


def decode(verses, int_to_phoneme, batch_size, cmu_dict, text):
    """
    Decode the generated verses.

    Args:

        dict: the integer to phoneme dictionary
        int: the batch_size
        dict: the CMU Dictionary
        str: the name of the text file containing the raw text data
    Returns:
    """
    cmu_decode_dict = get_decoding_dictionary(cmu_dict)
    words = tokenize(replacements(get_text(text)))
    for b in range(batch_size):

        t = [int_to_phoneme[i[0]] for i in verses[b]]
        new_words = []

        while t:

            new_phonemes = []

            while t and (t[0] != ' ') and (t[0] != '\n'):
                new_phonemes.append(t.pop(0))

            if str(new_phonemes) not in cmu_decode_dict.keys():
                new_phonemes = difflib.get_close_matches(new_phonemes,
                                                         cmu_dict.keys())
            if new_phonemes:
                list_words = cmu_decode_dict[str(new_phonemes)]
                counting_words = []
                for word in list_words:
                    if word in words:
                        counting_words.append(words.count(word))
                    else:
                        counting_words.append(0)
                    new_words.append(cmu_decode_dict[str(new_phonemes)]
                                     [counting_words.index(max(counting_words))
                                      ])

            if t:
                new_words.append(t.pop(0))

        print("".join(new_words))
        print("\n=====================\n")