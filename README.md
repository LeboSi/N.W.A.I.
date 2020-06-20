# N.W.A.I.
Automatically Generating Rap Lyrics with Neural Networks.

# Introduction
Text generation is often based on a character approach. The latter does not allow us to obtain the rhythm of the text, useful when we want to generate poems for example (rhymes, alliterations...).

The aim of this project was to build a NLP solution for rhythm-focused text generation.

# Related Works
This project was inspired by the paper "Automatically Generate Rhythmic Verses with Neural Networks" by Jack Hopkins and Douwe Kiela, who described a phoneme based approach in order to generate rhythmic texts, with rhymes for instance.

# Dataset
The dataset is composed of US Rap Lyrics (Ice Cube, Eminem, Tupac, Dr Dre, Eazy-E, etc...) from 1990 to the present.
This dataset has the advantage to be:
- Consistent
- Provided (1,255,142 phonemes, 1,449,714 characters)
- Rhythmed (rimes, alliterations)

**Cleaning the dataset:**
- Removal of chorus repetition (this choice tends to be discussed, since chorus are part of rap lyrics, but we wanted to avoid the model to focus on it)
- Removal of comments in songs
- Removal of special characters (punctuation, ...)
- Writing numbers in letters

The validation set was created by hand, it permits to not break the text rhythm from the training text by doing so. It represents 25% of the dataset.

**Phoneme Encoding**
The text to phoneme encoding was done using the Carnegie Mellon University Dictionary.

- 134,000 words and their pronunciations 
- AA	    odd     AA D      (39 phonemes)
- 3 lexical accentuation markers :
  - 0 : No stress
  - 1 : Primary stress
  - 2 : Secondary stress
  
Some examples:
- Bird                B ER1 D
- Pusher           P UH1 SH ER0
- Pushers         P UH1 SH ER0 Z
- When             W EH1 N / HH W EH1 N / W IH1 N / HH W IH1 N

In case of an Out Of Vocabulary word, we use the most similar word in the CMU Dictionary.
