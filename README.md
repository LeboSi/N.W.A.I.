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

# Network architecture and training
## Architecture
We used an LSTM model which predicts for a character (or in our case phoneme) and a sequence of previous characters the next character.

![model](https://github.com/jdelaunay/N.W.A.I./blob/master/ressources/img/model_architecture.png)

## Training

- Loss : Crossentropy
- Metric : Accuracy
- Optimizer : Adam
- Learning Rate = 0.001
We trained the model with a Google Colaboratory.

By 25 epochs, the model converges.

# Results
Some examples of verses we obtained with our model :

### Example 1:
> Friends  money   from the city of a **bitch**

> Life gold digger and who ass nigger  be **teach**

> And you  seen your  from the **park**

> Little scrap  i got my **stacks**

> And  seeing my city  still on the seber street

> Like ha said **like**

> When  while life is **high**

### Example 2:
> Six six drinks in the **bend**

> But i got my friends and the crib **and**

> And now that i was like to **stranger**

> I can tell them to come to me

> Sipping on that pussy ass **nigger**

### Example 3:
> In the car we gonna go down that we can see

> This is for the fifty five with the **liquor**

> The bigger was the strips to the **floor**

> And the pride when i was the same with the black

> With the bitch she was still a little streets

As we can see, the phoneme level approach permits to obtain some rimes, which is satisfying since we didn't intend to generate lyrics which had meaning. It could be very interesting however to combine this approach with another one focused on the meaning of the generated verses.
