# Word Embeddings

Bag of Words and TF-IDF give each word its own column. `"nlp"` and `"nltk"` are just two different slots. The numbers do not show that those words are related.

A word embedding is a short list of numbers for each word. Words with a similar meaning should end up close to each other.

Last week I learned the idea:

- what a word embedding looks like
- CBOW
- Skip-gram

This week I am training a tiny Word2Vec model myself, then using AvgWord2Vec.

I am using small Python examples, one idea at a time. No extra libraries. gensim did not install on my Python version, so I wrote a small from-scratch example instead of calling a library.

## Word embedding

In `examples/01_word_embedding.py` I wrote a tiny table by hand.

`"nlp"`, `"nltk"`, and `"python"` have similar numbers. `"apple"` and `"mango"` have similar numbers too, but they sit in a different place.

I also print a simple distance. A smaller distance means the two words are closer.

These vectors are made up. The point is only to see the shape of an embedding before I train one.

## CBOW

CBOW means Continuous Bag of Words.

It looks at the words around a target word, then tries to guess that target word.

If the sentence is `I am learning NLP with simple Python examples` and the window is 2, then for the word `NLP` the context is `am learning` and `with simple`.

CBOW says: from those nearby words, predict `NLP`.

Example: `examples/02_cbow.py` prints those (context -> word) pairs for the whole sentence.

## Skip-gram

Skip-gram is the other way around.

It starts from one word and tries to guess the words around it.

For the same `NLP` example, skip-gram says: from `NLP`, predict `am`, `learning`, `with`, and `simple`.

That is why skip-gram makes more training pairs than CBOW. One middle word becomes many guesses.

Example: `examples/03_skipgram.py` uses the same sentence and window as the CBOW example, so I can compare them.

## Word2Vec

Word2Vec is the model. CBOW and skip-gram are the two ways it can learn.

I am using skip-gram: from one word, guess the nearby words. Each word starts as random numbers. When two words sit next to each other, I move their numbers a little closer.

I used two groups of sentences. NLP words keep showing up together. Fruit words keep showing up together. After training, `nlp` should be closer to `nltk` than to `apple`.

This is not the real neural-net Word2Vec. It is the same idea on a tiny set of sentences, so I can see the vectors change.

Example: `examples/04_word2vec.py`

## AvgWord2Vec

Word2Vec gives a vector to each word. Most of the time I care about a whole sentence.

AvgWord2Vec is just the average of the word vectors in that sentence.

`"I like NLP"` and `"I like NLTK"` should end up close. `"apple is fruit"` should sit somewhere else.

Example: `examples/05_avg_word2vec.py`

## What I understood

Learning in progress — I will add my notes after I run the examples.

## Next week

I will pick the next NLP topic after Word2Vec. FastText is one option. A small text classifier is another.
