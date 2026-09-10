# Word Embeddings

Bag of Words and TF-IDF give each word its own column. `"nlp"` and `"nltk"` are just two different slots. The numbers do not show that those words are related.

A word embedding is a short list of numbers for each word. Words with a similar meaning should end up close to each other.

This week I am only learning the idea:

- what a word embedding looks like
- CBOW
- Skip-gram

I am not training a model yet. Next week I will do Word2Vec, train a small model, and look at AvgWord2Vec.

I am using small Python examples, one idea at a time. No extra libraries for these three files.

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

## What I understood

Learning in progress — I will add my notes after I run the examples.

## Next week

Word2Vec (this is the model that uses CBOW or skip-gram), training a small Word2Vec model, and AvgWord2Vec.
