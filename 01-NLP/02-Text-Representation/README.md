# Text Representation

I am learning how to turn text into numbers.

In the NLTK folder I split text, removed stopwords, stemmed words, and counted them. That helped me see the words. A model still cannot read a sentence the way I do. It needs numbers.

This week I am doing n-grams and Bag of Words. Next week I will do TF-IDF.

I am using small Python examples, one idea at a time.

NLTK is still useful for splitting text. For Bag of Words I am using scikit-learn.

If it is not installed yet:

```
pip install scikit-learn
```

The n-grams example still uses NLTK, so I also need NLTK and the `punkt_tab` data from the earlier folder.

## N-grams

An n-gram is just n words that sit next to each other.

- unigram = 1 word (`learning`)
- bigram = 2 words (`learning NLP`)
- trigram = 3 words (`am learning NLP`)

Bag of Words only looks at single words, so it throws away order. That is why `"not good"` and `"good"` can look too similar. N-grams keep a little bit of that order.

Example: `examples/01_ngrams.py` tokenizes a short sentence and prints bigrams and trigrams.

This example uses `word_tokenize`, so I need the same `punkt_tab` data as in the NLTK tokenization example.

## Bag of Words

Bag of Words turns each document into a list of word counts.

First it builds a vocabulary from all the documents. Then each document becomes a row of numbers. Each number is how many times that word appears.

It is called a bag because the order is thrown away. `"I like NLP"` and `"NLP like I"` get the same counts.

`CountVectorizer` also lowercases the text. It skips very short tokens like `"I"`, so that word will not show up in the table.

Example: `examples/02_bag_of_words.py` uses the same kind of short sentences as the frequency distribution example. Then it does the same thing again with bigrams, so I can see the extra columns.

## What I understood

Learning in progress — I will add my notes after I run the examples.

## Next week

TF-IDF with `TfidfVectorizer`. Same idea as Bag of Words, but the numbers show how important a word is, not only how often it appears.
