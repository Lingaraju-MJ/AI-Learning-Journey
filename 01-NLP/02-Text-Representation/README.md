# Text Representation

I am learning how to turn text into numbers.

In the NLTK folder I split text, removed stopwords, stemmed words, and counted them. That helped me see the words. A model still cannot read a sentence the way I do. It needs numbers.

Last week I did n-grams and Bag of Words. This week I did TF-IDF, then started word embeddings.

I am using small Python examples, one idea at a time.

NLTK is still useful for splitting text. For Bag of Words and TF-IDF I am using scikit-learn.

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

## TF-IDF

TF-IDF is still a table of numbers, like Bag of Words. The difference is that it does not only count words.

- TF (term frequency) is how often a word appears in one document.
- IDF (inverse document frequency) is how rare that word is across all the documents.

A word that shows up in every document gets a smaller score. A word that shows up in only one document gets a bigger score.

In these examples, `"like"` is in two sentences, so it is less special. `"fun"` is only in the third sentence, so it stands out more.

That is why TF-IDF is often a better starting point than raw counts.

Example: `examples/03_tfidf.py` uses the same three sentences as the Bag of Words example, so I can compare the two tables.

## What I understood

Learning in progress — I will add my notes after I run the examples.

## Next week

I moved on to word embeddings in this same week. CBOW and skip-gram are there too.

See `../03-Word-Embeddings/`.

After that, next week is Word2Vec training, including a small from-scratch example and AvgWord2Vec.
