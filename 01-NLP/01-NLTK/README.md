# NLTK Learning

I am learning Natural Language Processing (NLP) using NLTK.

NLTK is the Natural Language Toolkit. It is a Python library that helps us work with human language text. We can use it to split text, find word patterns, tag parts of speech, and try other basic NLP tasks.

I am following the NLTK book: https://www.nltk.org/book/ch00.html

## What I am learning

I am using small Python examples to understand basic NLP ideas, one concept at a time.

## Tokenization

Tokenization means splitting text into smaller pieces called tokens.

Computers do not automatically know where a word or a sentence starts and ends. Tokenization divides a long string of text into units we can count, compare, and process.

- Word tokenization splits text into words (and often punctuation).
- Sentence tokenization splits text into sentences.

Example: `examples/01_tokenization.py` shows basic word and sentence tokenization.

### NLTK data for tokenization

`word_tokenize` and `sent_tokenize` need a tokenizer model from NLTK.

If NLTK is not installed yet:

```
pip install nltk
```

Download the data once (current NLTK versions use `punkt_tab`):

```
python -m nltk.downloader punkt_tab
```

If you are using an older NLTK version and the command above does not work, try:

```
python -m nltk.downloader punkt
```

## Stopwords

Stopwords are common words that often add little meaning on their own, such as "the", "is", "and", and "I".

In many NLP examples, we remove stopwords so we can focus on the more important words. This is a choice for a task, not a rule that always applies.

Example: `examples/02_stopwords.py` tokenizes a short sentence, shows some English stopwords, and then removes them.

### NLTK data for stopwords

This example uses `word_tokenize` (same `punkt_tab` data as before) and the stopwords list.

Download the stopwords list once:

```
python -m nltk.downloader stopwords
```

## Stemming

Stemming cuts a word down to a shorter base form, called a stem.

The idea is that related words like "learn", "learning", and "learned" can be treated as the same word. Stemming uses simple rules. The stem is not always a real English word. For example, "quickly" may become "quickli".

Example: `examples/03_stemming.py` tokenizes a short text and then stems each word with NLTK's Porter stemmer.

### NLTK data for stemming

`PorterStemmer` does not need an extra data download. This example still uses `word_tokenize`, so you need the same `punkt_tab` data as in the tokenization example.

## POS tagging

POS means part of speech. Tagging assigns a label to each word, such as noun, verb, or adjective.

This helps later when we want to know *how* a word is used, not only what the word is. NLTK tags look like `NN` (noun), `VB` (verb), `JJ` (adjective), and `DT` (determiner). I do not need to memorize all of them at once.

Example: `examples/04_pos_tagging.py` tokenizes a short sentence and then tags each word.

### NLTK data for POS tagging

`pos_tag` needs a trained tagger. Current NLTK versions use `averaged_perceptron_tagger_eng`. You still need `punkt_tab` for `word_tokenize`.

```
python -m nltk.downloader averaged_perceptron_tagger_eng
```

If that fails on an older NLTK version, try:

```
python -m nltk.downloader averaged_perceptron_tagger
```

## Frequency distribution

A frequency distribution counts how often each word appears in the text.

After tokenization, we can see which tokens show up the most. Punctuation is counted too, because it is also a token.

Example: `examples/05_frequency_distribution.py` tokenizes a short text and prints word counts with NLTK's `FreqDist`.

### NLTK data for frequency distribution

`FreqDist` does not need an extra download. This example still uses `word_tokenize`, so you need `punkt_tab`.

## What I understood

Learning in progress — I will add my understanding after practicing the examples.

## Next topics

I moved on to text representation, then word embeddings.

See `../02-Text-Representation/` and `../03-Word-Embeddings/`.
