from nltk.tokenize import word_tokenize
from nltk import ngrams

text = "I am learning NLP with simple Python examples."

words = word_tokenize(text)
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))

print("Original text:")
print(text)

print("\nWord tokens:")
print(words)

print("\nBigrams (2 words together):")
print(bigrams)

print("\nTrigrams (3 words together):")
print(trigrams)
