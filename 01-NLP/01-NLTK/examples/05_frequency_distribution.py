from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

text = "I like NLP. I like NLTK. NLP is fun to learn."

words = word_tokenize(text)
freq = FreqDist(words)

print("Original text:")
print(text)

print("\nWord tokens:")
print(words)

print("\nHow many times each word appears:")
print(freq)

print("\nMost common words:")
print(freq.most_common(5))
