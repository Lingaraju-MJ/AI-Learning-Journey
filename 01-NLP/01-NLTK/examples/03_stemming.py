from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

text = "I am learning NLP. She learned quickly. They learn every day."

words = word_tokenize(text)
stemmer = PorterStemmer()

stems = []
for word in words:
    stems.append(stemmer.stem(word))

print("Original text:")
print(text)

print("\nWord tokens:")
print(words)

print("\nStemmed words:")
print(stems)
