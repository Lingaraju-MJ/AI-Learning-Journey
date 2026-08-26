from nltk.tokenize import word_tokenize
from nltk import pos_tag

text = "NLTK helps me learn NLP with simple Python examples."

words = word_tokenize(text)

# each item is (word, tag) like ('NLTK', 'NNP')
tagged_words = pos_tag(words)

print("Original text:")
print(text)

print("\nWord tokens:")
print(words)

print("\nPOS tags:")
print(tagged_words)
