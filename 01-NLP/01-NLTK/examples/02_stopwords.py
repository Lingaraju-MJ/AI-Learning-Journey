from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "I am learning NLP with NLTK, and this is a simple example."

words = word_tokenize(text)
stop_words = stopwords.words("english")

filtered_words = []
for word in words:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print("Original text:")
print(text)

print("\nWord tokens:")
print(words)

print("\nSome English stopwords:")
print(stop_words[:15])

print("\nWords after removing stopwords:")
print(filtered_words)
