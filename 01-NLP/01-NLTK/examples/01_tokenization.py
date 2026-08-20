from nltk.tokenize import word_tokenize, sent_tokenize

text = "I am learning NLP with NLTK. Tokenization splits text into smaller pieces!"

words = word_tokenize(text)
sentences = sent_tokenize(text)

print("Original text:")
print(text)

print("\nWord tokens:")
print(words)

print("\nSentence tokens:")
print(sentences)
