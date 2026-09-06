from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "I like NLP",
    "I like NLTK",
    "NLP is fun to learn"
]

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(texts)
counts = bow.toarray()

print("Documents:")
for text in texts:
    print("-", text)

print("\nWords in the vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nBag of Words (how many times each word appears):")
for i, text in enumerate(texts):
    print(text, "->", counts[i])

# ngram_range=(1, 2) keeps single words and pairs of words
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_bow = bigram_vectorizer.fit_transform(texts)
bigram_counts = bigram_bow.toarray()

print("\nVocabulary with unigrams and bigrams:")
print(bigram_vectorizer.get_feature_names_out())

print("\nBag of Words with bigrams:")
for i, text in enumerate(texts):
    print(text, "->", bigram_counts[i])
