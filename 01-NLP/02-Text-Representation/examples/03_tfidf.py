from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "I like NLP",
    "I like NLTK",
    "NLP is fun to learn"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(texts)
scores = tfidf.toarray()

print("Documents:")
for text in texts:
    print("-", text)

print("\nWords in the vocabulary:")
print(vectorizer.get_feature_names_out())

# round so the long decimals are easier to read
print("\nTF-IDF scores:")
for i, text in enumerate(texts):
    rounded = [round(float(n), 2) for n in scores[i]]
    print(text, "->", rounded)
