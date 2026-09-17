# a sentence has many words. AvgWord2Vec is just the average of those word vectors.

embeddings = {
    "i": [0.2, 0.1, 0.1],
    "like": [0.7, 0.2, 0.1],
    "nlp": [0.9, 0.2, 0.1],
    "nltk": [0.8, 0.3, 0.1],
    "apple": [0.1, 0.8, 0.7],
    "is": [0.3, 0.3, 0.3],
    "fruit": [0.1, 0.9, 0.6],
}


def avg_word2vec(text):
    words = text.lower().split()
    vecs = []
    for word in words:
        if word in embeddings:
            vecs.append(embeddings[word])

    size = len(vecs[0])
    avg = []
    for i in range(size):
        total = 0
        for vector in vecs:
            total = total + vector[i]
        avg.append(round(total / len(vecs), 2))
    return avg


def distance(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + (x - y) ** 2
    return round(total ** 0.5, 2)


s1 = "I like NLP"
s2 = "I like NLTK"
s3 = "apple is fruit"

v1 = avg_word2vec(s1)
v2 = avg_word2vec(s2)
v3 = avg_word2vec(s3)

print("Word vectors:")
for word, vector in embeddings.items():
    print(word, "->", vector)

print("\nSentence vectors (average of the words):")
print(s1, "->", v1)
print(s2, "->", v2)
print(s3, "->", v3)

print("\nCloser sentences have a smaller distance:")
print("I like NLP vs I like NLTK ->", distance(v1, v2))
print("I like NLP vs apple is fruit ->", distance(v1, v3))
