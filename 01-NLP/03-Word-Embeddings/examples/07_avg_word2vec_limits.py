# same average as 05. this file is where that average goes wrong.

embeddings = {
    "dog": [0.9, 0.1, 0.1],
    "bites": [0.2, 0.8, 0.2],
    "man": [0.1, 0.2, 0.9],
    "nlp": [0.9, 0.2, 0.1],
    "is": [0.3, 0.3, 0.3],
}


def avg_word2vec(text):
    words = text.lower().split()
    vecs = []
    for word in words:
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


s1 = "dog bites man"
s2 = "man bites dog"
v1 = avg_word2vec(s1)
v2 = avg_word2vec(s2)

print("Word order:")
print(s1, "->", v1)
print(s2, "->", v2)
print("distance ->", distance(v1, v2))
print("same numbers. the average does not care which word came first.")

nlp = avg_word2vec("nlp")
nlp_is = avg_word2vec("nlp is")
buried = avg_word2vec("is is is nlp")

print("\nA common word pulls as hard as the real one:")
print("nlp          ->", nlp)
print("nlp is       ->", nlp_is)
print("is is is nlp ->", buried)
print("nlp vs nlp is       ->", distance(nlp, nlp_is))
print("nlp vs is is is nlp ->", distance(nlp, buried))
print("one 'is' already moves it. three of them bury nlp.")
