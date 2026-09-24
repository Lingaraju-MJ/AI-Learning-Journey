import math

# one sentence. I want "it" to look at "dog", not at "saw".
# numbers are made up. nothing is trained yet.

words = ["it", "saw", "dog"]

# query: what this word is looking for
# key: what this word offers
# value: the numbers that get mixed
query = {
    "it": [1.0, 0.0],
    "saw": [0.0, 1.0],
    "dog": [0.2, 0.0],
}
key = {
    "it": [0.2, 0.0],
    "saw": [0.0, 1.0],
    "dog": [1.0, 0.1],
}
value = {
    "it": [0.1, 0.1],
    "saw": [0.0, 0.8],
    "dog": [0.9, 0.2],
}


def dot(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + x * y
    return total


def softmax(values):
    biggest = max(values)
    exps = []
    for value in values:
        exps.append(math.exp(value - biggest))
    total = 0
    for value in exps:
        total = total + value
    probs = []
    for value in exps:
        probs.append(value / total)
    return probs


print("Sentence:", " ".join(words))

for word in words:
    raw = []
    for other in words:
        raw.append(dot(query[word], key[other]))
    weights = softmax(raw)

    print("\n" + word)
    print("scores (query dot key):")
    for i, other in enumerate(words):
        print(" ", other, "->", round(raw[i], 2))

    print("attention:")
    for i, other in enumerate(words):
        print(" ", other, "->", round(weights[i], 2))

    size = len(value[word])
    mixed = []
    for j in range(size):
        total = 0
        for i, other in enumerate(words):
            total = total + weights[i] * value[other][j]
        mixed.append(round(total, 2))
    print("mixed value ->", mixed)
