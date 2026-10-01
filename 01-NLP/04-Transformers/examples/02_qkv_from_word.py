import math

# same sentence as 01_attention.py.
# last time I typed query, key, and value myself.
# this time each one is the word vector dotted with a small weight list.

words = ["it", "saw", "dog"]

word_vec = {
    "it": [1.0, 0.0],
    "saw": [0.5, 0.0],
    "dog": [0.0, 1.0],
}

# two weights each, because the word vector has two numbers.
# I picked these so "it" still lines up with "dog".
wq = [
    [0.0, 1.0],
    [1.0, 0.0],
]
wk = [
    [1.0, 0.0],
    [0.0, 2.0],
]
wv = [
    [1.0, 0.0],
    [0.0, 1.0],
]


def dot(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + x * y
    return total


def project(vec, weights):
    out = []
    for w in weights:
        out.append(dot(vec, w))
    return out


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


def rounded(vec):
    return [round(n, 2) for n in vec]


query = {}
key = {}
value = {}
for word in words:
    query[word] = project(word_vec[word], wq)
    key[word] = project(word_vec[word], wk)
    value[word] = project(word_vec[word], wv)

print("Sentence:", " ".join(words))

print("\nWord vectors:")
for word in words:
    print(" ", word, "->", word_vec[word])

print("\nOne step, for it. Query = word vector dot each query weight.")
print("word vector:", word_vec["it"])
print("dot", wq[0], "->", dot(word_vec["it"], wq[0]))
print("dot", wq[1], "->", dot(word_vec["it"], wq[1]))
print("query for it ->", rounded(query["it"]))
print("key for dog  ->", rounded(key["dog"]))
print("those two line up, so the score should be the big one.")

for word in words:
    raw = []
    for other in words:
        raw.append(dot(query[word], key[other]))
    weights = softmax(raw)

    print("\n" + word)
    print("query ->", rounded(query[word]))
    print("key   ->", rounded(key[word]))
    print("value ->", rounded(value[word]))
    print("scores (query dot key):")
    for i, other in enumerate(words):
        print(" ", other, "->", round(raw[i], 2))
    print("attention:")
    for i, other in enumerate(words):
        print(" ", other, "->", round(weights[i], 2))

    mixed = []
    size = len(value[word])
    for j in range(size):
        total = 0
        for i, other in enumerate(words):
            total = total + weights[i] * value[other][j]
        mixed.append(round(total, 2))
    print("mixed value ->", mixed)
