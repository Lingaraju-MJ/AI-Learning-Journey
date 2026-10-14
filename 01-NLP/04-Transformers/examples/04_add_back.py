import math

# same sentence as 03_scaled_attention.py.
# last time the mixed vector replaced the word.
# here I add it back onto the original word vector.

words = ["it", "saw", "dog"]

word_vec = {
    "it": [1.0, 0.0],
    "saw": [0.5, 0.0],
    "dog": [0.0, 1.0],
}

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


def add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x + y)
    return out


query = {}
key = {}
value = {}
for word in words:
    query[word] = project(word_vec[word], wq)
    key[word] = project(word_vec[word], wk)
    value[word] = project(word_vec[word], wv)

scale = math.sqrt(len(word_vec["it"]))

raw = []
for other in words:
    raw.append(dot(query["it"], key[other]) / scale)
weights = softmax(raw)

mixed = [0.0, 0.0]
for i, other in enumerate(words):
    mixed[0] = mixed[0] + weights[i] * value[other][0]
    mixed[1] = mixed[1] + weights[i] * value[other][1]

kept = add(word_vec["it"], mixed)

print("Sentence:", " ".join(words))
print("original it ->", word_vec["it"])
print("mixed       ->", rounded(mixed))
print("added back  ->", rounded(kept))
print("it is still itself, and it also carries some of dog.")
