import math

# one look is not the whole model.
# I add the mix back, then I let "it" look at the sentence again.

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


def one_pass(vectors):
    query = {}
    key = {}
    value = {}
    for word in words:
        query[word] = project(vectors[word], wq)
        key[word] = project(vectors[word], wk)
        value[word] = project(vectors[word], wv)

    scale = math.sqrt(2)
    nxt = {}
    for word in words:
        raw = []
        for other in words:
            raw.append(dot(query[word], key[other]) / scale)
        weights = softmax(raw)
        mixed = [0.0, 0.0]
        for i, other in enumerate(words):
            mixed[0] = mixed[0] + weights[i] * value[other][0]
            mixed[1] = mixed[1] + weights[i] * value[other][1]
        nxt[word] = add(vectors[word], mixed)
    return nxt


first = one_pass(word_vec)
second = one_pass(first)

print("Sentence:", " ".join(words))
print("it at the start ->", word_vec["it"])
print("after one look  ->", rounded(first["it"]))
print("after two looks ->", rounded(second["it"]))
print("the second look moves it again. one mix was not the end.")
