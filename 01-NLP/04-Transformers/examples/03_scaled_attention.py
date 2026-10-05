import math

# same sentence and weights as 02_qkv_from_word.py.
# last time I used the raw dot as the score.
# this time I divide by the square root of the vector length first.

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


query = {}
key = {}
value = {}
for word in words:
    query[word] = project(word_vec[word], wq)
    key[word] = project(word_vec[word], wk)
    value[word] = project(word_vec[word], wv)

length = len(word_vec["it"])
scale = math.sqrt(length)

print("Sentence:", " ".join(words))
print("Vector length:", length, "  sqrt:", round(scale, 2))

raw = []
for other in words:
    raw.append(dot(query["it"], key[other]))

scaled = []
for score in raw:
    scaled.append(score / scale)

raw_w = softmax(raw)
scaled_w = softmax(scaled)

print("\nScores for it, raw then divided:")
for i, other in enumerate(words):
    print(" ", other, "->", round(raw[i], 2), "   scaled ->", round(scaled[i], 2))

print("\nAttention for it:")
print("raw    ->", rounded(raw_w))
print("scaled ->", rounded(scaled_w))
print("dog is still the biggest. the others get a bit more.")

# a longer vector makes bigger dots. this is just to see that.
big = [0.0, 0.0, 8.0]
big_scale = math.sqrt(16)
big_scaled = []
for score in big:
    big_scaled.append(score / big_scale)

print("\nSame idea with a big score, vector length 16:")
print("raw scores    ->", big)
print("raw attention ->", rounded(softmax(big)))
print("scaled scores ->", rounded(big_scaled))
print("scaled attn   ->", rounded(softmax(big_scaled)))
