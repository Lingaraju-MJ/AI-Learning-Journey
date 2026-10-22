import math

# same numbers as 01_attention.py.
# this time dog is not allowed. I do not delete it.
# I give it a terrible score so softmax almost skips it.

words = ["it", "saw", "dog"]
blocked = "dog"

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


def attention(mask):
    raw = []
    for other in words:
        score = dot(query["it"], key[other])
        if mask and other == blocked:
            score = -100
        raw.append(score)
    weights = softmax(raw)
    mixed = [0.0, 0.0]
    for i, other in enumerate(words):
        mixed[0] = mixed[0] + weights[i] * value[other][0]
        mixed[1] = mixed[1] + weights[i] * value[other][1]
    return raw, weights, mixed


print("Sentence:", " ".join(words))
print("blocked word:", blocked)

open_raw, open_w, open_mix = attention(False)
hid_raw, hid_w, hid_mix = attention(True)

print("\nno mask (it can look at dog):")
for i, other in enumerate(words):
    print(" ", other, "-> score", round(open_raw[i], 2), "  attn", round(open_w[i], 2))
print("mixed ->", [round(n, 2) for n in open_mix])

print("\nwith mask (dog is blocked):")
for i, other in enumerate(words):
    print(" ", other, "-> score", round(hid_raw[i], 2), "  attn", round(hid_w[i], 2))
print("mixed ->", [round(n, 2) for n in hid_mix])
print("dog is still in the sentence. it just got almost no weight.")
