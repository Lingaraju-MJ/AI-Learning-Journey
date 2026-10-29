import math

# last week query, key, and value all came from the same sentence.
# that is self-attention. the encoder does that.
#
# now the decoder word looks at the encoder.
# query comes from the word I am writing.
# key and value come from the encoder words.
# that is cross-attention.

encoder_words = ["it", "saw", "dog"]
decoder_word = "it"

# what the decoder word is looking for
query = {
    "it": [1.0, 0.0],
}

# what the encoder words offer
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


raw = []
for other in encoder_words:
    raw.append(dot(query[decoder_word], key[other]))
weights = softmax(raw)

mixed = [0.0, 0.0]
for i, other in enumerate(encoder_words):
    mixed[0] = mixed[0] + weights[i] * value[other][0]
    mixed[1] = mixed[1] + weights[i] * value[other][1]

print("encoder:", " ".join(encoder_words))
print("decoder word so far:", decoder_word)
print("query from decoder. key and value from encoder.")

print("\nscores:")
for i, other in enumerate(encoder_words):
    print(" ", other, "->", round(raw[i], 2))

print("attention:")
for i, other in enumerate(encoder_words):
    print(" ", other, "->", round(weights[i], 2))

print("mixed ->", [round(n, 2) for n in mixed])
print("same mix as file 01, but the look goes across the two sides.")
