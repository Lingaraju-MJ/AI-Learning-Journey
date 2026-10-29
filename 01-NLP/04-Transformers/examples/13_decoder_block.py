import math

# one decoder block for saw, in order:
# 1. look at decoder words so far (it, saw). dog is not written yet.
# 2. look at the encoder words (cross-attention)
# 3. add both mixes back onto saw
# 4. the tiny map from last week

encoder_words = ["it", "saw", "dog"]
decoder_so_far = ["it", "saw"]
word = "saw"

word_vec = {
    "it": [1.0, 0.0],
    "saw": [0.5, 0.0],
    "dog": [0.0, 1.0],
}

self_query = {
    "it": [1.0, 0.0],
    "saw": [0.0, 1.0],
}
self_key = {
    "it": [0.2, 0.0],
    "saw": [0.0, 1.0],
    "dog": [1.0, 0.1],
}
self_value = {
    "it": [0.1, 0.1],
    "saw": [0.0, 0.8],
    "dog": [0.9, 0.2],
}

cross_query = {
    "saw": [1.0, 0.0],
}
cross_key = {
    "it": [0.2, 0.0],
    "saw": [0.0, 1.0],
    "dog": [1.0, 0.1],
}
cross_value = {
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


def mix(query, key, value, look_at):
    raw = []
    for other in look_at:
        raw.append(dot(query[word], key[other]))
    weights = softmax(raw)
    mixed = [0.0, 0.0]
    for i, other in enumerate(look_at):
        mixed[0] = mixed[0] + weights[i] * value[other][0]
        mixed[1] = mixed[1] + weights[i] * value[other][1]
    return weights, mixed


def add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x + y)
    return out


def feed_forward(vec):
    weights = [
        [1.0, 0.0],
        [0.0, 2.0],
    ]
    out = []
    for w in weights:
        total = 0
        for x, y in zip(vec, w):
            total = total + x * y
        if total < 0:
            total = 0
        out.append(total)
    return out


self_w, self_mix = mix(self_query, self_key, self_value, decoder_so_far)
cross_w, cross_mix = mix(cross_query, cross_key, cross_value, encoder_words)

kept = add(word_vec[word], self_mix)
kept = add(kept, cross_mix)
out = feed_forward(kept)

print("encoder:", " ".join(encoder_words))
print("decoder so far:", " ".join(decoder_so_far))
print("walking", word)
print("start saw ->", word_vec[word])

print("\n1. decoder look (cannot see dog yet):")
for i, other in enumerate(decoder_so_far):
    print(" ", other, "->", round(self_w[i], 2))
print("mixed ->", [round(n, 2) for n in self_mix])

print("\n2. encoder look (cross-attention):")
for i, other in enumerate(encoder_words):
    print(" ", other, "->", round(cross_w[i], 2))
print("mixed ->", [round(n, 2) for n in cross_mix])

print("\n3. added both back ->", [round(n, 2) for n in kept])
print("4. after the map  ->", [round(n, 2) for n in out])
print("that is one decoder block.")
