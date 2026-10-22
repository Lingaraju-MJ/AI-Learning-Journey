import math

# one small block for it, in order:
# two heads, stick them together, shrink back to 2 numbers,
# add onto the original word, then the tiny map from last week.

words = ["it", "saw", "dog"]
word_vec = {
    "it": [1.0, 0.0],
    "saw": [0.5, 0.0],
    "dog": [0.0, 1.0],
}

head1_query = {
    "it": [1.0, 0.0],
    "saw": [0.0, 1.0],
    "dog": [0.2, 0.0],
}
head1_key = {
    "it": [0.2, 0.0],
    "saw": [0.0, 1.0],
    "dog": [1.0, 0.1],
}
head1_value = {
    "it": [0.1, 0.1],
    "saw": [0.0, 0.8],
    "dog": [0.9, 0.2],
}

head2_query = {
    "it": [0.0, 1.0],
    "saw": [1.0, 0.0],
    "dog": [0.0, 0.2],
}
head2_key = {
    "it": [0.0, 0.2],
    "saw": [0.0, 1.0],
    "dog": [1.0, 0.0],
}
head2_value = {
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


def mix(query, key, value):
    raw = []
    for other in words:
        raw.append(dot(query["it"], key[other]))
    weights = softmax(raw)
    mixed = [0.0, 0.0]
    for i, other in enumerate(words):
        mixed[0] = mixed[0] + weights[i] * value[other][0]
        mixed[1] = mixed[1] + weights[i] * value[other][1]
    return weights, mixed


def shrink(joined):
    # four numbers back to two, so I can add them onto it
    return [
        joined[0] + joined[2],
        joined[1] + joined[3],
    ]


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


print("Sentence:", " ".join(words))
print("start it ->", word_vec["it"])

w1, mix1 = mix(head1_query, head1_key, head1_value)
w2, mix2 = mix(head2_query, head2_key, head2_value)

print("\nhead 1 attention (wants dog):")
for i, other in enumerate(words):
    print(" ", other, "->", round(w1[i], 2))
print("mixed ->", [round(n, 2) for n in mix1])

print("\nhead 2 attention (wants saw):")
for i, other in enumerate(words):
    print(" ", other, "->", round(w2[i], 2))
print("mixed ->", [round(n, 2) for n in mix2])

joined = mix1 + mix2
back = shrink(joined)
kept = [word_vec["it"][0] + back[0], word_vec["it"][1] + back[1]]
out = feed_forward(kept)

print("\nstuck together ->", [round(n, 2) for n in joined])
print("shrunk to 2    ->", [round(n, 2) for n in back])
print("added back     ->", [round(n, 2) for n in kept])
print("after the map  ->", [round(n, 2) for n in out])
print("that is one block. next word would do the same walk.")
