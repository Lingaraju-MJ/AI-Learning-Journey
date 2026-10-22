import math

# last week one mix had to pick. here I run two mixes at once.
# head 1 looks for dog. head 2 looks for saw.
# then I stick the two mixed lists together.

words = ["it", "saw", "dog"]

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


def mix(query, key, value, word):
    raw = []
    for other in words:
        raw.append(dot(query[word], key[other]))
    weights = softmax(raw)
    mixed = [0.0, 0.0]
    for i, other in enumerate(words):
        mixed[0] = mixed[0] + weights[i] * value[other][0]
        mixed[1] = mixed[1] + weights[i] * value[other][1]
    return raw, weights, mixed


def show_head(name, raw, weights, mixed):
    print("\n" + name)
    print("scores:")
    for i, other in enumerate(words):
        print(" ", other, "->", round(raw[i], 2))
    print("attention:")
    for i, other in enumerate(words):
        print(" ", other, "->", round(weights[i], 2))
    print("mixed ->", [round(n, 2) for n in mixed])


print("Sentence:", " ".join(words))
print("I only print it. two heads, two different looks.")

raw1, w1, mix1 = mix(head1_query, head1_key, head1_value, "it")
raw2, w2, mix2 = mix(head2_query, head2_key, head2_value, "it")

show_head("head 1 (looking for dog)", raw1, w1, mix1)
show_head("head 2 (looking for saw)", raw2, w2, mix2)

joined = mix1 + mix2
print("\nstuck together ->", [round(n, 2) for n in joined])
print("four numbers now. one head could not hold both looks.")
