import math
import random

random.seed(1)

# short on purpose. "nlp" and "nltk" both sit next to "like".
# "apple" and "mango" both sit next to "eat".
texts = [
    "like nlp",
    "like nltk",
    "eat apple",
    "eat mango",
]

sentences = [text.split() for text in texts]
window = 1

words = []
for sentence in sentences:
    for word in sentence:
        if word not in words:
            words.append(word)

pairs = []
for sentence in sentences:
    for i, word in enumerate(sentence):
        start = max(0, i - window)
        end = min(len(sentence), i + window + 1)
        for nearby in sentence[start:i] + sentence[i + 1:end]:
            pairs.append((word, nearby))

# hidden layer is just a vector per word. picking the word picks that vector.
# output side is another vector per word, used to score the guess.
size = 3
hidden = {}
output = {}
for word in words:
    hidden[word] = [random.uniform(-0.5, 0.5) for _ in range(size)]
    output[word] = [random.uniform(-0.5, 0.5) for _ in range(size)]

before = {}
for word in words:
    before[word] = hidden[word][:]


def dot(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + x * y
    return total


def softmax(values):
    # take off the biggest number so exp() stays small
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


def scores(word):
    h = hidden[word]
    raw = []
    for other in words:
        raw.append(dot(output[other], h))
    return softmax(raw)


def best_word(probs):
    best_i = 0
    for i in range(len(probs)):
        if probs[i] > probs[best_i]:
            best_i = i
    return words[best_i]


def distance(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + (x - y) ** 2
    return round(total ** 0.5, 2)


def rounded(vector):
    return [round(n, 2) for n in vector]


def train_one(word, nearby, rate):
    h = hidden[word][:]
    probs = scores(word)

    # how far off each guess is. the word we wanted should have been 1.
    grad_h = [0.0] * size
    for i, other in enumerate(words):
        if other == nearby:
            error = probs[i] - 1
        else:
            error = probs[i]

        old = output[other]
        updated = []
        for j in range(size):
            updated.append(old[j] - rate * error * h[j])
            grad_h[j] = grad_h[j] + error * old[j]
        output[other] = updated

    moved = []
    for j in range(size):
        moved.append(h[j] - rate * grad_h[j])
    hidden[word] = moved
    return probs


print("Sentences:")
for text in texts:
    print("-", text)

print("\nPairs (word -> word I am trying to guess):")
for word, nearby in pairs:
    print(word, "->", nearby)

# one step, before the loop, so I can see a miss
word, nearby = "nlp", "like"
probs = scores(word)
guess = best_word(probs)

print("\nOne step, before training. Word in:", word)
print("Hidden layer (this is the embedding for nlp):")
print(rounded(hidden[word]))
print("Chance of each next word:")
for i, other in enumerate(words):
    print(" ", other, "->", round(probs[i], 2))
print("Guess:", guess)
print("Wanted:", nearby)
if guess == nearby:
    print("Hit it on the first try.")
else:
    print("Wrong, so I nudge the nlp vector a little.")

train_one(word, nearby, 0.4)
print("nlp vector after that one nudge:")
print(rounded(hidden[word]))

# now the rest of the pairs, a few passes
rate = 0.4
for _ in range(30):
    for pair_word, pair_nearby in pairs:
        train_one(pair_word, pair_nearby, rate)

print("\nSame step after training:")
probs = scores("nlp")
print("Guess for nlp:", best_word(probs), "(wanted like)")
probs = scores("apple")
print("Guess for apple:", best_word(probs), "(wanted eat)")

print("\nDistance before training:")
print("nlp vs nltk  ->", distance(before["nlp"], before["nltk"]))
print("apple vs mango ->", distance(before["apple"], before["mango"]))
print("nlp vs apple  ->", distance(before["nlp"], before["apple"]))

print("\nDistance after training:")
print("nlp vs nltk  ->", distance(hidden["nlp"], hidden["nltk"]))
print("apple vs mango ->", distance(hidden["apple"], hidden["mango"]))
print("nlp vs apple  ->", distance(hidden["nlp"], hidden["apple"]))
