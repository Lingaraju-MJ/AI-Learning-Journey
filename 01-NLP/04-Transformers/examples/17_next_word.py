import math

# decoder just finished "it saw".
# the last vector has to become a word.

out = [1.05, 1.82]

vocab = ["it", "saw", "dog"]
table = [
    [1.0, 0.0],
    [0.5, 0.0],
    [0.0, 1.0],
]

scores = []
for row in table:
    total = 0
    for i in range(len(out)):
        total = total + out[i] * row[i]
    scores.append(total)

biggest = max(scores)
exps = []
for s in scores:
    exps.append(math.exp(s - biggest))

total = 0
for e in exps:
    total = total + e

probs = []
for e in exps:
    probs.append(e / total)

print("decoder so far: it saw")
print("last vector:", out)

print("\nscores:")
for i in range(len(vocab)):
    print(vocab[i], round(scores[i], 2))

print("\nprobs:")
for i in range(len(vocab)):
    print(vocab[i], round(probs[i], 2))

guess = vocab[0]
best = probs[0]
for i in range(len(vocab)):
    if probs[i] > best:
        best = probs[i]
        guess = vocab[i]

print("\nnext word:", guess)
