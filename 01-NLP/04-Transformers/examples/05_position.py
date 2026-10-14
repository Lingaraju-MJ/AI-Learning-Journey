# the word vector so far does not know where the word sits.
# "it" at the start and "it" at the end would be the same numbers.
# I add a small position list before attention.

words = ["it", "saw", "dog"]

word_vec = {
    "it": [1.0, 0.0],
    "saw": [0.5, 0.0],
    "dog": [0.0, 1.0],
}

# one list per spot in the sentence. I made these up.
position = [
    [0.0, 0.0],
    [0.1, 0.2],
    [0.2, 0.4],
]


def add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(round(x + y, 2))
    return out


print("Sentence:", " ".join(words))
print("\nWord, then the same word plus its place:")
for i, word in enumerate(words):
    print(" ", word, word_vec[word], "->", add(word_vec[word], position[i]))

print("\nSame word, two places:")
print(" it at the start ->", add(word_vec["it"], position[0]))
print(" it at the end   ->", add(word_vec["it"], position[2]))
print("the word did not change. the place did.")
