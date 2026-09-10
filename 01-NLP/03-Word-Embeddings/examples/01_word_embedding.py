# each word is just a short list of numbers
# I made these up so I can see the idea. they are not trained.

embeddings = {
    "nlp": [0.9, 0.2, 0.1],
    "nltk": [0.8, 0.3, 0.1],
    "python": [0.7, 0.4, 0.2],
    "apple": [0.1, 0.8, 0.7],
    "mango": [0.1, 0.9, 0.6],
}


def distance(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + (x - y) ** 2
    return round(total ** 0.5, 2)


print("Each word as numbers:")
for word, vector in embeddings.items():
    print(word, "->", vector)

print("\nCloser numbers mean the words are more similar:")
print("nlp vs nltk  ->", distance(embeddings["nlp"], embeddings["nltk"]))
print("nlp vs python ->", distance(embeddings["nlp"], embeddings["python"]))
print("nlp vs apple  ->", distance(embeddings["nlp"], embeddings["apple"]))
print("apple vs mango ->", distance(embeddings["apple"], embeddings["mango"]))
