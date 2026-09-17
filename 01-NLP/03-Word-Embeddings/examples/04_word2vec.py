import random

random.seed(1)

# two groups of sentences on purpose: NLP words sit together, fruit words sit together
texts = [
    "I like NLP",
    "I like NLTK",
    "NLP uses Python",
    "NLTK uses Python",
    "apple is fruit",
    "mango is fruit",
    "eat apple",
    "eat mango",
]

sentences = [text.lower().split() for text in texts]
window = 2

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

# each word starts as 3 random numbers
embeddings = {}
for word in words:
    embeddings[word] = [random.random() for _ in range(3)]


def distance(a, b):
    total = 0
    for x, y in zip(a, b):
        total = total + (x - y) ** 2
    return round(total ** 0.5, 2)


def rounded(vector):
    return [round(n, 2) for n in vector]


print("Sentences:")
for text in texts:
    print("-", text)

print("\nSkip-gram is the Word2Vec mode I am using.")
print("A few pairs (word -> nearby word):")
for word, nearby in pairs[:8]:
    print(word, "->", nearby)

print("\nDistance before training:")
print("nlp vs nltk  ->", distance(embeddings["nlp"], embeddings["nltk"]))
print("apple vs mango ->", distance(embeddings["apple"], embeddings["mango"]))
print("nlp vs apple  ->", distance(embeddings["nlp"], embeddings["apple"]))

# for each pair, move the word a little toward the nearby word
rate = 0.02
for _ in range(25):
    for word, nearby in pairs:
        a = embeddings[word]
        b = embeddings[nearby]
        moved = []
        for x, y in zip(a, b):
            moved.append(x + rate * (y - x))
        embeddings[word] = moved

print("\nA few vectors after training:")
for word in ["nlp", "nltk", "apple", "mango"]:
    print(word, "->", rounded(embeddings[word]))

print("\nDistance after training:")
print("nlp vs nltk  ->", distance(embeddings["nlp"], embeddings["nltk"]))
print("apple vs mango ->", distance(embeddings["apple"], embeddings["mango"]))
print("nlp vs apple  ->", distance(embeddings["nlp"], embeddings["apple"]))
