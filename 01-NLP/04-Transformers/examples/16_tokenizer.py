# so far I typed the word vectors myself.
# a tokenizer first turns each word into a number.
# then I look that number up in a table.

text = "it saw dog"
words = text.split()

vocab = {}
ids = []
for word in words:
    if word not in vocab:
        vocab[word] = len(vocab)
    ids.append(vocab[word])

table = [
    [1.0, 0.0],
    [0.5, 0.0],
    [0.0, 1.0],
]

print("text:", text)
print("words:", words)
print("ids:", ids)

print("\nlookup:")
for i in range(len(ids)):
    print(words[i], ids[i], "->", table[ids[i]])
