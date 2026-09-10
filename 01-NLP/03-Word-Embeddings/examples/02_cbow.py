text = "I am learning NLP with simple Python examples"
words = text.split()
window = 2

print("Original text:")
print(text)

print("\nWords:")
print(words)

print("\nWindow size:", window)
print("CBOW looks at nearby words and tries to guess the middle word.")

print("\nCBOW pairs (context -> word to predict):")
for i, word in enumerate(words):
    start = max(0, i - window)
    end = min(len(words), i + window + 1)
    context = words[start:i] + words[i + 1:end]
    print(context, "->", word)
