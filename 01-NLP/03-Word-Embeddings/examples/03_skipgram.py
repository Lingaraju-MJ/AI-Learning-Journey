text = "I am learning NLP with simple Python examples"
words = text.split()
window = 2

print("Original text:")
print(text)

print("\nWords:")
print(words)

print("\nWindow size:", window)
print("Skip-gram starts from one word and tries to guess the nearby words.")

print("\nSkip-gram pairs (word -> context word to predict):")
for i, word in enumerate(words):
    start = max(0, i - window)
    end = min(len(words), i + window + 1)
    context = words[start:i] + words[i + 1:end]
    for nearby in context:
        print(word, "->", nearby)
