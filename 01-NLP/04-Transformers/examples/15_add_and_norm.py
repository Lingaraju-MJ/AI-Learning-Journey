import math

# add the mix back, then pull the numbers back.

word = [1.0, 0.0]
mixed = [0.25, 0.67]

added = []
for i in range(len(word)):
    added.append(word[i] + mixed[i])

total = 0
for n in added:
    total = total + n
avg = total / len(added)

shifted = []
for n in added:
    shifted.append(n - avg)

sq = 0
for n in shifted:
    sq = sq + n * n
spread = math.sqrt(sq / len(added))

normed = []
for n in shifted:
    normed.append(round(n / spread, 2))

print("word:", word)
print("mixed:", mixed)
print("after add:", added)
print("after add and norm:", normed)
