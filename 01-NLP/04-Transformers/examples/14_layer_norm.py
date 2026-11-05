import math

# added back numbers can get big. layer norm pulls them back.

vec = [1.25, 0.67]

total = 0
for n in vec:
    total = total + n
avg = total / len(vec)

shifted = []
for n in vec:
    shifted.append(n - avg)

sq = 0
for n in shifted:
    sq = sq + n * n
spread = math.sqrt(sq / len(vec))

normed = []
for n in shifted:
    normed.append(round(n / spread, 2))

print("added back:", vec)
print("mean:", round(avg, 2))
print("spread:", round(spread, 2))
print("after layer norm:", normed)

# same thing on the bigger vector from the second look
vec2 = [2.08, 1.62]
total = 0
for n in vec2:
    total = total + n
avg = total / len(vec2)

shifted = []
for n in vec2:
    shifted.append(n - avg)

sq = 0
for n in shifted:
    sq = sq + n * n
spread = math.sqrt(sq / len(vec2))

normed = []
for n in shifted:
    normed.append(round(n / spread, 2))

print("\nsecond look:", vec2)
print("after layer norm:", normed)
