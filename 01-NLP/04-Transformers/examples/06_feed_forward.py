# after attention adds the mix back, the block still runs the vector
# through a small map of its own. nothing here is trained.

# this is the "added back" vector for it from 04_add_back.py
vec = [1.25, 0.67]


def step(vec, weights):
    out = []
    for w in weights:
        total = 0
        for x, y in zip(vec, w):
            total = total + x * y
        if total < 0:
            total = 0
        out.append(total)
    return out


# two numbers in, two numbers out. the second weight doubles the dog part.
weights = [
    [1.0, 0.0],
    [0.0, 2.0],
]

out = step(vec, weights)

print("in  ->", vec)
print("out ->", [round(n, 2) for n in out])

# a negative is wiped. attention never did that.
print("\nnegative in  ->", [1.0, -0.4])
print("negative out ->", [round(n, 2) for n in step([1.0, -0.4], weights)])
print("the minus became 0. the rest stayed.")
