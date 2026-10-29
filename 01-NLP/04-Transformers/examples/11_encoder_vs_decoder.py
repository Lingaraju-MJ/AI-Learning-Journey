# last week's block was the encoder. it already has the whole sentence.
# every word can look at every word.
#
# the paper also has a decoder. it writes the next word.
# a decoder word cannot look at words I have not written yet.

encoder = ["it", "saw", "dog"]
decoder_so_far = ["it", "saw"]

print("encoder (already read):", " ".join(encoder))
print("every word can look at every word.")

print("\ndecoder so far:", " ".join(decoder_so_far))
print("I have not written dog yet.")

print("\nwho can look at who:")
print(" encoder it  ->", ", ".join(encoder))
print(" encoder saw ->", ", ".join(encoder))
print(" encoder dog ->", ", ".join(encoder))
print(" decoder it  -> it")
print(" decoder saw -> it, saw")
print(" decoder dog -> not written yet")

print("\nencoder uses the mix from last week.")
print("decoder uses the mask from 09, then looks at the encoder too.")
