# Transformers

Word2Vec gives each word one vector and leaves it there. A transformer lets each word look at the other words in the sentence and mix a bit of them in.

This file is only that mix. One sentence. No layers, no training.

## Attention

Sentence: `it saw dog`

Each word has three small lists:

- query: what it is looking for
- key: what it offers
- value: the numbers that get copied across

The score is just query dot key. Softmax turns those scores into weights that add up to 1. The new vector is those weights times the values.

I wrote the numbers so `it` is looking for something like `dog`.

`it` scores: `dog` 1.0, itself 0.2, `saw` 0.0. After softmax that is 0.55, 0.25, 0.20. The mixed vector is `[0.52, 0.3]`, pulled toward `dog`'s value `[0.9, 0.2]`.

`saw` mostly looks at itself. That is fine. The point was `it`.

Example: `examples/01_attention.py`

## Query, key, and value from the word vector

In the first file I typed those three lists myself. Here each one is the word vector dotted with a small weight list.

Same sentence: `it saw dog`

`it` is `[1.0, 0.0]`. Dotted with the two query weights it becomes `[0.0, 1.0]`. `dog`'s key comes out as `[0.0, 2.0]`. Those line up, so the score is `2.0`. After softmax, `it` puts `0.79` on `dog`. The mixed vector is `[0.16, 0.79]`, mostly `dog`.

The value weights just copy the word vector through. `it`'s value is still `[1.0, 0.0]`.

Example: `examples/02_qkv_from_word.py`

## What I understood

Attention is a weighted mix. The word does not get replaced by one other word. It keeps a bit of everyone, more from the word with the higher score.

Query, key, and value are not a second embedding I invent. They are the same word vector, passed through three small weight lists.

## Next

The scores here are used raw. Next I want to divide them by the square root of the vector length before the softmax. That is the "scaled" part of scaled dot-product attention.
