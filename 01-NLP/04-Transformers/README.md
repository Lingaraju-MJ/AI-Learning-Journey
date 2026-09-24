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

## What I understood

Attention is a weighted mix. The word does not get replaced by one other word. It keeps a bit of everyone, more from the word with the higher score.

## Next

The query, key, and value here are handwritten. Next I want to see how those three lists are made from the word vector.
