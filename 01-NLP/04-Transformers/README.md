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

## Scaled scores

The last file used the raw dot. A longer vector makes those dots big, and softmax then gives almost everything to one word.

I divide the score by the square root of the vector length. Here the length is 2, so the divisor is about 1.41. `dog` goes from 2.0 to 1.41. Attention on `dog` goes from 0.79 to 0.67. `it` and `saw` each go from 0.11 to 0.16.

To see it more clearly I also tried a fake score of 8 with length 16. Raw softmax is about `[0.0, 0.0, 1.0]`. After dividing by 4 it is `[0.11, 0.11, 0.79]`. The other words are still in the mix.

Example: `examples/03_scaled_attention.py`

## What I understood

Attention is a weighted mix. The word does not get replaced by one other word. It keeps a bit of everyone, more from the word with the higher score.

Query, key, and value are not a second embedding I invent. They are the same word vector, passed through three small weight lists.

Dividing by the square root keeps one large dot from taking the whole softmax.

The later files keep that same idea and add the rest of one small block: add the mix back, mark the word's place, run a tiny map, then look a second time.

## Add the mix back

The mixed vector for `it` is `[0.25, 0.67]`. Adding it onto `[1.0, 0.0]` gives `[1.25, 0.67]`. The word is still itself, and it also carries some of `dog`.

Example: `examples/04_add_back.py`

## Position

`it` at the start and `it` at the end were the same vector. I add a small list for the place. Start stays `[1.0, 0.0]`. The end becomes `[1.2, 0.4]`.

Example: `examples/05_position.py`

## A small map after the mix

The added-back vector goes through another tiny weight list. `[1.25, 0.67]` becomes `[1.25, 1.34]`. A negative number is wiped to 0. Attention did not do that.

Example: `examples/06_feed_forward.py`

## A second look

I add the mix back, then run the same look again. `it` goes `[1.0, 0.0]`, then `[1.25, 0.67]`, then `[2.08, 1.62]`. One mix was not the end.

Example: `examples/07_second_pass.py`

## Next

These are still one sentence and made-up weights. Next I want two mixes side by side, so the word can look for two different things at once.
