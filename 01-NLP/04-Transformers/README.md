# Transformers

Word2Vec gives each word one vector and leaves it there. A transformer lets each word look at the other words in the sentence and mix a bit of them in.

This folder is still one sentence and made-up weights. No training.

Last week I ran two mixes, hid a word with a mask, then walked one encoder-style block. This week I split encoder and decoder, then let a decoder word look at the encoder.

I am reading The Illustrated Transformer while I do this: https://jalammar.github.io/illustrated-transformer/

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

## Two heads

One mix has to pick. Two mixes can look for two different things at the same time.

Head 1 is the old `it` → `dog` look. Head 2 is written so `it` looks at `saw` instead. I print both, then stick the two mixed lists into one longer list.

That longer list is why people say multi-head. Nothing is trained. I just typed a second query.

Example: `examples/08_two_heads.py`

## Mask

Sometimes a word should not count. Padding is like that. A later word is like that if I am guessing the next word.

I do not delete `dog`. I give it a score of `-100` so softmax almost skips it. Compare that with the same numbers and no mask. `dog` is still in the sentence. It just gets almost no weight.

Example: `examples/09_mask.py`

## One small block

The pieces from last week plus the two heads, in one walk for `it`:

1. two heads
2. stick the mixes together
3. shrink back to two numbers so I can add them onto `it`
4. the tiny map that wipes a negative

That is one encoder-style step. The next word would do the same walk. This week I treat that file as the encoder.

Example: `examples/10_one_block.py`

## Encoder vs decoder

The encoder already has the whole sentence. Every word can look at every word. That is last week's block.

The decoder writes the next word. I have only written `it saw`. `dog` is not there yet, so a decoder word cannot look at it.

- encoder `it`, `saw`, `dog` each look at `it, saw, dog`
- decoder `it` looks at `it`
- decoder `saw` looks at `it, saw`
- decoder `dog` is not written yet

The mask from `09_mask.py` is what the decoder uses on its own side.

Example: `examples/11_encoder_vs_decoder.py`

## Cross-attention

Self-attention keeps query, key, and value in the same sentence. Cross-attention does not.

The decoder word `it` is looking for something. The encoder words are what it can look at. Query comes from the decoder. Key and value come from the encoder.

Same numbers as file 01, so the mix still leans toward `dog`: scores `0.2`, `0.0`, `1.0`, attention `0.25`, `0.20`, `0.55`, mixed `[0.52, 0.3]`. The only change is which side each list comes from.

Example: `examples/12_cross_attention.py`

## One decoder block

Last week `10_one_block.py` was the encoder walk for `it`. Here I walk decoder `saw`.

I already wrote `it`. Now I am on `saw`.

1. decoder look: `saw` can see `it` and itself, not `dog`. Attention is `0.27` and `0.73`. Mixed `[0.03, 0.61]`.
2. encoder look: same cross-attention as the last file. `dog` gets `0.55`. Mixed `[0.52, 0.3]`.
3. add both mixes onto `saw` `[0.5, 0.0]` -> `[1.05, 0.91]`
4. the tiny map -> `[1.05, 1.82]`

That is one decoder-style step.

Example: `examples/13_decoder_block.py`

## Next

These are still one sentence and made-up weights. Next I can watch Krish's architecture video and skim the paper. Layer norm is still missing. After that, a tokenizer or a tiny real model.
