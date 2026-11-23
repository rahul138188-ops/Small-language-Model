# StoryForge model card

## Included real-data checkpoint

- File: `checkpoints/tiny/best.pt` (weights, configuration and tokenizer included).
- Initialization: random; all parameters trained in this session.
- Parameters: 1,433,280 (1.433M), with tied input/output weights.
- Architecture: 3 decoder blocks, 3 heads, hidden width 192, GELU FFN width 768,
  LayerNorm, RoPE, dropout 0.1. Context limit: 256 tokens.
- Tokenizer: byte-level BPE, 512 entries, trained only on the training subset.
- Corpus: first 10,000 training and 1,000 validation examples streamed from
  `roneneldan/TinyStories`; exact duplicates/overlap filtered before tokenization.
- Retained tokens: 3,759,687 train, 337,282 validation.
- Training: 2,000 optimizer steps, batch 4, accumulation 1, seed 42,
  float32, AdamW, base LR 0.0003, cosine decay and 100-step warmup.
- Total training-token positions: 2,048,000, sampled in random windows. This does not
  imply every story or every unique token was seen.
- Hardware: CPU, two PyTorch threads; measured training time 363.8 seconds.
- Best measured validation cross-entropy: **2.6402**, at step 2000.
- Corresponding token perplexity: **14.02**.
- Validation estimate: fixed random windows, 20 batches of 4 sequences (20,480 target
  tokens), not an exhaustive evaluation of the full validation split.

## Scope and limitations

This is an **early, limited-budget checkpoint**, not a fully trained or instruction-tuned
chat model. Outputs can contain broken words, repetition, inconsistent grammar and weak
story structure. No baseline quality advantage is claimed. A longer run with more token
exposure is needed before making claims about useful story generation. Losses from other
tokenizers or differently sampled validation sets are not directly comparable.

The training budget is smaller than the reference repository's effective batch-64 run.
Do not describe the bundled result as a reproduction of its reported quality. Cached
and uncached decoding were checked for matching greedy tokens; speed measurements are
specific to this CPU, prompt length and output length.

`docs/samples.json` contains unedited seeded outputs. `metrics.csv`, `run.json`,
`data_metadata.json`, `tokenizer.json`, `docs/training_loss.png` and
`docs/benchmark.json` provide evidence for this run. `last.pt` includes optimizer/RNG
state for continuation. It is excluded from Git by default.

## Separate integration fixture

`checkpoints/demo.pt` is a 100-step synthetic-template checkpoint (1,404,672 parameters,
363-token vocabulary, 64-token context). It is included only to demonstrate the offline
pipeline. Its loss is not a TinyStories evaluation result. Use `tiny/best.pt` for the
actual TinyStories experiment.

## Provenance

New implementation written with AI assistance, inspired by turancannb02/minigpt.
See ATTRIBUTION.md for references. The MIT license covers the software; this project
does not change the underlying dataset's terms. No upstream trained weights were used.
