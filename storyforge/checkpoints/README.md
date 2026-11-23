# Checkpoint files
Training creates `tiny/best.pt` (portable model, tokenizer, configuration and metadata)
and `tiny/last.pt` (also optimizer and RNG state, for resuming).
A `.pt` file is real serialized weights, not Python model source.
If `demo.pt` is included, it is a short synthetic-data integration checkpoint only.
It is NOT a trained TinyStories story generator. Consult MODEL_CARD.md for actual status.
Only load checkpoints from trusted sources. This code uses torch.load(weights_only=True).
After your full run, publish best.pt with the matching metrics.csv and run.json.
Use a GitHub Release or Hugging Face for larger training checkpoints; the best checkpoint
for this tiny configuration is approximately 6 MB and can also be explicitly added to git.
