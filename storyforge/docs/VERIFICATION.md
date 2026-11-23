# Verification performed

- Seven pytest tests passed: future-token masking, cached multi-token logits,
  gradients and tied embeddings, parameter reduction, generation across context-window
  rollover, data deduplication/split separation, checkpoint/tokenizer round-trip and reused-tokenizer checksum preservation.
- Python compilation passed for all command-line modules.
- FastAPI test client: startup, health, generation and invalid-temperature rejection passed.
- A 100-step synthetic CPU run completed; its saved best checkpoint reloaded and generated.
- Resume from synthetic last.pt completed step 101 successfully.
- TinyStories streaming was exercised against the real dataset. Parquet background
  pre-buffering is disabled so early-stopped streams shut down cleanly. A fresh 30-story /
  10-story preparation completed with exit status zero after this fix.
- The bundled full-width model has 1,433,280 parameters for a 512-token vocabulary.
  The corresponding learned-position version has 1,482,432.

Tests establish the checked behaviours, not freedom from all possible bugs. CUDA/MPS
execution was not tested in this environment. CPU measurements and dependency versions
are included so the local validation scope is clear. Generation quality needs further
training and evaluation; do not present synthetic-fixture loss as real-data performance.

- The real TinyStories training run completed all 2,000 steps. Check MODEL_CARD.md and the run logs for measurements.
