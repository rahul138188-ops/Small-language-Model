# Project origins
The project scope and tiny model dimensions were inspired by
https://github.com/turancannb02/minigpt (turancannb02).
This package contains a newly written implementation, not a renamed copy of that repository.
The README, training code, data pipeline, tests and model implementation were written for this package with AI assistance.
Architecture: standard decoder-only causal Transformer; RoPE positional rotations; tied token/output embeddings.
Dataset used by the main training command: https://huggingface.co/datasets/roneneldan/TinyStories
Dataset paper: https://arxiv.org/abs/2305.07759
RoPE reference: https://arxiv.org/abs/2104.09864
Retain this provenance when publishing; describe the experiments you personally ran.
The demo corpus is generated from simple templates and is not TinyStories.
No upstream checkpoints or reported training results are redistributed or claimed here.
