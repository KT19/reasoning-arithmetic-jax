# Reasoning model implemented in JAX
This is an implementation of a reasoning model in JAX.

For demonstration purposes, a synthetic mathematical task is used.

The publicly-available pretrained model, smolLM2, is used as the base model.

Through SFT and (Dr.) GRPO, the model supports arithmetic calculation.

# Environment
`uv` package manager is used.

**install**
```bash
uv sync
```

# SFT
Before (Dr.) GRPO, we need to tell the model the basic ability to solve arithmetic.

SFT is performed using Data Parallel.

**training**
```bash
uv run train_sft.py
```

# GRPO
Once SFT is completed, the model is further trained to boost performance.

**training**
```bash
uv run train_grpo.py
```

# Inference
CLI-based inference is supported.

```bash
uv run inference.py
```

# Reference
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)