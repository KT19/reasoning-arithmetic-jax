# GRPO implementation with frontend

This repository implements (Dr.) GRPO in JAX.
For efficient training, smolLM is used for the base model, and synthetic arithmetic math is performed.
Additionally, for faster training, distributed training (Data Parallel) is used.
