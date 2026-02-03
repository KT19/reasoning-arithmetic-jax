from dataclasses import dataclass


@dataclass(frozen=True)
class SFTTrainConfig:
    model_id: str
    max_len: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    steps: int
    log_every: int
    max_new_tokens: int
    seed: int
    save_dir: str
