from dataclasses import dataclass


@dataclass(frozen=True)
class GRPOTrainConfig:
    model_id: str
    max_len: int
    lr: float
    weight_decay: float
    grad_clip: float
    steps: int
    log_every: int
    monitor_answer_step: int
    max_new_tokens: int
    seed: int
    save_dir: str
    # ppo
    ppo_clip_eps: float
    kl_beta: float
    # group
    prompts_per_step: int
    group_size: int
    # sampling
    temperature: float
    top_k: int
