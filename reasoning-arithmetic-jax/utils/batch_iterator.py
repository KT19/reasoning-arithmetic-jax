from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rewards.reward_func import correctness_reward_func, format_reward_func
from train.grpo_train_config import GRPOTrainConfig
from train.sft_train_config import SFTTrainConfig
from utils.sampling import sample_K_completion
from utils.synthetic_math import build_arithmetic_step_data, encode_synthetic_data


def batch_iterator_synthetic_math_sft(
    tokenizer: Any,
    cfg: SFTTrainConfig,
    pad_id: int,
    seed: int,
) -> Iterator[dict[str, jax.Array]]:
    rng = np.random.default_rng(seed)

    while True:
        batch_ids: list[np.ndarray] = []
        batch_lm: list[np.ndarray] = []

        for _ in range(cfg.batch_size):
            # manual data
            example = build_arithmetic_step_data(rng=rng)
            question = example["question"]
            reasoning = example["reasoning"]
            ans = example["answer"] + tokenizer.eos_token

            ids, lm = encode_synthetic_data(
                tokenizer=tokenizer,
                prompt=question,
                reasoning=reasoning,
                answer=ans,
                max_len=cfg.max_len,
                pad_id=pad_id,
            )
            batch_ids.append(ids)
            batch_lm.append(lm)

        input_ids = jnp.asarray(np.stack(batch_ids, axis=0), dtype=jnp.int32)
        loss_mask = jnp.asarray(np.stack(batch_lm, axis=0), dtype=jnp.int32)

        yield {"input_ids": input_ids, "loss_mask": loss_mask}


def batch_iterator_synth_arith_rl(
    seed: int,
    model: Any,
    tokenizer: Any,
    params: Any,
    pad_id: int,
    cfg: GRPOTrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      ids_np: (N,T)
      comp_np: (N,T)
      r_np: (N,)
    """
    rng = np.random.default_rng(seed)

    B = int(cfg.prompts_per_step)
    K = int(cfg.group_size)
    N = B * K

    max_len = int(cfg.max_len)

    prompts: list[str] = []
    gts: list[str] = []
    prompts_ids: list[list[int]] = []

    for _ in range(B):
        example = build_arithmetic_step_data(rng=rng)
        # extract only question and answer
        prompt = example["question"]
        ans = example["gt"]

        gt = str(ans)
        prompts.append(prompt)
        gts.append(gt)
        prompts_ids.append(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    ids_np = np.zeros((N, max_len), dtype=np.int32)
    comp_np = np.zeros((N, max_len), dtype=np.float32)
    r_np = np.zeros((N,), dtype=np.float32)

    key = jax.random.PRNGKey(seed)

    for b in range(B):
        start_idx = b * K
        gt = gts[b]
        k_prompts = jnp.asarray([prompts_ids[b] for _ in range(K)])

        key, subkey = jax.random.split(key)
        prompt_len = len(prompts_ids[b])
        full_ids, comp_mask, _ = sample_K_completion(
            key=subkey,
            model=model,
            tokenizer=tokenizer,
            params=params,
            prompt_ids=k_prompts,
            pad_id=pad_id,
            max_len=cfg.max_len,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        ids_np[start_idx : start_idx + K] = np.array(full_ids)
        comp_np[start_idx : start_idx + K] = np.array(comp_mask)

        for k_idx in range(K):
            f_id = full_ids[k_idx]
            completion = tokenizer.decode(f_id[prompt_len:].tolist(), skip_special_tokens=True)
            answer_reward = correctness_reward_func(completion=completion, answer=gt)
            format_reward = format_reward_func(completion=completion)

            r_np[start_idx + k_idx] = answer_reward + format_reward

    return ids_np, comp_np, r_np
