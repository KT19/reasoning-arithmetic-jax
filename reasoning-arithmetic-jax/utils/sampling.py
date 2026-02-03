from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from utils.functions import logits_from_ids


def greedy_generate(
    model: Any, tokenizer: Any, params: Any, prompt: str, pad_id: int, max_len: int, max_new_tokens: int
) -> str:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    ids: list[int] = list(prompt_ids)

    padded_input = jnp.full((1, max_len), pad_id, dtype=jnp.int32)
    for _ in range(max_new_tokens):
        if len(ids) >= max_len:
            window = ids[-max_len:]
        else:
            window = ids

        cur_len = len(window)
        padded_input = padded_input.at[0, :cur_len].set(jnp.asarray(window, dtype=jnp.int32))

        logits = logits_from_ids(model, params, padded_input, pad_id)[0, cur_len - 1]  # (B, V,)

        next_id = int(jnp.argmax(logits))

        if next_id == int(tokenizer.eos_token_id):
            break

        ids.append(next_id)

    text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)

    return text


def top_k_sample(rng: np.random.Generator, logits: np.ndarray, top_k: int, temperature: float) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / max(temperature, 1e-6)
    if top_k > 0 and top_k < logits.shape[-1]:
        idx = np.argpartition(logits, -top_k)[-top_k:]
        vals = logits[idx]
        vals = vals - np.max(vals)
        probs = np.exp(vals)
        probs = probs / np.sum(probs)

        return int(idx[rng.choice(len(idx), p=probs)])

    vals = logits - np.max(logits)
    probs = np.exp(vals)
    probs = probs / np.sum(probs)

    return int(rng.choice(logits.shape[-1], p=probs))


def sample_K_completion(
    key: jax.Array,
    model: Any,
    tokenizer: Any,
    params: Any,
    prompt_ids: jax.Array,
    pad_id: int,
    max_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
        full_ids: (K,T) padded to max_len
        comp_mask: (K,T) 1 on completion tokens
        prompt_len: int
    """
    K, prompt_len = prompt_ids.shape

    # init
    full_ids = jnp.full((K, max_len), pad_id, dtype=jnp.int32)
    full_ids = full_ids.at[:, :prompt_len].set(prompt_ids)

    finished = jnp.zeros((K,), dtype=jnp.bool_)

    cur_len = prompt_len
    for _ in range(max_new_tokens):
        if cur_len >= max_len:
            break

        # forward
        logits = logits_from_ids(model, params, full_ids, pad_id)[:, cur_len - 1]  # (B, V,)

        logits = logits / temperature

        top_vals, _ = jax.lax.top_k(logits, k=top_k)
        kth = top_vals[:, -1][:, None]
        logits = jnp.where(logits < kth, -1e8, logits)

        # sample
        key, subkey = jax.random.split(key)
        next_tokens = jax.random.categorical(subkey, logits, axis=-1)  # type: ignore

        # Update if not already finished
        next_tokens = jnp.where(finished, pad_id, next_tokens)
        full_ids = full_ids.at[:, cur_len].set(next_tokens)

        # Update
        finished = finished | (next_tokens == tokenizer.eos_token_id)
        cur_len += 1
        if jnp.all(finished):
            break

    # post-process
    indices = jnp.arange(max_len)[None, :]

    after_prompt = indices >= prompt_len
    is_eos = full_ids == tokenizer.eos_token_id
    eos_pos = jnp.where(is_eos.any(axis=1), jnp.argmax(is_eos, axis=1), max_len)
    before_eos = indices <= eos_pos[:, None]

    comp_mask = jnp.where(after_prompt & before_eos & (full_ids != tokenizer.pad_token_id), 1, 0)

    return full_ids, comp_mask, prompt_len  # type: ignore
