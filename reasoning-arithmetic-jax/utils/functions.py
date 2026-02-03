import os
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


def cast_tree(tree: Any, dtype: Any) -> Any:
    """
    cast to the given type
    """
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)


def count_nonpad_tokens(input_ids: jax.Array, pad_id: int) -> jax.Array:
    return (input_ids != pad_id).astype(jnp.int32).sum()


@partial(jax.jit, static_argnums=(0,))
def logits_from_ids(model: Any, params: Any, input_ids: jax.Array, pad_id: int) -> jax.Array:
    attn = (input_ids != pad_id).astype(jnp.int32)
    out = model(input_ids, attention_mask=attn, params=params)

    return out.logits


def gather_logp_next(logits: jax.Array, labels_next: jax.Array) -> jax.Array:
    """
    Args:
        logits: (B, T-1, V)
        labels_next: (B, T-1)

    Return:
        logp: (B, T-1)
    """
    logp = jax.nn.log_softmax(logits, axis=-1)  # (B, T-1, V)
    return jnp.take_along_axis(logp, labels_next[..., None], axis=-1)[..., 0]


def save_checkpoint(save_dir: str, model: Any, params: Any) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # model params
    params_bf16 = cast_tree(params, jnp.bfloat16)
    model.save_pretrained(save_dir, params=params_bf16)

    print(f"[Saved] {save_dir}")
