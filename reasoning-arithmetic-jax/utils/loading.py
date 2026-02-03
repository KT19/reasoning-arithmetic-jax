from typing import Any

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
from transformers import logging as transformers_logging

# Suppress transformers warnings about weight initialization
transformers_logging.set_verbosity_error()


def _resize_llama_token_embeddings(
    params: Any,
    new_vocab_size: int,
    rng: jax.Array,
    initializer_range: float = 0.02,
) -> Any:
    # 1) input embeddings: (V, H)
    emb = params["model"]["embed_tokens"]["embedding"]
    old_vocab_size, hidden = emb.shape

    add = new_vocab_size - old_vocab_size
    # match dtype
    new_rows = initializer_range * jax.random.normal(rng, (add, hidden), dtype=emb.dtype)
    emb2 = jnp.concatenate([emb, new_rows], axis=0)
    params = params.copy()
    params["model"] = params["model"].copy()
    params["model"]["embed_tokens"] = params["model"]["embed_tokens"].copy()
    params["model"]["embed_tokens"]["embedding"] = emb2

    # 2) lm-head
    if "lm_head" in params and "kernel" in params["lm_head"]:
        k = params["lm_head"]["kernel"]
        h2, v2 = k.shape
        if h2 != hidden or v2 != old_vocab_size:
            return params

        new_cols = initializer_range * jax.random.normal(jax.random.split(rng, 2)[1], (hidden, add), dtype=k.dtype)
        k2 = jnp.concatenate([k, new_cols], axis=1)
        params["lm_head"] = params["lm_head"].copy()
        params["lm_head"]["kernel"] = k2

    return params


def load_model_and_tokenizer_with_vocab_change(model_id: str, seed: int = 0) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    # Add pad
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"

    model = FlaxAutoModelForCausalLM.from_pretrained(model_id, from_pt=True, dtype=jnp.bfloat16)

    # Manually resize params
    new_vocab = len(tokenizer)
    old_vocab = model.config.vocab_size
    if new_vocab != old_vocab:
        rng = jax.random.PRNGKey(seed)
        params = _resize_llama_token_embeddings(
            model.params, new_vocab, rng, initializer_range=model.config.initializer_range
        )
        model.params = params
        model.config.vocab_size = new_vocab

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_params(ckpt_path: str, params: Any) -> Any:
    with open(ckpt_path + "/flax_model.msgpack", "rb") as f:
        msgpack_data = f.read()

    updated_params = from_bytes(params, msgpack_data)
    return updated_params
