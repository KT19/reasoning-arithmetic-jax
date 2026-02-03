from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from train.make_optimizer import make_optimizer
from train.sft_train_config import SFTTrainConfig
from utils.batch_iterator import batch_iterator_synthetic_math_sft
from utils.functions import cast_tree, save_checkpoint
from utils.loading import load_model_and_tokenizer_with_vocab_change
from utils.sampling import greedy_generate
from utils.sharding import create_dp_sharding
from utils.synthetic_math import build_arithmetic_step_data


def build_train_step(
    model: Any, optimizer: optax.GradientTransformation, pad_id: int, replicated: jax.NamedSharding
) -> Any:
    """
    Returns a jitted train step
    replicated: Sharding pattern
    """

    @jax.jit
    def train_step(params: Any, opt_state: Any, batch: dict[str, jax.Array]) -> tuple[Any, Any, jax.Array]:
        input_ids = batch["input_ids"]  # (B, T)
        loss_mask = batch["loss_mask"]  # (B, T)
        attn_mask = (input_ids != pad_id).astype(jnp.int32)

        def loss_fn(p: Any) -> jax.Array:
            outputs = model(input_ids, attention_mask=attn_mask, params=p)  # apply_fn
            logits = outputs.logits  # (B, T, V)

            # next-token ce
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_attn = attn_mask[:, 1:]
            # 1-shift
            shift_loss_mask = loss_mask[:, 1:] * shift_attn  # (B, T-1)

            per_token_loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)

            total_loss = jnp.sum(per_token_loss * shift_loss_mask)
            denom = jnp.maximum(jnp.sum(shift_loss_mask), 1.0)
            loss = jnp.where(denom > 0, total_loss / denom, 0.0)

            loss = jax.lax.with_sharding_constraint(loss, replicated)

            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params=params, updates=updates)

        return params, opt_state, loss

    return train_step


def main() -> None:
    with open("configs/sft_train.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = SFTTrainConfig(**config_dict)

    data_sharding, replicated = create_dp_sharding()
    model, tokenizer = load_model_and_tokenizer_with_vocab_change(model_id=cfg.model_id)

    print("Loaded model/tokenizer.")
    pad_id = tokenizer.pad_token_id
    print(f"Vocab size: {len(tokenizer)} | Pad id: {pad_id} | EOS id {tokenizer.eos_token_id}")

    # get parameter
    params = model.params
    params = jax.device_put(params, replicated)

    optimizer = make_optimizer(lr=cfg.lr, steps=cfg.steps, grad_clip=cfg.grad_clip, wd=cfg.weight_decay)
    # make params f32
    params = cast_tree(params, jnp.float32)
    opt_state = optimizer.init(params)

    train_step = build_train_step(model=model, optimizer=optimizer, pad_id=pad_id, replicated=replicated)

    it = batch_iterator_synthetic_math_sft(tokenizer, cfg, pad_id, seed=cfg.seed)

    # A fixed prompt for monitoring purpose
    rng = np.random.default_rng(123)
    test_example = build_arithmetic_step_data(rng=rng)
    monitor_prompt = test_example["question"]
    monitor_gt = test_example["gt"]

    running_loss: list[float] = []
    for step in range(cfg.steps):
        batch_jnp = next(it)
        # shard batch over data axis
        batch = {
            "input_ids": jax.device_put(batch_jnp["input_ids"], data_sharding),
            "loss_mask": jax.device_put(batch_jnp["loss_mask"], data_sharding),
        }

        params, opt_state, loss = train_step(params, opt_state, batch)
        running_loss.append(float(loss))
        if (step % cfg.log_every) == 0:
            loss_f = np.mean(running_loss)
            if step == 0:
                gen = greedy_generate(
                    model=model,
                    tokenizer=tokenizer,
                    params=params,
                    prompt=monitor_prompt,
                    pad_id=pad_id,
                    max_len=cfg.max_len,
                    max_new_tokens=cfg.max_new_tokens,
                )

                print(f"Step {step:04d} | loss={loss_f:.6f}")
                print(f"Prompt: {monitor_prompt}")
                print(f"Response: {gen}\n | Ground Truth: {monitor_gt} |")
            else:
                print(f"Step {step:04d} | loss={loss_f:.6f}")
            running_loss = []

    print("Final")
    print(f"Prompt: {monitor_prompt}")
    gen = greedy_generate(
        model=model,
        tokenizer=tokenizer,
        params=params,
        prompt=monitor_prompt,
        pad_id=pad_id,
        max_len=cfg.max_len,
        max_new_tokens=cfg.max_new_tokens,
    )
    print(f"Response: {gen}\n | Ground Truth: {monitor_gt} |")
    print("\nDone.")
    save_checkpoint(save_dir=cfg.save_dir + "/sft", model=model, params=params)


if __name__ == "__main__":
    main()
