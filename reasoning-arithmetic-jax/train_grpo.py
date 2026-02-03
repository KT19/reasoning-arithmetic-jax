from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from tqdm import tqdm

from train.grpo_train_config import GRPOTrainConfig
from train.make_optimizer import make_optimizer
from utils.batch_iterator import batch_iterator_synth_arith_rl
from utils.functions import cast_tree, gather_logp_next, logits_from_ids, save_checkpoint
from utils.loading import load_model_and_tokenizer_with_vocab_change, load_params
from utils.sampling import greedy_generate
from utils.sharding import create_dp_sharding
from utils.synthetic_math import build_arithmetic_step_data


def build_train_step_grpo(
    model: Any,
    optimizer: optax.GradientTransformation,
    pad_id: int,
    replicated: jax.NamedSharding,
    clip_eps: float,
    kl_beta: float,
) -> Any:
    """
    input_ids: (N, T)
    comp_mask: (N, T)
    advantages: (N)
    """

    @jax.jit
    def train_step(
        params: Any,
        opt_state: Any,
        params_ref: Any,
        batch: dict[str, jax.Array],
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        input_ids = batch["input_ids"]  # (N, T)
        comp_mask = batch["comp_mask"]  # (N, T)
        adv = batch["advantages"]  # (N, 1)

        shift_comp = comp_mask[:, 1:]
        shift_labels = input_ids[:, 1:]

        logits_ref = logits_from_ids(model, params_ref, input_ids, pad_id)[:, :-1, :]  # (N, T-1, V)
        logp_ref = gather_logp_next(logits_ref, shift_labels)
        logp_ref = jax.lax.stop_gradient(logp_ref)

        # old
        logits_old = logits_from_ids(model, params, input_ids, pad_id)[:, :-1]
        logp_old = gather_logp_next(logits_old, shift_labels)
        logp_old = jax.lax.stop_gradient(logp_old)

        adv = jax.lax.stop_gradient(adv)
        adv_token = adv * shift_comp  # broadcast to (N, T-1)

        def loss_fn(p: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            # policy logits
            logits_policy = logits_from_ids(model, p, input_ids, pad_id)[:, :-1, :]  # (N, T-1, V)
            logp_policy = gather_logp_next(logits_policy, shift_labels)

            ratio = jnp.exp(logp_policy - logp_old)

            # PPO clip
            clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            # objective per token
            obj1 = ratio * adv_token
            obj2 = clipped * adv_token
            obj = jnp.minimum(obj1, obj2)

            # Normalize
            denom = jnp.maximum(shift_comp.sum(), 1.0)
            ppo_loss = -(obj.sum() / denom)

            # KL penalty
            kl_token = (logp_policy - logp_ref) * shift_comp
            kl = kl_token.sum() / denom

            loss = ppo_loss + kl_beta * kl
            loss = jax.lax.with_sharding_constraint(loss, replicated)

            metrics = {
                "loss": loss,
                "ppo_loss": ppo_loss,
                "kl": kl,
                "mean_ratio": (ratio * shift_comp).sum() / denom,
                "mean_adv": (adv_token.sum() / denom),
            }

            metrics = jax.tree_util.tree_map(lambda x: jax.lax.with_sharding_constraint(x, replicated), metrics)

            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, metrics

    return train_step


def main() -> None:
    with open("configs/grpo_train.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = GRPOTrainConfig(**config_dict)

    data_sharding, replicated = create_dp_sharding()
    model, tokenizer = load_model_and_tokenizer_with_vocab_change(cfg.model_id)
    params_ref = load_params(cfg.save_dir + "/sft", model.params)
    params = load_params(cfg.save_dir + "/sft", model.params)

    pad_id = tokenizer.pad_token_id

    print("Loaded model/tokenizer.")
    print(f"Vocab size: {len(tokenizer)} | Pad id: {pad_id}")

    # get reference parameter
    params_ref = cast_tree(params_ref, jnp.float32)
    params_ref = jax.device_put(params_ref, replicated)

    # get parameter
    params = cast_tree(params, jnp.float32)
    params = jax.device_put(params, replicated)

    optimizer = make_optimizer(lr=cfg.lr, steps=cfg.steps, grad_clip=cfg.grad_clip, wd=cfg.weight_decay)
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state, replicated)

    train_step = build_train_step_grpo(
        model=model,
        optimizer=optimizer,
        pad_id=pad_id,
        replicated=replicated,
        clip_eps=cfg.ppo_clip_eps,
        kl_beta=cfg.kl_beta,
    )

    # A fixed prompt for monitoring purpose
    rng = np.random.default_rng(123)
    test_example = build_arithmetic_step_data(rng=rng)
    monitor_prompt = test_example["question"]
    monitor_gt = test_example["gt"]

    running_loss: list[float] = []
    running_rewards: list[float] = []
    running_ppo: list[float] = []
    running_kl: list[float] = []

    log_rewards: list[float] = []

    for step in tqdm(range(cfg.steps)):
        ids_np, comp_np, r_np = batch_iterator_synth_arith_rl(
            seed=step + cfg.seed,
            model=model,
            tokenizer=tokenizer,
            params=params,
            pad_id=pad_id,
            cfg=cfg,
        )

        # Group-relative advantages
        B, K = cfg.prompts_per_step, cfg.group_size

        r = r_np.reshape(B, K)
        mean = r.mean(axis=1, keepdims=True)
        adv = (r - mean).flatten()[:, np.newaxis].astype(np.float32)  # Data sharding (2D)

        # logging purpose
        log_rewards.append(mean.mean())

        # shard batch over data axis
        batch = {
            "input_ids": jax.device_put(jnp.asarray(ids_np, jnp.int32), data_sharding),
            "comp_mask": jax.device_put(jnp.asarray(comp_np, jnp.float32), data_sharding),
            "advantages": jax.device_put(jnp.asarray(adv, jnp.float32), data_sharding),
        }

        params, opt_state, metrics = train_step(params, opt_state, params_ref, batch)

        running_rewards.append(float(np.mean(r)))
        running_loss.append(float(metrics["loss"]))
        running_ppo.append(float(metrics["ppo_loss"]))
        running_kl.append(float(metrics["kl"]))

        if step % cfg.log_every == 0:
            r_mean = np.mean(running_rewards)
            r_std = np.std(running_rewards)
            loss_mean = np.mean(running_loss)
            loss_ppo = np.mean(running_ppo)
            loss_kl = np.mean(running_kl)
            print(
                f"[GRPO] step {step:04d} | R_mean={r_mean:.3f} | R_std={r_std:.3f}\n"
                f"loss={loss_mean:.6f} | ppo ={loss_ppo:.6f} | kl={loss_kl:.6f}"
            )

            running_loss, running_rewards, running_ppo, running_kl = [], [], [], []

        if (step % cfg.monitor_answer_step) == 0:
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

    print("Final")
    gen = greedy_generate(
        model=model,
        tokenizer=tokenizer,
        params=params,
        prompt=monitor_prompt,
        pad_id=pad_id,
        max_len=cfg.max_len,
        max_new_tokens=cfg.max_new_tokens,
    )
    print(f"Ground Truth: {monitor_gt} | Response:\n {gen}")
    save_checkpoint(cfg.save_dir + "/grpo", model, params)
    print("\nDone.")

    x = np.arange(cfg.steps)
    plt.plot(x, log_rewards)
    plt.title("GRPO reward")
    plt.legend()
    plt.savefig("grpo_reward.png")


if __name__ == "__main__":
    main()
