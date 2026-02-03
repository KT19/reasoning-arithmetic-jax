import optax


def make_optimizer(lr: float, steps: int, grad_clip: float, wd: float) -> optax.GradientTransformation:
    warmup = int(steps * 0.1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup, decay_steps=steps, end_value=0.1 * lr
    )
    return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(learning_rate=schedule, weight_decay=wd))
