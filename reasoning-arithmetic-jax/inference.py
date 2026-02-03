import numpy as np
import yaml

from train.grpo_train_config import GRPOTrainConfig
from utils.loading import load_model_and_tokenizer_with_vocab_change, load_params
from utils.sampling import greedy_generate
from utils.synthetic_math import build_arithmetic_step_data


def main() -> None:
    with open("configs/grpo_train.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = GRPOTrainConfig(**config_dict)

    model, tokenizer = load_model_and_tokenizer_with_vocab_change(cfg.model_id)
    params = load_params(cfg.save_dir + "/grpo", model.params)
    pad_id = tokenizer.pad_token_id

    print("Loaded model/tokenizer.")
    print(f"Vocab size: {len(tokenizer)} | Pad id: {pad_id}")

    # A fixed prompt for monitoring purpose
    rng = np.random.default_rng(1)
    q_num = 10
    print(f"--- {q_num} questions are given --- \n")
    for _ in range(q_num):
        test_example = build_arithmetic_step_data(rng=rng)
        test_prompt = test_example["question"]
        test_gt = test_example["gt"]

        print(f"Question: {test_prompt} | GT: {test_gt}")
        response = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            params=params,
            prompt=test_prompt,
            pad_id=pad_id,
            max_len=cfg.max_len,
            max_new_tokens=cfg.max_new_tokens,
        )
        print(f"Response:\n{response}\n")


if __name__ == "__main__":
    main()
