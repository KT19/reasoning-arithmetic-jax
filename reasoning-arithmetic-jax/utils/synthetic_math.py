import random
import re
from typing import Any

import numpy as np

from utils.special_tokens import ANS_CLOSE, ANS_OPEN, THINK_CLOSE, THINK_OPEN


def build_arithmetic_step_data(rng: Any) -> dict[str, str]:
    """
    Synthetic data for reasoning
    """
    operators = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]

    # Generate three random numbers
    a, b, c = rng.integers(-100, 100), rng.integers(0, 100), rng.integers(0, 100)
    op1_sign, op1_func = random.choice(operators)
    op2_sign, op2_func = random.choice(operators)

    # Create the logic: (a op1 b) op2 c
    step1_res = op1_func(a, b)
    final_res = op2_func(step1_res, c)

    prompt = f"Calculate ({a} {op1_sign} {b}) {op2_sign} {c}"

    # Reasoning path for SFT
    prob = random.random()
    if prob < 0.2:
        thought_process = (
            f"First, I need to solve the expression inside the parentheses: {a} {op1_sign} {b}. "
            f"{a} {op1_sign} {b} equals {step1_res}. "
            f"Now, I take that result and apply the next operation: {step1_res} {op2_sign} {c}. "
            f"{step1_res} {op2_sign} {c} equals {final_res}."
        )
    elif prob < 0.5:
        thought_process = f"{a} {op1_sign} {b} -> {step1_res}. Apply the next operation and get, {final_res}"
    else:
        thought_process = ""

    reasoning = f"{THINK_OPEN}{thought_process}{THINK_CLOSE}\n" if random.random() < 0.5 else f"{thought_process}\n"
    answer = f"{ANS_OPEN}{final_res}{ANS_CLOSE}" if random.random() < 0.5 else f"{final_res}"

    data = {
        "question": prompt,
        "reasoning": reasoning,
        "answer": answer,
        "gt": f"{final_res}",
    }

    return data


def encode_synthetic_data(
    tokenizer: Any, prompt: str, reasoning: str, answer: str, max_len: int, pad_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    prompt: 0
    other: 1
    """

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    reasoning_ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    # prompt
    ids = prompt_ids
    loss_mask = [0] * len(prompt_ids)

    # reasoning
    ids = ids + reasoning_ids
    loss_mask = loss_mask + [1] * len(reasoning_ids)

    # answer
    ids = ids + answer_ids
    loss_mask = loss_mask + [1] * len(answer_ids)

    # Truncate
    if len(ids) > max_len:
        ids = ids[:max_len]
        loss_mask = loss_mask[:max_len]

    # Pad
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = ids + [pad_id] * pad_n
        loss_mask = loss_mask + [0] * pad_n

    return np.asarray(ids, np.int32), np.asarray(loss_mask, np.int32)


def normalize_text(x: str) -> str:
    x = x.strip()
    x = re.sub(r"\s+", " ", x)

    return x


def extract_answer_from_text(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not m:
        return ""
    return normalize_text(m.group(1))
