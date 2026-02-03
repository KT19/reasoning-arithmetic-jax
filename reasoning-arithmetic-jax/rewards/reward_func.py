import re


def format_reward_func(completion: str) -> float:
    """
    Rewards the format
    <think>...</think>
    <answer>...</answer>
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    match = re.search(pattern, completion, re.DOTALL)

    return 1.0 if match else 0.0


def correctness_reward_func(completion: str, answer: str) -> float:
    """
    Rewards the model if the integer inside <answer></answer>
    the ground truth.
    """

    match = re.search("<answer>(.*?)</answer>", completion)
    if match:
        predicted_ans = match.group(1).strip()
        return 2.0 if predicted_ans == answer else 0.0

    return 0.0
