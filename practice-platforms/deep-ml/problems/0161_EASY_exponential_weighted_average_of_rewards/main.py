import numpy as np

def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    k = len(rewards)

    initial_term = (1 - alpha) ** k * Q1

    weighted_sum = sum(
        alpha * (1 - alpha) ** (k - i) * reward
        for i, reward in enumerate(rewards, start=1)
    )

    return float(initial_term + weighted_sum)

if __name__ == "__main__":
    Q1 = 10.0
    rewards = [4.0, 7.0, 13.0]
    alpha = 0.5
    result = round(exp_weighted_average(Q1, rewards, alpha), 4)
    print(result) # Expected output: 10.0