import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the total discounted return for a sequence of rewards.
    Args:
        rewards (list or np.ndarray): List or array of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return
    """
    discounted_rewards = [rewards[t] * gamma**t for t in range(len(rewards))]
    return sum(discounted_rewards)

if __name__ == "__main__":
    rewards = [1, 1, 1, 1, 1]
    gamma = 0.9
    result = round(discounted_return(rewards, gamma), 4)
    print(result) # Expected output: 4.0951