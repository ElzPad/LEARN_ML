import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the discounted return for a given list of rewards.
    Args:
      rewards (list of float): sequence of rewards R_{t+1}, R_{t+2}, ...
      gamma (float): discount factor (0 <= gamma <= 1)
    Returns:
      float: discounted return G_t
    """
    discounted_rewards = [rewards[t] * gamma**t for t in range(len(rewards))]
    return sum(discounted_rewards)

if __name__ == "__main__":
    rewards = [1, 2, 3, 4]
    gamma = 0.9
    result = round(discounted_return(rewards, gamma), 4)
    print(result) # Expected output: 8.146