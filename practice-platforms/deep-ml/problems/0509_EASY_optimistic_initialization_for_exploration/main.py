import numpy as np

def optimistic_greedy_bandit(
    true_rewards: list,
    initial_q: float,
    n_steps: int,
    step_size: float
) -> tuple:
    """
    Simulate a greedy bandit agent with optimistic initialization.
    
    Args:
        true_rewards: List of true deterministic rewards for each arm
        initial_q: Optimistic initial Q-value for all arms
        n_steps: Number of steps to simulate
        step_size: Constant step-size (alpha) for Q-value updates
    
    Returns:
        Tuple of (Q_values, action_counts) where Q_values is a list of
        floats rounded to 4 decimal places, and action_counts is a list of ints.
    """
    n_arms = len(true_rewards)
    estimated_values = np.ones((n_arms,)) * initial_q
    action_counts = [0] * n_arms

    for _ in range(n_steps):
        action = np.random.choice(np.flatnonzero(estimated_values == np.max(estimated_values)))
        reward = true_rewards[action]
        action_counts[action] += 1

        estimated_values[action] += step_size * (reward - estimated_values[action])

    return np.round(estimated_values, 4).tolist(), action_counts

if __name__ == "__main__":
    result = optimistic_greedy_bandit([1.0, 2.0, 3.0], 5.0, 6, 0.1)
    print(result) # Expected output: ([4.6, 4.43, 4.458], [1, 2, 3])