import numpy as np

def create_bandit_testbed(k: int, num_pulls: int, seed: int = 42) -> tuple:
    """
    Build a k-armed bandit testbed and simulate pulling each arm.
    
    Args:
        k: Number of arms
        num_pulls: Number of times to pull each arm
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (true_values, sample_means, optimal_arm)
    """
    np.random.seed(seed)

    true_values = np.random.randn(k)
    sample_means = []

    for arm in range(k):
        rewards = true_values[arm] + np.random.randn(num_pulls)
        sample_means.append(round(float(np.mean(rewards)), 4))

    optimal_arm = int(np.argmax(true_values))
    true_values = np.round(true_values, 4).tolist()

    return true_values, sample_means, optimal_arm

if __name__ == "__main__":
    result = create_bandit_testbed(3, 1000, seed=42)
    print(result) # Expected output: ([0.4967, -0.1383, 0.6477], [0.5174, -0.0714, 0.652], 2)