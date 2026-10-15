import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    action = 0
    if np.random.random() < epsilon:
        action = np.random.randint(len(Q))
    else:
        action = np.random.choice(np.flatnonzero(Q == np.max(Q)))
    return action

if __name__ == "__main__":
    np.random.seed(0)
    result = [epsilon_greedy(np.array([1, 2, 3]), epsilon=0.0) for _ in range(5)]
    print(result) # Expected output: [2, 2, 2, 2, 2]