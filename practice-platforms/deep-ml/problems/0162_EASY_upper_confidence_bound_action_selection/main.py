import numpy as np

def ucb_action(counts, values, t, c):
    """
    Choose an action using the UCB1 formula.
    Args:
        counts (np.ndarray): Number of times each action has been chosen
        values (np.ndarray): Average reward of each action
        t (int): Current timestep (starts from 1)
        c (float): Exploration coefficient
    Returns:
        int: Index of action to select
    """
    ucb_values = values + c * np.sqrt(np.log(t) / counts)
    action = np.argmax(ucb_values)

    return action

if __name__ == "__main__":
    counts = np.array([1, 1, 1, 1])  # Each action tried once
    values = np.array([1.0, 2.0, 1.5, 0.5])
    t = 4
    c = 2.0
    result = ucb_action(counts, values, t, c)
    print(result) # Expected output: 1