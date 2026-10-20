import numpy as np

def extract_optimal_policy(Q: np.ndarray) -> dict:
    """
    Extract the optimal policy, state-value function, and advantage
    function from a Q-value table.
    
    Args:
        Q: Q-value table of shape (num_states, num_actions)
    
    Returns:
        Dictionary with keys:
        - 'optimal_actions': list of int (optimal action per state)
        - 'state_values': list of float (V*(s) per state)
        - 'advantages': nested list of float (A(s,a) for all pairs)
    """
    optimals = np.argmax(Q, axis=1)
    state_values = Q[np.arange(len(optimals)), optimals]
    advantages = Q - np.reshape(state_values, (state_values.size, 1))

    result = {
        "optimal_actions": np.round(optimals,4).tolist(),
        "state_values": np.round(state_values,4).tolist(),
        "advantages": np.round(advantages,4).tolist(),
    }
    return result

if __name__ == "__main__":
    Q = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [9.0, 8.0, 7.0]])
    result = extract_optimal_policy(Q)
    print(result) # Expected output: {'optimal_actions': [2, 2, 0], 'state_values': [3.0, 6.0, 9.0], 'advantages': [[-2.0, -1.0, 0.0], [-2.0, -1.0, 0.0], [0.0, -1.0, -2.0]]}