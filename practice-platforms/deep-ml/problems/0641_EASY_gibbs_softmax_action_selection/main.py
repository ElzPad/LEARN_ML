import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    """
    Perform Gibbs softmax (Boltzmann) action selection.

    Args:
        q_values: list of floats, estimated action values
        temperature: float, temperature parameter (tau > 0)
        seed: int, random seed for reproducibility

    Returns:
        tuple: (probabilities as list of floats, selected action as int)
    """
    q_values = np.asarray(q_values, dtype=float)
    scaled_values = (q_values - np.max(q_values)) / temperature
    exp_values = np.exp(scaled_values)
    denominator = np.sum(exp_values)
    probabilities = exp_values / denominator

    np.random.seed(seed)
    action = np.random.choice(len(q_values), p=probabilities)

    return (probabilities.tolist(), action)

if __name__ == "__main__":
    probs, action = gibbs_softmax_action_selection([0.0, 1.0], 1.0, 0)
    print([round(p, 4) for p in probs], action) # Expected output: [0.2689, 0.7311] 1