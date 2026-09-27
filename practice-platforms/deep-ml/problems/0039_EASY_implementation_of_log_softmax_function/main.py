import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    """
    Compute the log-softmax of a sequence of scores.

    Log-softmax converts input scores into log probabilities in a
    numerically stable way by subtracting the maximum score before
    computing the log-sum-exp.

    Args:
        scores (list): A list of numeric scores.

    Returns:
        np.ndarray: An array containing the log-softmax value for each
        input score.
    """
    normalized_scores = scores-np.max(scores)
    normalized_scores_logSumExp = np.log(np.exp(normalized_scores).sum())
    return normalized_scores - normalized_scores_logSumExp

if __name__ == "__main__":
    result = np.round(log_softmax([1, 2, 3]), 4)
    print(result) # Expected output: [-2.4076, -1.4076, -0.4076]