import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    """
    Compute the average cross-entropy loss for multi-class predictions.

    Args:
        predicted_probs: Predicted class probabilities for each sample.
        true_labels: One-hot encoded true labels.
        epsilon: Small value used to clip probabilities for numerical stability.

    Returns:
        The average cross-entropy loss across the batch.
    """
    clipped_predicted_probs = np.clip(predicted_probs, epsilon, 1)
    losses = -np.sum(np.log(clipped_predicted_probs) * true_labels, axis=1)
    result = np.mean(losses)
    return result

if __name__ == "__main__":
    pred = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]
    true = [[1, 0, 0], [0, 1, 0]]
    result = round(compute_cross_entropy_loss(pred, true))
    print(result) # Expected output 0.4338

    pred = np.array([[0.0, 0.5, 0.5], [0.2, 0.8, 0.0]])
    true = np.array([[1, 0, 0], [0, 1, 0]])
    result = round(compute_cross_entropy_loss(pred, true), 4)
    print(result) # Expected output: 