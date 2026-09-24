import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    best_impurity, best_feature_index, best_threshold = float("inf"), -1, -1
    total_samples = X.shape[0]

    for feature_id in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_id])

        for threshold in thresholds:
            left_indices = np.where(X[:, feature_id] <= threshold)[0]
            right_indices = np.where(X[:, feature_id] > threshold)[0]

            gini_impurity = \
                (len(left_indices) / total_samples) * compute_binary_gini_impurity(y[left_indices]) \
                + (len(right_indices) / total_samples) * compute_binary_gini_impurity(y[right_indices])

            if gini_impurity < best_impurity:
                best_impurity = gini_impurity
                best_feature_index = feature_id
                best_threshold = threshold
    return best_feature_index, best_threshold

def compute_binary_gini_impurity(y: np.ndarray):
    if y.size == 0:
        return 0

    p_positive = (y == 1).mean()
    p_negative = 1 - p_positive
    return 1 - (p_positive)**2 - (p_negative)**2

if __name__ == "__main__":
    X = np.array([[2.5],[3.5],[1.0],[4.0]])
    y = np.array([0,1,0,1])
    print(find_best_split(X, y))