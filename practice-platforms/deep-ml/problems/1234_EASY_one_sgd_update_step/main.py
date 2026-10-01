import torch

def sgd_step(w, grad, lr):
    """Perform one SGD update step.

    Args:
        w: Current parameter tensor.
        grad: Gradient tensor (same shape as w).
        lr: Learning rate (float).

    Returns:
        Updated parameter tensor w - lr * grad.
    """
    return w - lr * grad

if __name__ == "__main__":
    result = sgd_step(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0]), 0.1)
    print(result) # Expected output: tensor([0.9500, 1.9000])
