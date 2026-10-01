import torch

def momentum_step(w, grad, v, lr, mu):
    """One SGD-with-momentum step.

    Args:
        w: parameter tensor
        grad: gradient tensor (same shape as w)
        v: velocity tensor (same shape as w)
        lr: learning rate (float)
        mu: momentum coefficient (float)

    Returns:
        (w_new, v_new) tuple of tensors
    """
    v_new = mu * v + grad
    w_new = w - lr * v_new
    return (w_new, v_new)

if __name__ == "__main__":
    w = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.1, 0.2])
    v = torch.tensor([0.0, 0.0])
    result = momentum_step(w, grad, v, 0.1, 0.9)
    print(result) # Expected output: (tensor([0.9900, 1.9800]), tensor([0.1000, 0.2000]))