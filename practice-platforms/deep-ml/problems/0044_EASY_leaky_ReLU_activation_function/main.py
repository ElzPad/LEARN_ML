def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    """Return the Leaky ReLU activation of z using alpha for negative values."""
    return max(alpha * z, z)

if __name__ == "__main__":
    print(leaky_relu(0))  # Expected output: 0
    print(leaky_relu(1))  # Expected output: 1
    print(leaky_relu(-1)) # Expected output: -0.01
    print(leaky_relu(-2, alpha=0.1))  # Expected output: -0.2