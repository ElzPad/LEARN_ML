import math
import numpy as np

def unet_time_embedding(timesteps: list, embed_dim: int, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, max_period: int = 10000) -> np.ndarray:
    """
    Compute time embeddings for a diffusion model U-Net.
    
    Args:
        timesteps: list or 1D array of shape (B,) with timestep values
        embed_dim: dimension of sinusoidal embedding (must be even)
        W1: weight matrix of first linear layer, shape (embed_dim, hidden_dim)
        b1: bias of first linear layer, shape (hidden_dim,)
        W2: weight matrix of second linear layer, shape (hidden_dim, output_dim)
        b2: bias of second linear layer, shape (output_dim,)
        max_period: controls the frequency range for sinusoidal embedding
    
    Returns:
        numpy array of shape (B, output_dim) with time embeddings
    """
    d_half = embed_dim // 2
    i = np.arange(d_half)
    timesteps = np.asarray(timesteps, dtype=np.float32).reshape(-1, 1)
    frequencies = np.exp(-np.log(max_period) *i / d_half)
    e = np.hstack([np.sin(timesteps * frequencies), np.cos(timesteps * frequencies)])

    h = silu(e @ W1 + b1)
    res = h @ W2 + b2

    return res

def silu(x):
    return x / (1 + np.exp(-x))

if __name__ == "__main__":
    timesteps = [0]
    embed_dim = 4
    W1 = 0.1 * np.eye(4)
    b1 = np.zeros(4)
    W2 = np.eye(4)
    b2 = np.zeros(4)
    result = unet_time_embedding(timesteps, embed_dim, W1, b1, W2, b2)
    print(np.round(result, 4).tolist()) # Expected output: [[0.0, 0.0, 0.0525, 0.0525]]

    np.random.seed(42)
    timesteps = [0, 100, 500, 999]
    embed_dim = 8
    hidden_dim = 6
    output_dim = 4
    W1 = np.random.randn(embed_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)
    result = unet_time_embedding(timesteps, embed_dim, W1, b1, W2, b2, max_period=1000)
    print(np.round(result, 6).tolist()) # Expected output: [[0.000199, 0.000198, -3.7e-05, -0.000109], [-7.8e-05, 0.000308, 0.000145, 0.000202], [-0.000272, 0.000109, 0.000263, 0.000234], [0.000249, -0.000233, -0.000119, -0.000322]]