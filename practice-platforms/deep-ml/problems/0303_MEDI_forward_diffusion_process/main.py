import math
import numpy as np

def forward_diffusion(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    alpha = np.linspace(1-beta_start, 1-beta_end, num_timesteps)
    alpha_bar = np.cumprod(alpha)

    x_t = math.sqrt(alpha_bar[t-1]) * x_0 + math.sqrt(1-alpha_bar[t-1]) * noise
    return x_t

if __name__ == "__main__":

    result = forward_diffusion(np.array([1.0]), 1, 0.1, 0.2, 5, np.array([1.0]))
    print(np.round(result, 4).tolist()) # Expected output: [1.2649]

    result = forward_diffusion(np.array([1.0, 2.0]), 3, 0.1, 0.2, 5, np.array([0.5, -0.5]))
    print(np.round(result, 4).tolist()) # Expected output: [1.1057, 1.3488]