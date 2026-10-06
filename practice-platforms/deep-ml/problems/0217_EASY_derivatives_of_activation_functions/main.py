import math

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	tanh = math.tanh(x)
	sigmoid = 1 / (1 + math.exp(-x))

	return {
		"sigmoid": sigmoid * (1 - sigmoid),
		"tanh": 1 - tanh**2,
		"relu": 1.0 if x > 0 else 0.0,
	}

if __name__ == "__main__":
    result = activation_derivatives(0.0)
    print({k: round(v, 4) for k, v in result.items()}) # Expected output: {'sigmoid': 0.25, 'tanh': 1.0, 'relu': 0.0}