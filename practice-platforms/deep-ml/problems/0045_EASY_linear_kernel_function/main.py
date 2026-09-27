import numpy as np

def kernel_function(x1, x2):
	"""
    Compute the linear kernel between two input vectors x1 and x2, defined as the dot product (inner product) of two vectors.
    """
	return np.sum(np.multiply(x1, x2))

if __name__ == "__main__":
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    result = kernel_function(x1, x2)
    print(result) # Expected output: 32