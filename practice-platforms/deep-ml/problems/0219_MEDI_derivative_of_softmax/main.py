import math
import numpy

def softmax(scores: list[float]) -> list[float]:
    scores_np = numpy.array(scores)
    scores_npStable = scores_np - numpy.max(scores_np)
    numerator = numpy.exp(scores_npStable)
    denominator = numerator.sum()

    return numerator / denominator

def kronecker_delta(i: int, j: int) -> int:
	return 1 if i==j else 0

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	dim = len(x)
	softmax_values = softmax(x)
	jacobian = [[softmax_values[i] * (kronecker_delta(i,j) - softmax_values[j]) for j in range(dim)] for i in range(dim)]
	return jacobian

if __name__ == "__main__":
    result = softmax_derivative([1.0, 2.0, 3.0])
    print([[round(v, 4) for v in row] for row in result]) # Expected output: [[0.0819, -0.022, -0.0599], [-0.022, 0.1848, -0.1628], [-0.0599, -0.1628, 0.2227]]