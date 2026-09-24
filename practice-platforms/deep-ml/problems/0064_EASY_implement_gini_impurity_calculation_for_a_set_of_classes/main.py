import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	impurity = 1
	class_labels = np.unique(y)

	for label in class_labels:
		impurity -= ((y == label).mean()) ** 2
	return impurity

if __name__ == "__main__":
    y = [0, 1, 1, 1, 0]
    print(gini_impurity(y)) # Expected ouput: 0.48

    y = [0, 1, 2, 2, 2, 1, 2]
    print(gini_impurity(y)) # Expected output: 0.571
