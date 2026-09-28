import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if not training:
            return x
        
        self.mask  = np.random.binomial(1, 1-self.p, x.shape) >= self.p
        scale = 1.0 / (1.0 - self.p)
        return x * self.mask * scale

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        scale = 1.0 / (1.0 - self.p)
        return grad * self.mask * scale

if __name__ == "__main__":
    np.random.seed(42)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.5, 0.2], [1.0, 2.0]])

    dropout = DropoutLayer(0.2)

    print(dropout.forward(x, training=True), dropout.forward(x, training=False), dropout.backward(grad))
        # Expected output: (array([[1.25, 0.  ], [3.75, 5.  ]]), array([[1., 2.], [3., 4.]]), array([[0.625, 0.   ], [1.25 , 2.5  ]]))