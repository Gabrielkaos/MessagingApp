import numpy as np

class NumpyNeuralNet:
    def __init__(self, n_input, n_hidden, n_output, weight_scale=0.01, seed=None):
        """
        A simple three-layer feedforward neural network implemented in NumPy.

        Args:
            n_input (int):  Size of input layer.
            n_hidden (int): Size of each hidden layer.
            n_output (int): Size of output layer.
            weight_scale (float): Standard deviation for random weight initialization.
            seed (int, optional): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and biases
        self.W1 = np.random.randn(n_input,  n_hidden)  * weight_scale
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden,  n_hidden)  * weight_scale
        self.b2 = np.zeros((1, n_hidden))
        self.W3 = np.random.randn(n_hidden,  n_output) * weight_scale
        self.b3 = np.zeros((1, n_output))

    @staticmethod
    def relu(x):
        """ReLU activation"""
        return np.maximum(0, x)

    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (np.ndarray): Input array of shape (batch_size, n_input).
        
        Returns:
            np.ndarray: Output scores of shape (batch_size, n_output).
        """
        # Layer 1
        z1 = X.dot(self.W1) + self.b1
        a1 = self.relu(z1)

        # Layer 2
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.relu(z2)

        # Output layer (no activation)
        out = a2.dot(self.W3) + self.b3
        return out
