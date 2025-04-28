import json
import numpy as np
import random
from utils import bag_of_words, tokenize, stem
# from .model import NumpyNeuralNet

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


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Load intents and saved model parameters
with open('intents.json', 'r', encoding='UTF-8') as f:
    intents = json.load(f)

data = np.load('julAi_brain.npz', allow_pickle=True)
all_words = data['all_words'].tolist()
tags      = data['tags'].tolist()

# Instantiate and load weights
n_input  = data['W1'].shape[0]
n_hidden = data['W1'].shape[1]
n_output = data['W3'].shape[1]
model    = NumpyNeuralNet(n_input, n_hidden, n_output)
model.W1, model.b1 = data['W1'], data['b1']
model.W2, model.b2 = data['W2'], data['b2']
model.W3, model.b3 = data['W3'], data['b3']

# Fallback responses when confidence is low
not_responses = [
    "I'm sorry, I didn't quite catch that. Could you please rephrase your question or provide more context?",
    "I'm still learning, and I might not fully understand what you're asking. Could you please try rephrasing or providing more details?",
    "It seems like I'm having trouble understanding your question. Can you please try phrasing it differently?",
    # ... (keep the same list of fallback replies) ...
]

bot_name = "julAi"
print("Type 'q' to exit")

while True:
    sentence = input("GAB: ")
    if sentence.lower() == 'q':
        break

    tokens = tokenize(sentence)
    stems = [stem(w) for w in tokens]
    bow = bag_of_words(stems, all_words)
    X = bow.reshape(1, -1)

    # Forward pass
    scores = model.forward(X)
    probs  = softmax(scores)
    pred_i = np.argmax(probs, axis=1)[0]
    tag     = tags[pred_i]
    confidence = probs[0, pred_i]

    # Choose a response
    print(confidence)
    if confidence > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(f"\n{bot_name}: {response}\n")
                break
    else:
        print(f"\n{bot_name}: {random.choice(not_responses)}\n")
