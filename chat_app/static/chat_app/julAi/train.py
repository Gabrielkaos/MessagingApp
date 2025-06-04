import json
import numpy as np
from utils import bag_of_words, stem, tokenize

class NumpyNeuralNet:
    def __init__(self, n_input, n_hidden, n_output, weight_scale=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(n_input, n_hidden) * weight_scale
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_hidden) * weight_scale
        self.b2 = np.zeros((1, n_hidden))
        self.W3 = np.random.randn(n_hidden, n_output) * weight_scale
        self.b3 = np.zeros((1, n_output))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        scores = self.a2.dot(self.W3) + self.b3
        return scores

    def backward(self, X, y, scores):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = X.shape[0]

        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        self.dW3 = self.a2.T.dot(dscores)
        self.db3 = np.sum(dscores, axis=0, keepdims=True)

        da2 = dscores.dot(self.W3.T)
        dz2 = da2 * (self.z2 > 0)

        self.dW2 = self.a1.T.dot(dz2)
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * (self.z1 > 0)

        self.dW1 = X.T.dot(dz1)
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update_params(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3


def get_data_from_intents():
    with open("intents.json", "r", encoding="UTF-8") as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    ignore = ["?", ",", "!", ".", "'"]
    all_words = sorted(set(stem(w) for w in all_words if w not in ignore))
    tags = sorted(set(tags))

    X_train, y_train = [], []
    for (tokens, tag) in xy:
        bow = bag_of_words(tokens, all_words)
        X_train.append(bow)
        y_train.append(tags.index(tag))

    return np.array(X_train), np.array(y_train), tags, all_words


def save_data(file_path, model, all_words, tags):
    np.savez(file_path,
             W1=model.W1, b1=model.b1,
             W2=model.W2, b2=model.b2,
             W3=model.W3, b3=model.b3,
             all_words=all_words, tags=tags)
    print(f"\nTRAINING COMPLETE, saved to: '{file_path}.npz'")


if __name__ == '__main__':
    batch_size = 4
    n_hidden = 100
    lr = 2e-1
    num_epochs = 1000

    X_train, y_train, tags, all_words = get_data_from_intents()
    n_input = X_train.shape[1]
    n_output = len(tags)

    model = NumpyNeuralNet(n_input, n_hidden, n_output, seed=42, weight_scale=0.1)

    for epoch in range(1, num_epochs + 1):
        perm     = np.random.permutation(len(X_train))
        X_shuf   = X_train[perm]
        y_shuf   = y_train[perm]

        for i in range(0, len(X_shuf), batch_size):
            X_batch = X_shuf[i : i + batch_size]
            y_batch = y_shuf[i : i + batch_size]
            N       = X_batch.shape[0]

            scores = model.forward(X_batch)

            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs      = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            loss       = -np.mean(np.log(probs[np.arange(N), y_batch]))

            model.backward(X_batch, y_batch, scores)
            model.update_params(lr)

            assert probs.shape == (N, n_output)
            assert y_batch.shape[0] == N

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss:.5f}")
        if loss < 1e-4:
            print(f"\n[TARGET REACHED] Epoch: {epoch}/{num_epochs}, Loss: {loss:.5f}\n")
            break


    print(f"Final Loss: {loss:.4f}")
    save_data("julAi_brain", model, all_words, tags)
