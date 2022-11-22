import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


# scroll to the bottom to start coding your solution
def plot(loss_history: list, accuracy_history: list, filename="plot"):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title("Loss on train dataframe from epoch")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title("Accuracy on test dataframe from epoch")
    plt.grid()

    plt.savefig(f"{filename}.png")


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def scale(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    max_val = max(X_train.max(), X_test.max())
    return X_train / max_val, X_test / max_val


def xavier(n_in: int, n_out: int) -> np.ndarray:
    val = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-val, val, (n_in, n_out))


def sigmoid(x: int | float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-x))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return np.mean((x - y) ** 2)


def mse_der(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (x - y)


def sigmoid_der(x: int | float | np.ndarray) -> float | np.ndarray:
    f = sigmoid(x)
    return f * (1 - f)


class OneLayerNeural:
    def __init__(self, n_features: int, n_classes: int):
        self.weights = xavier(n_features, n_classes)
        self.bias = xavier(1, n_classes)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights + self.bias)

    def backprop(self, X: np.ndarray, y: np.ndarray, alpha: float):
        yp = self.forward(X)
        db = alpha * ((yp - y) * yp * (1 - yp))
        self.weights -= X.T @ db
        self.bias -= np.ones((1, X.shape[0])) @ db


if __name__ == "__main__":

    if not os.path.exists("../Data"):
        os.mkdir("../Data")

    # Download data if it is unavailable.
    if "fashion-mnist_train.csv" not in os.listdir(
        "../Data"
    ) and "fashion-mnist_test.csv" not in os.listdir("../Data"):
        print("Train dataset loading.")
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open("../Data/fashion-mnist_train.csv", "wb").write(r.content)
        print("Loaded.")

        print("Test dataset loading.")
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open("../Data/fashion-mnist_test.csv", "wb").write(r.content)
        print("Loaded.")

    # Read train, test data.
    raw_train = pd.read_csv("../Data/fashion-mnist_train.csv")
    raw_test = pd.read_csv("../Data/fashion-mnist_test.csv")

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train["label"].values)
    y_test = one_hot(raw_test["label"].values)

    # write your code here
    X_train, X_test = scale(X_train, X_test)
    n_feats = X_train.shape[1]

    net = OneLayerNeural(n_feats, 10)
    net.backprop(X_train[:2], y_train[:2], alpha=0.1)
    y_pred = net.forward(X_train[:2])
    r1 = (
        mse(a1 := np.array([-1, 0, 1, 2]), a2 := np.array([4, 3, 2, 1]))
        .flatten()
        .tolist()
    )
    r2 = mse_der(a1, a2).flatten().tolist()
    r3 = sigmoid_der(a1).flatten().tolist()
    r4 = mse(y_pred, y_train[:2]).flatten().tolist()
    print(r1, r2, r3, r4)
