import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


# scroll to the bottom to start coding your solution


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def scale(X_train, X_test):
    max_val = max(X_train.max(), X_test.max())
    return X_train / max_val, X_test / max_val


def xavier(n_in, n_out):
    val = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-val, val, (n_in, n_out))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    r1 = [X_train[2, 778], X_test[0, 774]]
    r2 = xavier(2, 3).flatten().tolist()
    r3 = sigmoid(np.array([-1, 0, 1, 2])).flatten().tolist()

    print(r1, r2, r3)
