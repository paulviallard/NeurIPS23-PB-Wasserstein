import h5py
import numpy as np
import os
import pandas as pd
import random
import requests
import warnings

###############################################################################


def get_label(data):
    label = data[9]
    label.loc[label == "positive"] = 0
    label.loc[label == "negative"] = 1
    label = label.astype(int)
    label = label.to_numpy()
    return label


def get_input(data):
    data = data.drop([9], axis=1)
    for i in range(9):
        data[i].loc[data[i] == "x"] = 0.0
        data[i].loc[data[i] == "b"] = 0.5
        data[i].loc[data[i] == "o"] = 1.0
    input = data.to_numpy().astype(np.float32)
    input = input/np.max(np.linalg.norm(input, 2, axis=1))
    return input


def get_train_test(input, label, ratio_test):
    size_test = int(ratio_test*len(input))

    input_test = input[:size_test, :]
    input_train = input[size_test:, :]
    label_test = label[:size_test]
    label_train = label[size_test:]

    return input_train, input_test, label_train, label_test


def shuffle(input, label):
    permutation = np.arange(input.shape[0])
    np.random.shuffle(permutation)
    input = input[permutation, :]
    label = label[permutation]
    return input, label


def save(path, input_train, input_test, label_train, label_test):
    dataset_file = h5py.File(path, "w")

    dataset_file["x_train"] = input_train
    dataset_file["y_train"] = label_train
    dataset_file["x_test"] = input_test
    dataset_file["y_test"] = label_test


###############################################################################

def main():
    np.random.seed(42)
    random.seed(42)
    warnings.filterwarnings("ignore")

    if(not(os.path.exists("data-tictactoe/"))
       or not(os.path.exists("data-tictactoe/tictactoe.data"))):

        if(not(os.path.exists("data-tictactoe"))):
            os.mkdir("data-tictactoe")

        r = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            + "tic-tac-toe/tic-tac-toe.data", allow_redirects=True)
        f = open("data-tictactoe/tictactoe.data", "wb")
        f.write(r.content)
        f.close()

    data = pd.read_csv(
        "data-tictactoe/tictactoe.data", sep=",",
        na_values="?", header=None)

    label = get_label(data)
    input = get_input(data)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, 0.5)
    save("tictactoe.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
