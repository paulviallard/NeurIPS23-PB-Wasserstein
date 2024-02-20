import h5py
import numpy as np
import os
import pandas as pd
import random
import requests
import warnings
from sklearn.datasets import load_svmlight_files

###############################################################################


def get_label(label):
    label = label.astype(int)
    label = label.to_numpy()
    label = label[:, 0]
    return label


def get_input(input):
    input = input.to_numpy().astype(np.float32)
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

    if(not(os.path.exists("data-phishing/"))
       or not(os.path.exists("data-phishing/phishing"))):

        if(not(os.path.exists("data-phishing"))):
            os.mkdir("data-phishing")

        r = requests.get(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
            + "phishing",
            allow_redirects=True)
        f = open("data-phishing/phishing", "wb")
        f.write(r.content)
        f.close()

    data_list = load_svmlight_files(["data-phishing/phishing"])
    input = pd.DataFrame(data_list[0].toarray())
    label = pd.DataFrame(data_list[1])

    input = get_input(input)
    label = get_label(label)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, 0.5)

    input_train = input_train[1:]
    label_train = label_train[1:]

    save("phishing.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
