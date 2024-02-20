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
    label = (label-1).astype(int)
    label = label[:, 0]
    return label


def get_input(data):
    input = data.to_numpy().astype(np.float32)
    index_list = np.asarray(input.min(axis=0) == input.max(axis=0)).nonzero()
    input = np.delete(input, index_list, axis=1)
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

    if(not(os.path.exists("data-mushrooms/"))
       or not(os.path.exists("data-mushrooms/mushrooms"))):

        if(not(os.path.exists("data-mushrooms"))):
            os.mkdir("data-mushrooms")

        r = requests.get(
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
            + "mushrooms",
            allow_redirects=True)
        f = open("data-mushrooms/mushrooms", "wb")
        f.write(r.content)
        f.close()

    data_list = load_svmlight_files(["data-mushrooms/mushrooms"])
    input = pd.DataFrame(data_list[0].toarray())
    label = pd.DataFrame(data_list[1])

    input = get_input(input)
    label = get_label(label)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, 0.5)
    save("mushrooms.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
