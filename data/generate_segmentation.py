import h5py
import numpy as np
import os
import pandas as pd
import random
import requests
import warnings

###############################################################################


def get_label(data):
    label = data[0]
    label_unique = list(np.unique(data[0]))
    label_unique.sort()
    for i, unique in enumerate(label_unique):
        label.loc[label == unique] = i
    label = label.astype(int)
    label = label.to_numpy()
    return label


def get_input(data):
    input = data.drop([0, 3], axis=1)
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

    if(not(os.path.exists("data-segmentation/"))
       or not(os.path.exists("data-segmentation/segmentation.data"))
       or not(os.path.exists("data-segmentation/segmentation.test"))):

        if(not(os.path.exists("data-segmentation"))):
            os.mkdir("data-segmentation")

        r_1 = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/image/"
            + "segmentation.data", allow_redirects=True)
        r_2 = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/image/"
            + "segmentation.test", allow_redirects=True)
        f_1 = open("data-segmentation/segmentation.data", "wb")
        f_2 = open("data-segmentation/segmentation.test", "wb")
        f_1.write(r_1.content)
        f_2.write(r_2.content)
        f_1.close()
        f_2.close()

    data = pd.read_csv(
        "data-segmentation/segmentation.data", sep=",",
        header=None, skiprows=5)
    data_test = pd.read_csv(
        "data-segmentation/segmentation.test", sep=",",
        header=None, skiprows=5)

    label_train = get_label(data)
    label_test = get_label(data_test)

    data_train_test = pd.concat([data, data_test])
    input_train_test = get_input(data_train_test)
    label_train_test = np.concatenate([label_train, label_test])

    (input_train, input_test, label_train, label_test) = get_train_test(
        input_train_test, label_train_test, 0.5)
    save("segmentation.h5", input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
