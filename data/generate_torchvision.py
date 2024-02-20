import os
import h5py
import random
import torch
import warnings
import torchvision
import torchvision.datasets
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import inspect


def call_fun(fun, kwargs):

    kwargs = dict(kwargs)

    fun_param = list(inspect.signature(fun.__init__).parameters.keys())
    for key in list(kwargs.keys()):
        if(key not in fun_param):
            del kwargs[key]
    return fun(**kwargs)


def get_label(label):
    label = label.astype(int)
    return label


def get_input(input):
    input = np.reshape(input, (input.shape[0], -1))
    input = input.astype(np.float32)
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
    torch.manual_seed(42)
    warnings.filterwarnings("ignore")

    arg_parser = ArgumentParser(
        description="generate a torchvision dataset")
    arg_parser.add_argument(
        "dataset", metavar="dataset", type=str,
        help="name of the dataset"
    )
    arg_list = arg_parser.parse_args()

    dataset = arg_list.dataset

    if(os.path.exists(dataset)):
        input_label_train = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/train",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        input_label_test = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/test",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        test_size = len(input_label_test)
        train_size = len(input_label_train)
    else:
        dataset_fun = None
        _locals = locals()
        exec("dataset_fun = torchvision.datasets."+str(dataset),
             globals(), _locals)
        dataset_fun = _locals["dataset_fun"]
        kwargs = {
            "root": "./data-"+dataset,
            "train": True,
            "download": True,
            "split": "train",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        }

        input_label_train = call_fun(dataset_fun, kwargs)
        if("train" in kwargs):
            kwargs["train"] = False
        if("split" in kwargs):
            kwargs["split"] = "test"
        input_label_test = call_fun(dataset_fun, kwargs)

        test_size = input_label_test.data.shape[0]
        train_size = input_label_train.data.shape[0]

    train_loader = DataLoader(
        input_label_train,
        batch_size=train_size)
    test_loader = DataLoader(
        input_label_test, batch_size=test_size)
    input_label_train = list(train_loader)
    input_label_test = list(test_loader)
    input_train = input_label_train[0][0]
    label_train = input_label_train[0][1]
    input_test = input_label_test[0][0]
    label_test = input_label_test[0][1]

    data_train_test = np.concatenate([input_train, input_test])
    input_train_test = get_input(data_train_test)
    label_train_test = np.concatenate([label_train, label_test])
    label_train_test = get_label(label_train_test)

    (input_train, input_test, label_train, label_test) = get_train_test(
        input_train_test, label_train_test, 0.5)

    save(dataset.lower()+".h5",
         input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
