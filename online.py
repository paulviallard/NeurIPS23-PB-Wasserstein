import argparse
import logging
import numpy as np
import random
import torch
import os
from h5py import File

from core.nd_data import NDData
from module.module import Module
from learner.online_learner import OnlineLearner
from learner.ogd_learner import OGDLearner

###############################################################################


def main():
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path csv")

    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")

    arg_parser.add_argument(
        "--model_name", metavar="model_name", default="LinearModel", type=str,
        help="model_name")
    arg_parser.add_argument(
        "--depth_size", metavar="depth_size", default=1, type=int,
        help="depth_size")
    arg_parser.add_argument(
        "--width_size", metavar="width_size", default=256, type=int,
        help="width_size")
    arg_parser.add_argument(
        "--iter_size", metavar="iter_size", default=1, type=int,
        help="iter_size")
    arg_parser.add_argument(
        "--ogd_iter_size", metavar="ogd_iter_size", default=-1, type=int,
        help="ogd_iter_size")
    arg_parser.add_argument(
        "--alpha", metavar="alpha", default=100000, type=float,
        help="alpha")

    arg_parser.add_argument(
        "--seed", metavar="seed", default=0, type=int,
        help="seed")

    arg_list = arg_parser.parse_known_args()[0]

    path = arg_list.path

    model_name = arg_list.model_name
    depth_size = arg_list.depth_size
    width_size = arg_list.width_size
    iter_size = arg_list.iter_size
    ogd_iter_size = arg_list.ogd_iter_size
    alpha = arg_list.alpha

    seed = arg_list.seed

    # ----------------------------------------------------------------------- #

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------- #
    # Loading the h5py dataset

    data = File("data/"+arg_list.data+".h5", "r")

    X_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    X_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])

    # ----------------------------------------------------------------------- #
    # Learning the model

    # Learning the model with the bound
    learner = OnlineLearner(
        model_name=model_name,
        iter_size=iter_size,
        alpha=alpha,
        width_size=width_size,
        depth_size=depth_size,
    )
    learner.fit(X_train, y_train)

    # Learning the posterior model from online gradient descent
    ogd_learner = OGDLearner(
        model_name=model_name,
        iter_size=ogd_iter_size,
        alpha=alpha,
        width_size=width_size,
        depth_size=depth_size,
    )
    ogd_learner.fit(X_train, y_train)

    # ----------------------------------------------------------------------- #
    # Evaluating the bound and the test risk

    zero_one_ = Module("OnlineZeroOneLoss")

    model = learner.model
    ogd_model = ogd_learner.model

    learner.forward(X=X_train, y=y_train)
    train_risk = zero_one_(model)
    learner.forward(X=X_test, y=y_test)
    test_risk = zero_one_(model)

    ogd_learner.forward(X=X_train, y=y_train)
    ogd_train_risk = zero_one_(ogd_model)
    ogd_learner.forward(X=X_test, y=y_test)
    ogd_test_risk = zero_one_(ogd_model)

    del learner.model
    del ogd_learner.model

    # ----------------------------------------------------------------------- #
    # Saving

    save_dict = {
        "test_risk": test_risk.item(),
        "train_risk": train_risk.item(),
        "ogd_test_risk": ogd_test_risk.item(),
        "ogd_train_risk": ogd_train_risk.item(),
    }

    dump = {
        "data": arg_list.data,
        "model_name": model_name,
        "depth_size": depth_size,
        "width_size": width_size,
        "iter_size": iter_size,
        "ogd_iter_size": ogd_iter_size,
        "alpha": alpha,
        "seed": seed,
    }

    os.chdir(os.path.dirname(__file__))
    save_data = NDData(os.path.abspath(path))
    save_data.set(save_dict, dump)

###############################################################################


if __name__ == "__main__":
    main()
