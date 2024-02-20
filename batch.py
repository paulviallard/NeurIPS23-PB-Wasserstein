import argparse
import logging
import numpy as np
import random
import torch
import os
from h5py import File
import math

from core.nd_data import NDData
from module.module import Module
from learner.batch_learner import BatchLearner
from learner.erm_learner import ERMLearner

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
        "--batch_size", metavar="batch_size", default=100, type=int,
        help="batch_size")
    arg_parser.add_argument(
        "--iter_prior_size", metavar="iter_prior_size", default=1, type=int,
        help="iter_prior_size")
    arg_parser.add_argument(
        "--iter_post_size", metavar="iter_post_size", default=1, type=int,
        help="iter_post_size")
    arg_parser.add_argument(
        "--alpha", metavar="alpha", default=100000, type=float,
        help="alpha")

    arg_parser.add_argument(
        "--ratio_set_size", metavar="ratio_set_size", default=1.0, type=float,
        help="ratio_set_size")
    arg_parser.add_argument(
        "--epsilon", metavar="epsilon", default="m", type=str,
        help="epsilon")

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
        "--seed", metavar="seed", default=0, type=int,
        help="seed")

    arg_list = arg_parser.parse_known_args()[0]

    path = arg_list.path

    batch_size = arg_list.batch_size
    iter_prior_size = arg_list.iter_prior_size
    iter_post_size = arg_list.iter_post_size
    alpha = arg_list.alpha

    ratio_set_size = arg_list.ratio_set_size
    epsilon = arg_list.epsilon

    model_name = arg_list.model_name
    depth_size = arg_list.depth_size
    width_size = arg_list.width_size

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
    train_size = X_train.shape[0]

    set_size = int(ratio_set_size*np.sqrt(train_size))
    if(set_size == 0):
        set_size = 1

    # ----------------------------------------------------------------------- #
    # Learning the model

    epoch_prior_size = math.ceil(iter_prior_size/(train_size/batch_size))
    epoch_post_size = math.ceil(iter_post_size/(train_size/batch_size))

    learner = BatchLearner(
        model_name=model_name,
        batch_size=batch_size,
        epoch_prior_size=epoch_prior_size,
        epoch_post_size=epoch_post_size,
        alpha=alpha,
        set_size=set_size,
        epsilon=epsilon,
        width_size=width_size,
        depth_size=depth_size,
    )
    learner.fit(X_train, y_train)

    # Learning the posterior model from risk minimization
    erm_learner = ERMLearner(
        bound_model=learner.model,
        model_name=model_name,
        batch_size=batch_size,
        epoch_size=epoch_post_size,
        alpha=alpha,
        width_size=width_size,
        depth_size=depth_size,
    )
    erm_learner.fit(X_train, y_train)

    # Learning the posterior model from risk minimization
    # (regularized with L2 norm)
    reg_L2_learner = ERMLearner(
        bound_model=learner.model,
        model_name=model_name,
        batch_size=batch_size,
        epoch_size=epoch_post_size,
        alpha=alpha,
        reg="L2",
        epsilon=epsilon,
        width_size=width_size,
        depth_size=depth_size,
    )
    reg_L2_learner.fit(X_train, y_train)

    # ----------------------------------------------------------------------- #
    # Evaluating the bound and the test risk

    model = learner.model
    erm_model = erm_learner.model
    reg_L2_model = reg_L2_learner.model

    zero_one_ = Module("BatchZeroOneLoss")

    learner.forward(X=X_train, y=y_train)
    train_risk = zero_one_(model)
    learner.forward(X=X_test, y=y_test)
    test_risk = zero_one_(model)

    erm_learner.forward(X=X_train, y=y_train)
    erm_train_risk = zero_one_(erm_model)
    erm_learner.forward(X=X_test, y=y_test)
    erm_test_risk = zero_one_(erm_model)

    reg_L2_learner.forward(X=X_train, y=y_train)
    reg_L2_train_risk = zero_one_(reg_L2_model)
    reg_L2_learner.forward(X=X_test, y=y_test)
    reg_L2_test_risk = zero_one_(reg_L2_model)

    # ----------------------------------------------------------------------- #
    # Saving

    save_dict = {
        "train_risk": train_risk.item(),
        "test_risk": test_risk.item(),
        "erm_train_risk": erm_train_risk.item(),
        "erm_test_risk": erm_test_risk.item(),
        "reg_L2_train_risk": reg_L2_train_risk.item(),
        "reg_L2_test_risk": reg_L2_test_risk.item(),
    }

    dump = {
        "data": arg_list.data,
        "batch_size": batch_size,
        "iter_prior_size": iter_prior_size,
        "iter_post_size": iter_post_size,
        "alpha": alpha,
        "ratio_set_size": ratio_set_size,
        "epsilon": epsilon,
        "model_name": model_name,
        "depth_size": depth_size,
        "width_size": width_size,
        "seed": seed,
    }

    os.chdir(os.path.dirname(__file__))
    save_data = NDData(os.path.abspath(path))
    save_data.set(save_dict, dump)

###############################################################################


if __name__ == "__main__":
    main()
