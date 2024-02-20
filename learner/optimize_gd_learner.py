#!/usr/bin/env python
import torch
import numpy as np
import logging
import random

from sklearn.base import BaseEstimator, ClassifierMixin
from core.numpy_dataset import NumpyDataset


###############################################################################

class OptimizeGDLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self, model, device, batch_size=None
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def fit(self, X, y, X_valid=None, y_valid=None, X_test=None, y_test=None):

        random.seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        self.data = {"x_train": X, "y_train": y}
        if(X_valid is not None and y_valid is not None):
            self.data["x_valid"] = X_valid
            self.data["y_valid"] = y_valid
        if(X_test is not None and y_test is not None):
            self.data["x_test"] = X_test
            self.data["y_test"] = y_test

        self.data = NumpyDataset(self.data)
        self.data.set_mode("train")

        if(self.batch_size is None):
            self.batch_size = len(X)
        self.loader = torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size)

        # Computing batch size
        num_batch = int(len(self.data)/self.batch_size)
        if(len(self.data) % self.batch_size != 0):
            num_batch += 1

        while(not(self._meet_condition())):

            self._begin_epoch()

            loss_sum = 0.0

            for i, batch in enumerate(self.loader):

                self._begin_batch()

                batch["step"] = "train"
                batch["mode"] = batch["mode"][0]
                batch["size"] = batch["size"][0]
                batch["class_size"] = batch["class_size"][0]

                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long)

                self._label = batch["y"]
                self._size = float(batch["size"])

                # Optimize the model
                # REQUIRED:
                #   (1) create self._loss
                #   (2) create the dict self._log
                self._log = {}
                self._optimize(batch)

                loss_sum += self._loss
                loss_mean = loss_sum/(i+1)

                # Printing loss
                logging_str = "[{}/{}] - loss {:.4f}".format(
                        i+1, num_batch, loss_mean)
                for key, value in self._log.items():
                    logging_str += self.__print_logging(key, value)
                logging.info(logging_str+"\r")

                if i+1 == num_batch:
                    logging.info("\n")

                self._end_batch()

            self._end_epoch()

        self.data = None

        return self

    def _meet_condition(self):
        raise NotImplementedError

    def _begin_epoch(self):
        pass

    def _end_epoch(self):
        pass

    def _begin_batch(self):
        pass

    def _end_batch(self):
        pass

    def __print_logging(self, key, value):
        if(isinstance(value, int)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, float)):
            return " - {} {:.4f}".format(key, value)
        elif(isinstance(value, str)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, torch.Tensor)):
            return self.__print_logging(key, value.cpu().detach().numpy())
        elif(isinstance(value, np.ndarray)):
            if(value.ndim == 0):
                return self.__print_logging(key, value.item())
            else:
                raise ValueError("value cannot be an array")
        else:
            raise TypeError(
                "value must be of type torch.Tensor; np.ndarray;"
                + " int; float or str.")

    def forward(self, attr_list, X=None, y=None):
        model = self.model

        data = {}
        if(X is not None):
            data["x_test"] = X
        if(y is not None):
            data["y_test"] = y

        data = NumpyDataset(data)
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        attr_dict = {}

        for i, batch in enumerate(loader):

            if("x" in batch):
                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
            if("y" in batch):
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long)
            batch["step"] = "forward"
            batch["mode"] = batch["mode"][0]
            batch["size"] = batch["size"][0]

            model(batch)

            for attr in attr_list:
                if(attr not in attr_dict):
                    attr_dict[attr] = getattr(model, attr)
                    if(isinstance(getattr(model, attr), list)):
                        for i in range(len(attr_dict[attr])):
                            attr_dict[attr][i] = attr_dict[attr][i].detach()
                    else:
                        attr_dict[attr] = attr_dict[attr].detach()
                else:
                    if(isinstance(getattr(model, attr), list)):
                        for i in range(len(attr_dict[attr])):
                            attr_dict[attr][i] = torch.concatenate(
                                (attr_dict[attr][i],
                                 getattr(model, attr)[i].detach()), axis=0)
                    else:
                        attr_dict[attr] = torch.concatenate(
                            (attr_dict[attr], getattr(model, attr).detach()), axis=0)

        for attr in attr_list:
            setattr(model, attr, attr_dict[attr])

    def save(self):
        return self.model.state_dict()

    def load(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def _optimize(self):
        raise NotImplementedError

###############################################################################
