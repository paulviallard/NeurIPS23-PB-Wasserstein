import torch
import logging
import numpy as np

from module.module import Module
from learner.optimize_gd_learner import OptimizeGDLearner
from core.cocob_optim import COCOB
###############################################################################


class BatchLearner(OptimizeGDLearner):

    def __init__(
        self, model_name=None, batch_size=None, epoch_prior_size=None,
        epoch_post_size=None, alpha=None, set_size=None, epsilon=None,
        device="cuda", writer=None, **kwargs
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch_prior_size = epoch_prior_size
        self.epoch_post_size = epoch_post_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.writer = writer
        self.kwargs = kwargs

        self.device = torch.device('cpu')
        if(torch.cuda.is_available() and device != "cpu"):
            self.device = torch.device(device)

        self.set_size = set_size

        self.i = 0

        # ------------------------------------------------------------------- #

    def fit(self, X, y):
        logging.info("Learning ...\n")

        self.m = len(X)

        X_arange = np.arange(self.m)
        X_arange = np.expand_dims(X_arange, 1)
        X = np.concatenate((X_arange, X), axis=1)

        self.data_size = X.shape[0]
        self.feature_size = X.shape[1]-1
        self.class_size = len(np.unique(y))

        self.model = Module(
            "Batch",
            model_name=self.model_name,
            set_size=self.set_size,
            data_size=self.data_size,
            class_size=self.class_size,
            feature_size=self.feature_size,
            epsilon=self.epsilon,
            device=self.device,
            **self.kwargs
        )
        self.model.to(self.device)

        self.__post_parameters, self.__prior_parameters_list = (
            self.model.parameters_list())

        self.post_optim = COCOB(self.__post_parameters, alpha=self.alpha)
        self.prior_optim_list = []
        for parameters in self.__prior_parameters_list:
            self.prior_optim_list.append(COCOB(parameters, alpha=self.alpha))
        self.objective = Module("BatchObjective")
        self._epoch = 1

        self.__is_normalized = False
        super().fit(X, y)

    def forward(self, X=None, y=None):
        super().forward(["pred_list", "y_list"], X=X, y=y)

    def _optimize(self, batch):

        # Learning the priors
        if(self._epoch <= self.epoch_prior_size):

            for optim in self.prior_optim_list:
                optim.zero_grad()

            self.model(batch, step="prior")
            loss_list = self.objective(
                self.model, self.m, step="prior", epsilon=self.epsilon)

            self._loss = 0.0
            loss_size = 0
            for i in range(len(loss_list)):
                loss = loss_list[i]
                if(loss is not None):
                    loss.backward()
                    optim = self.prior_optim_list[i]
                    optim.step()
                    self._loss += loss.item()
                    loss_size += 1
            self._loss = self._loss/loss_size

        # Learning the posteriors
        else:
            if(not(self.__is_normalized)):
                self.__is_normalized = True

            self.post_optim.zero_grad()
            self.model(batch, step="post")
            self._loss = self.objective(
                self.model, self.m, step="post", epsilon=self.epsilon)
            self._loss.backward()
            self.post_optim.step()

    def _meet_condition(self):

        # We test if there is learnable priors
        if(len(self.prior_optim_list) == 1
           and self._epoch <= self.epoch_prior_size):
            self._epoch += self.epoch_prior_size

        if(self._epoch <= self.epoch_prior_size+self.epoch_post_size):
            if(self._epoch <= self.epoch_prior_size):
                logging.info(("Running epoch {} for priors ...\n").format(
                    self._epoch))
            else:
                logging.info(("Running epoch {} for posterior ...\n").format(
                    self._epoch-self.epoch_prior_size))
            return False
        return True

    def _begin_epoch(self):
        pass

    def _end_epoch(self):
        self._epoch += 1

###############################################################################
