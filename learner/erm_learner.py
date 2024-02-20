import torch
import logging
import numpy as np

from module.module import Module
from learner.optimize_gd_learner import OptimizeGDLearner
from core.cocob_optim import COCOB
###############################################################################


class ERMLearner(OptimizeGDLearner):

    def __init__(
        self, bound_model=None,
        model_name=None, batch_size=None, epoch_size=None, alpha=None,
        reg=None, epsilon=None,
        device="cuda", writer=None, **kwargs
    ):
        self.bound_model = bound_model
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.alpha = alpha
        self.reg = reg
        self.epsilon = epsilon
        self.writer = writer
        self.kwargs = kwargs

        self.device = torch.device('cpu')
        if(torch.cuda.is_available() and device != "cpu"):
            self.device = torch.device(device)

        self.i = 0

        # ------------------------------------------------------------------- #

    def fit(self, X, y):

        if(self.reg == "L2"):
            logging.info("Learning with ERM (with L2 regularization)...\n")
        else:
            logging.info("Learning with ERM...\n")

        self.m = len(X)

        X_arange = np.arange(self.m)
        X_arange = np.expand_dims(X_arange, 1)
        X = np.concatenate((X_arange, X), axis=1)

        self.data_size = X.shape[0]
        self.feature_size = X.shape[1]-1
        self.class_size = len(np.unique(y))

        self.model = Module(
            "Batch",
            model_name=self.bound_model.model_name,
            set_size=self.bound_model.set_size,
            data_size=self.bound_model.data_size,
            class_size=self.bound_model.class_size,
            feature_size=self.bound_model.feature_size,
            device=self.device,
            **self.kwargs
        )
        self.model.to(self.device)

        self.optim = COCOB(self.model.parameters(), alpha=self.alpha)
        self.objective = Module("ERMObjective")
        self._epoch = 1

        super().fit(X, y)
        self.model.prior_model_list = self.bound_model.prior_model_list

    def forward(self, X=None, y=None):
        super().forward(["pred_list", "y_list"], X=X, y=y)

    def _optimize(self, batch):

        self.optim.zero_grad()
        self.model(batch, step="post")
        self._loss = self.objective(
            self.model, reg=self.reg, m=self.m, epsilon=self.epsilon)
        self._loss.backward()
        self.optim.step()

    def _meet_condition(self):

        if(self._epoch <= self.epoch_size):
            logging.info(("Running epoch {} for ERM posterior ...\n").format(
                self._epoch))
            return False
        return True

    def _begin_epoch(self):
        pass

    def _end_epoch(self):
        self._epoch += 1

###############################################################################
