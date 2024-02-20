import torch
import logging
import numpy as np

from module.module import Module
from learner.optimize_gd_learner import OptimizeGDLearner
from core.cocob_optim import COCOB
###############################################################################


class OGDLearner(OptimizeGDLearner):

    def __init__(
        self, model_name=None, device="cpu", writer=None,
        alpha=None, iter_size=None, **kwargs
    ):
        self.model_name = model_name
        self.batch_size = 1
        self.epoch_size = 1
        self.writer = writer
        self.kwargs = kwargs
        self.alpha = alpha
        self.iter_size = iter_size

        self.device = torch.device('cpu')
        if(torch.cuda.is_available() and device != "cpu"):
            self.device = torch.device(device)

        self._iter = 1
        self.optim = None

        # ------------------------------------------------------------------- #

    def fit(self, X, y):
        logging.info("Learning with OGD...\n")

        self.m = len(X)
        self.feature_size = X.shape[1]
        self.class_size = len(np.unique(y))

        self.model = Module(
            "Online", model_name=self.model_name, data_size=self.m,
            class_size=self.class_size, feature_size=self.feature_size,
            device=self.device, **self.kwargs)
        self.model.to(self.device)

        self.online_objective = Module("OGDObjective")

        super().fit(X, y)
        self.model.learned = True

    def forward(self, X=None, y=None):
        super().forward(["pred_list", "y_list"], X=X, y=y)

    def _optimize(self, batch):
        if(not(self.model.learned)):
            self.__optimize_stochastic(batch)

    def _meet_condition(self):
        if(self._iter <= self.m):
            return False
        return True

    def _begin_batch(self):
        pass

    def _end_batch(self):
        self._iter += 1

    # ----------------------------------------------------------------------- #

    def __optimize_stochastic(self, batch):

        self.model.create()
        if(self.optim is None):
            self.optim = COCOB(list(self.model.parameters()), alpha=self.alpha)

        for i in range(self.iter_size):
            self.optim.zero_grad()
            self.model(batch)
            self._loss = self.online_objective(self.model)
            self._loss.backward()
            self.optim.step()
