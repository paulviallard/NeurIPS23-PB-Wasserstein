import math
import torch
from module.module import Module


class OGDObjective(Module):

    def __init__(self):
        super().__init__()
        self._margin_loss = Module("MarginLoss")

    def forward(self, model):
        risk = self._margin_loss(
            model.pred_list, model.y_list, model.class_size)
        return risk


class OnlineObjective(Module):

    def __init__(self):
        super().__init__()
        self._margin_loss = Module("MarginLoss")
        self._wasserstein = Module("Wasserstein")

    def __log_barrier(self, x, t=100):
        # Log barrier extension from
        # Constrained Deep Networks: Lagrangian Optimization
        # via Log-Barrier Extensions
        # Hoel Kervadec, Jose Dolz, Jing Yuan, Christian Desrosiers,
        # Eric Granger, Ismail Ben Ayed, 2019
        assert isinstance(x, torch.Tensor) and len(x.shape) == 0
        if(x <= -1.0/(t**2.0)):
            return -(1.0/t)*torch.log(-x)
        else:
            return t*x - (1.0/t)*math.log(1/(t**2.0))+(1/t)

    def forward(self, model, m):

        risk = self._margin_loss(
            model.pred_list, model.y_list, model.class_size)

        dist = self._wasserstein(
            model.model, model.prior_model,
            grad_1=True, grad_2=False)
        return risk + dist, risk + dist + self.__log_barrier(dist-1.0)


class OnlineZeroOneLoss(Module):

    def __init__(self):
        super().__init__()
        self._zero_one_loss = Module("ZeroOneLoss")

    def forward(self, model):
        pred = model.pred_list
        y = model.y_list
        return self._zero_one_loss(pred, y)
