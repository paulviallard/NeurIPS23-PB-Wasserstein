import numpy as np
from module.module import Module


class ERMObjective(Module):

    def __init__(self):
        super().__init__()
        self._margin_loss = Module("MarginLoss")
        self._wasserstein = Module("Wasserstein")

    def forward(self, model, reg=None, m=None, epsilon=None):
        risk = self._margin_loss(
            model.pred_list[0], model.y_list[0], model.class_size)

        if(reg is None or m is None or epsilon is None):
            return risk

        if(epsilon == "sqrtm"):
            epsilon = 1.0/np.sqrt(m)
        elif(epsilon == "m"):
            epsilon = 1.0/m
        else:
            raise RuntimeError("epsilon must be either sqrtm or m")

        if(reg == "L2"):
            ord = 2
        else:
            raise RuntimeError("reg must be L2")

        dist = self._wasserstein(model.post_model, ord=ord)
        return risk + epsilon*dist


class BatchObjective(Module):

    def __init__(self):
        super().__init__()
        self._margin_loss = Module("MarginLoss")
        self._wasserstein = Module("Wasserstein")

    def forward(self, model, m, epsilon="sqrtm", step=None):

        assert step == "prior" or step == "post"

        if(step == "prior"):
            risk_list = []
            for i in range(len(model.pred_list)):
                risk = self._margin_loss(
                    model.pred_list[i], model.y_list[i].long(),
                    model.class_size)

                if(len(model.pred_list[i]) == 0):
                    risk_list.append(None)
                else:
                    risk_list.append(risk)

            return risk_list

        else:

            model_1 = model.post_model

            risk = self._margin_loss(
                model.pred_list[0], model.y_list[0].long(), model.class_size)

            if(epsilon == "sqrtm"):
                epsilon = 1.0/np.sqrt(m)
            elif(epsilon == "m"):
                epsilon = 1.0/m
            else:
                raise RuntimeError("epsilon must be either sqrtm or m")

            dist_sum = 0.0
            for i in range(len(model.prior_model_list)):
                model_2 = model.prior_model_list[i]
                dist = self._wasserstein(model_1, model_2, grad_2=False)
                data_split_size = len(model_2.data_split)
                dist_sum = dist_sum + (data_split_size/m)*dist
            return risk + epsilon*dist_sum


class BatchZeroOneLoss(Module):

    def __init__(self):
        super().__init__()
        self._zero_one_loss = Module("ZeroOneLoss")

    def forward(self, model):
        pred = model.pred_list[-1]
        y = model.y_list[-1]
        return self._zero_one_loss(pred, y)
