import torch
import numpy as np
from module.module import Module


class BatchModel(Module):

    def __init__(
        self, model_name=None, data_split=None, class_size=None,
        feature_size=None, device=None, **kwargs
    ):

        super().__init__()
        self.model_name = model_name
        self.data_split = data_split
        self.class_size = class_size
        self.feature_size = feature_size
        self.device = device

        model_kwargs = {
            "model_name": self.model_name,
            "data_split": self.data_split,
            "class_size": self.class_size,
            "feature_size": self.feature_size,
            "device": self.device,
        }
        model_kwargs.update(kwargs)

        self._model = Module(model_name, **model_kwargs)

    def normalize(self):
        self._model.normalize()

    def forward(self, batch):

        x = batch["x"]
        x_mask = None

        if(x.shape[1] != self.feature_size):
            x_mask = torch.logical_not(
                torch.isin(x[:, 0].int(),
                           torch.tensor(self.data_split, device=x.device)))
            x = x[x_mask]
            x = x[:, 1:]

        y = None
        if("y" in batch):
            y = batch["y"]
            if(x_mask is not None):
                y = y[x_mask]

        if(len(x) == 0):
            return torch.zeros([0, self.class_size]), y

        x = self._model(x)

        return x, y

    def get_parameters(self):
        return self._model.get_parameters()


class Batch(Module):

    def __init__(
        self, model_name=None, set_size=None, data_size=None,
        class_size=None, feature_size=None, device=None, **kwargs
    ):

        super().__init__()

        self.model_name = model_name
        self.set_size = set_size
        self.data_size = data_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.device = device

        step = int(self.data_size/self.set_size)
        data_split = np.split(
            np.arange(self.data_size),
            np.arange(0, self.data_size, step)[1:], axis=0)

        self.post_model = Module(
            "BatchModel",
            model_name=self.model_name,
            data_split=[],
            class_size=self.class_size,
            feature_size=self.feature_size,
            device=self.device,
            **kwargs)

        self.prior_model_list = []
        for i in range(self.set_size):
            self.prior_model_list.append(
                Module(
                    "BatchModel",
                    model_name=self.model_name,
                    data_split=data_split[i],
                    class_size=self.class_size,
                    feature_size=self.feature_size,
                    device=self.device,
                    **kwargs))

        self.prior_model_list = torch.nn.ModuleList(self.prior_model_list)

    def prior_normalize(self):
        for model in self.prior_model_list:
            model.normalize()

    def post_normalize(self):
        self.post_model.normalize()

    def post_initialize(self):
        post_param, prior_param_list = self.parameters_list()
        for i in range(len(post_param)):

            new_param = prior_param_list[0][i]
            for j in range(1, len(prior_param_list)):
                new_param = new_param + prior_param_list[j][i]
            new_param = new_param / len(prior_param_list)
            post_param[i] = new_param

    def parameters_list(self):
        return list(self.post_model.parameters()), [
            list(model.parameters()) for model in self.prior_model_list]

    def forward(self, batch, step="all"):
        self.pred_list = []
        self.y_list = []
        if(step == "prior" or step == "all"):
            for i in range(self.set_size):
                pred, y = self.prior_model_list[i](batch)
                self.pred_list.append(pred)
                self.y_list.append(y)
        if(step == "post" or step == "all"):
            pred, y = self.post_model(batch)
            self.pred_list.append(pred)
            self.y_list.append(y)
