import torch
from module.module import Module
from module.model import Model


class LinearModel(Model, Module):

    def __init__(
        self, class_size=None, feature_size=None, device=None, **kwargs
    ):

        super().__init__()
        self.device = device
        self.class_size = class_size
        self.feature_size = feature_size

        self.linear = torch.nn.Parameter(torch.zeros(
            self.class_size, self.feature_size))
        self.bias = torch.nn.Parameter(
            torch.zeros(self.class_size))

        self._scaling = torch.nn.Parameter(torch.tensor(0.0))
        self._param_to_exclude = [self._scaling]

    def scaling(self):
        return torch.exp(self._scaling)

    def normalize(self):
        norm = torch.norm(self.get_parameters()).item()
        if(norm > 0.0):
            self.linear.data = self.linear.data/norm
            self.bias.data = self.bias.data/norm

    def forward(self, x):
        x = torch.nn.functional.linear(
            x, self.linear, bias=self.bias)
        x = self.scaling()*x
        return x
