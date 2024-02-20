import torch
from module.module import Module
from module.model import Model


class FullyConnectedModel(Model, Module):

    def __init_gaussian(self, shape, sigma=0.04, random_state=0):
        gen = torch.Generator()
        gen.manual_seed(random_state)
        init = torch.randn(shape, generator=gen)*sigma
        init = torch.clip(init, -2.0*sigma, 2.0*sigma)
        return init

    def __init__(
        self, width_size=None, depth_size=None, class_size=None,
        feature_size=None, device=None, **kwargs
    ):

        super().__init__()

        self.width_size = width_size
        self.depth_size = depth_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.device = device

        self.linear_list = []
        self.bias_list = []

        self.linear_list.append(
            torch.nn.Parameter(
                self.__init_gaussian((self.width_size, self.feature_size)))
        )
        self.bias_list.append(
            torch.nn.Parameter(0.1*torch.ones(
                self.width_size)))

        for i in range(self.depth_size-1):

            self.linear_list.append(
                torch.nn.Parameter(
                    self.__init_gaussian((self.width_size, self.width_size)))
            )

            self.bias_list.append(
                torch.nn.Parameter(torch.zeros(
                    self.width_size)))

        self.linear_list.append(torch.nn.Parameter(
            self.__init_gaussian((self.class_size, self.width_size))))
        self.bias_list.append(torch.nn.Parameter(
            torch.zeros(self.class_size)))

        self.linear_list = torch.nn.ParameterList(self.linear_list)
        self.bias_list = torch.nn.ParameterList(self.bias_list)

        self._scaling = torch.nn.Parameter(torch.tensor(0.0))
        self._param_to_exclude = [self._scaling]

    def scaling(self):
        return torch.exp(self._scaling)

    def normalize(self):
        norm = torch.norm(self.get_parameters()).item()
        for i in range(len(self.linear_list)):
            self.linear_list[i].data = self.linear_list[i].data/norm
            self.bias_list[i].data = self.bias_list[i].data/norm

    def forward(self, x):
        for i in range(len(self.linear_list)):
            x = torch.nn.functional.linear(
                x, self.linear_list[i], bias=self.bias_list[i])
            if(i != len(self.linear_list)-1):
                x = torch.nn.functional.leaky_relu(x)
        x = self.scaling()*x
        return x
