import torch
from module.module import Module


class Model(Module):

    def __init__(self):
        super().__init__()
        self._param_to_exclude = []

    def get_parameters(self):

        param_to_exclude = [id(param) for param in self._param_to_exclude]

        param_list = None
        for param in self.parameters():

            if(id(param) not in param_to_exclude):
                param_ = torch.flatten(param)
                if(param_list is None):
                    param_list = param_
                else:
                    param_list = torch.concat((param_list, param_))
        return param_list

    def lipschitz(self):
        raise NotImplementedError
