import re
import sys
import glob
import os
import importlib
import torch
import numpy as np
import inspect


class MetaModule(type):

    def __get_class_dict(cls):
        # Getting the current path, the file path and the model directory path
        cwd_path = os.getcwd()
        file_path = os.path.dirname(__file__)

        os.chdir(file_path)
        import_module_list = glob.glob("*.py")

        class_dict = {}
        for import_module in import_module_list:

            import_module = "module."+import_module.replace(".py", "")
            for class_name, class_ in inspect.getmembers(
                    importlib.import_module(import_module), inspect.isclass):
                if(class_name != "MetaModule" and class_name != "Module"):
                    class_dict[class_name] = class_

        return class_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, torch.nn.Module, )

        # Getting the name of the module
        if("name" not in kwargs):
            class_name = args[0]
        else:
            class_name = kwargs["name"]

        # Getting the module dictionnary
        class_dict = cls.__get_class_dict()

        # Checking that the module exists
        if(class_name not in class_dict):
            raise Exception(class_name+" doesn't exist")

        # Adding the new module in the base classes
        bases = (class_dict[class_name], )+bases

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaModule, new_cls).__call__(*(args[1:]), **kwargs)

# --------------------------------------------------------------------------- #

class Module(metaclass=MetaModule):

    def __init__(self):
        super().__init__()

    def numpy_to_torch(self, *var_list):
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(var_list[i], np.ndarray)):
                new_var_list.append(
                    torch.tensor(var_list[i], device=self.model.device))
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def torch_to_numpy(self, ref, *var_list):
        # Note: elements in var_list are considered as tensor
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(ref, np.ndarray)
               and isinstance(var_list[i], torch.Tensor)):
                new_var_list.append(var_list[i].detach().cpu().numpy())
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def fit(self, *args, **kwargs):
        raise NotImplementedError
