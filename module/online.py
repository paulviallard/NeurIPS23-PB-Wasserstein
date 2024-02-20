import os
import torch
import copy
from module.module import Module
from core.writer import Writer
import uuid
import shutil


class OnlineModel(Module):

    def __init__(
        self, model_name=None, **kwargs
    ):

        super().__init__()
        self.model_name = model_name
        self.model = Module(model_name, **kwargs)

    def forward(self, batch):

        x = batch["x"]
        y = batch["y"]
        x = self.model(x)
        return x, y

    def get_parameters(self):
        return self.model.get_parameters()


class Online(Module):

    def __init__(
        self, model_name=None, data_size=None,
        class_size=None, feature_size=None, device=None, **kwargs
    ):

        super().__init__()

        self.model_name = model_name
        self.data_size = data_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.device = device
        self.kwargs = kwargs

        self.model = None
        self.create()
        self.learned = False
        self.i = 1
        self.model_size = 1
        self.path_writer = os.path.join(
            os.path.dirname(__file__),
            "../writer/writer-{}".format(str(uuid.uuid4())))
        self.writer = Writer(self.path_writer)
        self.saved = False

    def create(self):
        if(self.model is None):
            self.model = Module(
                "OnlineModel", model_name=self.model_name,
                data_size=self.data_size, class_size=self.class_size,
                feature_size=self.feature_size, device=self.device,
                **self.kwargs)
        else:
            self.prior_model = copy.deepcopy(self.model)
            self.writer.open(iter=self.model_size-1)
            self.writer.write("state_dict", self.prior_model.state_dict())
            self.writer.save()
            self.model_size += 1

    def forward(self, batch):

        if(self.learned):
            if(not(self.saved)):
                self.save = True
                self.writer.open(iter=self.model_size-1)
                self.writer.write("state_dict", self.model.state_dict())
                self.writer.save()
                self.model_size += 1

            self.writer.open(iter=self.i)
            self.writer.load()
            self.model.load_state_dict(
                self.writer.file_dict["state_dict"][0])

            self.i = (self.i+1) % (self.model_size)
            if(self.i == 0):
                self.i += 1

        self.pred_list, self.y_list = self.model(batch)

    def __del__(self):
        shutil.rmtree(self.path_writer)
