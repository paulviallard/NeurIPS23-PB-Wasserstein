import torch
from module.module import Module


class MarginLoss(Module):

    def lipschitz(self):
        return 1.0

    def forward(self, pred, y, class_size, clip=False):

        if(pred.shape[0] == 0):
            return torch.tensor(0.0, device=pred.device)

        prior_size = int(pred.shape[1]/class_size)
        pred = pred.reshape(pred.shape[0], prior_size, class_size)
        pred = pred.reshape(-1, class_size)

        y = y.reshape(y.shape[0], 1, 1)
        y = y.repeat((1, prior_size, 1))
        y = y.reshape(-1, 1)

        loss = torch.nn.functional.multi_margin_loss(
            pred, y[:, 0], reduction="none")
        if(clip):
            loss = torch.clamp(loss, min=0.0, max=1.0)
        loss = torch.mean(loss)

        return loss


class Wasserstein(Module):

    def forward(
        self, model_1, model_2=None, grad_1=True, grad_2=True, ord=None
    ):

        weight_1 = model_1.get_parameters()
        if(model_2 is not None):
            weight_2 = model_2.get_parameters()

        if(not(grad_1)):
            weight_1 = weight_1.detach()
        if(model_2 is not None):
            if(not(grad_2)):
                weight_2 = weight_2.detach()

        if(model_2 is not None):
            wasserstein = torch.linalg.norm(weight_1-weight_2, ord=ord)
        else:
            wasserstein = torch.linalg.norm(weight_1, ord=ord)

        return wasserstein


class ZeroOneLoss(Module):

    def forward(self, pred, y):

        pred = torch.argmax(pred, dim=1)
        risk = (pred != y).float()
        risk = torch.mean(risk)

        return risk
