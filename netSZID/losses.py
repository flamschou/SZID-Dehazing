import torch
from torch import nn
import numpy as np
from torch.nn import functional


class AngularLossLayer(nn.Module):
    def __init__(self):
        super(AngularLossLayer, self).__init__()

    def forward(self, y_pred, y_true):
        # Normalize input vectors
        y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
        y_true = torch.nn.functional.normalize(y_true, p=2, dim=1)

        # Compute angle between vectors
        dot_product = torch.sum(y_pred * y_true, dim=1)
        angle = torch.acos(dot_product.clamp(-1.0 + torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps))

        # Compute loss
        angular_loss = 1 - torch.cos(angle)

        loss = torch.mean(angular_loss)

        return loss

