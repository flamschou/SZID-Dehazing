import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import *
from torch.nn import functional


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        # Get the device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()

        # Create tensor properly based on available device
        self.blur = nn.Parameter(
            data=torch.tensor(blur, dtype=torch.float).to(self.device),
            requires_grad=False,
        )

        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])

        # Create tensor properly based on available device
        self.image = nn.Parameter(
            data=torch.tensor(image, dtype=torch.float).to(self.device),
            requires_grad=False,
        )

        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(
            functional.conv2d(x, self.image), functional.conv2d(x, self.blur)
        )
