# Code adapted from segnet example:
# https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb

from torch import nn
from torchvision import models

class ResNetSegmentation(nn.Module):
    """
    A segmentation model based on ResNet architectures. This class supports either
    ResNet-18 or ResNet-50 as the backbone for feature extraction followed by a series
    of convolutional and transposed convolutional layers to produce a segmentation mask.

    Parameters:
    - n_classes (int): The number of classes for the segmentation task.
    - isResNet18 (bool): Flag to determine whether to use ResNet-18 (True) or ResNet-50 (False) as the backbone.

    The decoder part of the model upscales the output of the backbone to the input resolution using transposed convolutions.
    If ResNet-18 is used, the decoder starts processing from a 512-dimensional feature space.
    If ResNet-50 is used, it starts from a 2048-dimensional feature space, requiring an additional convolutional layer to reduce dimensionality before upsampling.
    """
    def __init__(self, n_classes, isResNet18):
        super(ResNetSegmentation, self).__init__()
        backbone = models.resnet18(weights=None) if isResNet18 else models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove the last fully connected layer and pooling layer
        
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
        ) if isResNet18 else nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, stride=1), # Extra layer for ResNet50
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size,
          C is the number of channels, and H and W are the height and width of the input images.

        Returns:
        - torch.Tensor: Output segmentation map of shape (N, n_classes, H_out, W_out),
          where H_out and W_out are the dimensions of the output segmentation map, resized to 128x128.
        """
        x = self.backbone(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return x