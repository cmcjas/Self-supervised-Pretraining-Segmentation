# Code adapted from segnet example:
# https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb

import torch
from enum import IntEnum
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt

class ToDevice(torch.nn.Module):
    """
    Transfers tensors to the specified device.
    
    Attributes:
        device (torch.device): The device to which tensors will be transferred.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
    
def to_device(x):
    """Transfers tensor to the GPU device, otherwise CPU device."""
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()
    
def get_device():
    """Returns GPU device, otherwise CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class TrimapClasses(IntEnum):
    """Enumeration for classifying pixels in trimaps used in image segmentation."""
    PET = 0
    BACKGROUND = 1
    BORDER = 2

def compute_metrics(pred_labels, true_labels):
    """
    Computes the pixel accuracy and IoU for segmentation predictions.

    Args:
        pred_labels (torch.Tensor): Predicted labels.
        true_labels (torch.Tensor): Ground truth labels.

    Returns:
        tuple: pixel accuracy and IoU accuracy.
    """
    correct_pixels = (pred_labels == true_labels).sum().item()
    total_pixels = pred_labels.numel()
    pixel_accuracy = correct_pixels / total_pixels

    intersection = torch.logical_and(pred_labels == true_labels, true_labels != TrimapClasses.BACKGROUND).sum().item()
    union = torch.logical_or(pred_labels != TrimapClasses.BACKGROUND, true_labels != TrimapClasses.BACKGROUND).sum().item()
    iou_accuracy = intersection / union

    return pixel_accuracy, iou_accuracy

def save_test_images(test_pets_targets, test_pets_labels, pred_labels, image_save_path, pixel_accuracy, iou_accuracy):
    """
    Creates and saves a visualisation of test images and their segmentation masks.

    Args:
        test_pets_targets (torch.Tensor): The test images.
        test_pets_labels (torch.Tensor): The true segmentation masks.
        pred_labels (torch.Tensor): The predicted segmentation masks.
        epoch (int): Current epoch number.
        image_save_path (str): Base directory to save the resulting image.
        pixel_accuracy (float): Pixel accuracy of the predictions.
        iou_accuracy (float): Intersection over Union (IoU) of the predictions.
    """
    to_pil_image = T.ToPILImage()
    pred_mask = pred_labels.to(torch.float)

    target_images = to_pil_image(torchvision.utils.make_grid(test_pets_targets[:16], nrow=8))
    ground_truth_images = to_pil_image(torchvision.utils.make_grid(test_pets_labels[:16].float() / 2.0, nrow=8))
    predicted_images = to_pil_image(torchvision.utils.make_grid(pred_mask[:16] / 2.0, nrow=8))

    fig, axes = plt.subplots(4, 1, figsize=(10, 15))

    # Plot each montage with titles
    axes[0].imshow(target_images)
    axes[0].set_title('Images')
    axes[0].axis('off')  

    axes[1].imshow(ground_truth_images)
    axes[1].set_title('Ground Truth Masks')
    axes[1].axis('off')

    axes[2].imshow(predicted_images)
    axes[2].set_title('Predicted Masks')
    axes[2].axis('off')

    axes[3].set_title(f'Pixel Accuracy: {pixel_accuracy}, IoU: {iou_accuracy}')
    axes[3].axis('off')

    epoch_path = image_save_path
    plt.savefig(image_save_path)