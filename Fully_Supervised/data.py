# Code adapted from segnet example:
# https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb

import torchvision
from torch.utils.data import DataLoader
import torch
import os
import torchvision.transforms as T
from utils import ToDevice, get_device

class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    """
    Custom dataset handler for Oxford-IIIT Pet Dataset to include preprocessing.

    Parameters:
        root (str): Directory where the dataset is located.
        split (str): Dataset split, typically 'trainval' or 'test'.
        target_types (str): Types of targets to use, e.g., 'segmentation'.
        download (bool): If `True`, downloads the dataset from the internet if not available at `root`.
        pre_transform (callable, optional): Transform to apply to the inputs before any other processing.
        post_transform (callable, optional): Transform to apply to the inputs after all other processing.
        pre_target_transform (callable, optional): Transform to apply to the targets before any other processing.
        post_target_transform (callable, optional): Transform to apply to the targets after all other processing.
        common_transform (callable, optional): Transform to apply to both the inputs and targets.
    """
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)
    
# Create a tensor for a segmentation trimap.
# Input: Float tensor with values in [0.0 .. 1.0]
# Output: Long tensor with values in {0, 1, 2}
def tensor_trimap(t):
    """
    Converts a tensor into a trimap tensor for segmentation tasks.

    Parameters:
        t (torch.Tensor): The tensor to convert.

    Returns:
        torch.Tensor: The trimap tensor.
    """
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

transform_dict = args_to_dict(
    pre_transform=T.ToTensor(),
    pre_target_transform=T.ToTensor(),
    common_transform=T.Compose([
        ToDevice(get_device()),
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
        T.RandomHorizontalFlip(p=0.5),
    ]),
    post_transform=T.Compose([
        T.ColorJitter(contrast=0.3),
    ]),
    post_target_transform=T.Compose([
        T.Lambda(tensor_trimap),
    ]),
)

def load_datasets(working_dir, batch_size):
    """
    Loads datasets for training and testing.

    Parameters:
        working_dir (str): The directory where datasets are stored or will be downloaded.
        batch_size (int): The size of the batch to use when loading data.

    Returns:
        tuple: A tuple containing loaders for train and test datasets.
    """
    pets_path_train = os.path.join(working_dir, 'OxfordPets', 'train')
    pets_path_test = os.path.join(working_dir, 'OxfordPets', 'test')
    
    pets_train = OxfordIIITPetsAugmented(
        root=pets_path_train,
        split="trainval",
        target_types="segmentation",
        download=True,
        **transform_dict,
    )
    pets_test = OxfordIIITPetsAugmented(
        root=pets_path_test,
        split="test",
        target_types="segmentation",
        download=True,
        **transform_dict,
    )

    pets_train_loader = DataLoader(
        pets_train,
        batch_size=batch_size,
        shuffle=True,
    )
    pets_test_loader = DataLoader(
        pets_test,
        batch_size=64,
        shuffle=False,
    )
    return pets_train_loader, pets_test_loader