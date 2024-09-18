import torch
import subprocess
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


def download_images(data_root, class_list, images_per_class, csv_path):
    """
    Download images into class folders using the downloader script, after checking against a CSV file for class names,
    using NumPy for handling the CSV file.

    Args:
        data_root (str): The root directory where images will be downloaded.
        class_list (list of str): List of class identifiers to download.
        images_per_class (int): Number of images to download per class.
        csv_path (str): Path to the CSV file containing class identifiers and names.
    """
    # Load CSV file using NumPy
    data = np.genfromtxt(csv_path, delimiter=',', dtype=None, encoding=None, skip_header=1)
    class_id_to_name = {row[0]: row[1] for row in data}

    for synid in class_list:
        class_name = class_id_to_name.get(synid)
        if not class_name:
            print(f"Class ID {synid} not found in CSV. Skipping.")
            continue
        
        # Check if folder exists
        class_folder_path = os.path.join('./Data/imagenet_images', class_name)
        if os.path.exists(class_folder_path):
            print(f"Folder for class {class_name} already exists. Skipping download.")
            continue
        
        # If folder doesn't exist, proceed with download
        command = [
            "python", "./ImageNet_Downloader/downloader.py",
            "-data_root", data_root,
            "-use_class_list", "True",
            "-class_list", synid,
            "-images_per_class", str(images_per_class)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print("Download script output:", result.stdout)
        if result.stderr:
            print("Download script errors:", result.stderr)

# Example usage of the download function
data_root_folder = "./Data"
csv_file_path = "./ImageNet_Downloader/classes_in_imagenet.csv"
class_list = ['n00015388', 'n01315805', 'n01316949', 'n01317294', 'n01317391', 'n01317541', 'n01319467', 'n02451575', 'n10159045', 
            'n01318894', 'n03920641', 'n10420507', 'n03679712', 'n03841666', 'n06209940', 'n10806113', 'n07805731', 'n03319745', 'n08494231', 
            'n00021265', 'n04019101', 'n04524313', 'n02021795', 'n09428628', 'n00453935', 'n02512053', 'n02691156', 'n04461879', 'n09359803', 
             'n08569482', 'n04358707', 'n03619890', 'n04081281', 'n09411189', 'n10078806' ]
images_per_class = 5000
for i in class_list:
    download_images(data_root_folder, [i], images_per_class, csv_file_path)

class UnlabelledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Traverse the root_dir and add images from subdirectories
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.images.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = io.read_image(img_path)  # This reads the image as a tensor
        image = image.float() / 255.0  # Normalize to [0, 1]

        if self.transform:
            image = self.transform(image)

        return image

# Define the transformations with extended data augmentation
transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.RandomRotation(10),  # Randomly rotate the images by up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly alter the brightness, contrast, saturation, and hue
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # Randomly crop and resize the image
    transforms.ToTensor(),  # This is now applied within the dataset class
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize each channel
])

# Loading and transforming the dataset with augmented transformations
# Replace 'your_unlabeled_dataset_path' with the path to your dataset
unlabeled_dataset = UnlabelledImageDataset(root_dir="./Data/imagenet_images/", transform=transform_augmented)

# Loading the dataset into a DataLoader for batch processing
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)


