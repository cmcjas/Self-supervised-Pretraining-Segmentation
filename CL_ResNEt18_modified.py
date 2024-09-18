
import torch
import time
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torch.optim as optim
from torch.utils.data import random_split
from extras_modified import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def pre_train():

    #### 

    print('Starting the stript...\n')

    ########

    # Check if CUDA is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    


    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(size=96),
                                transforms.RandomApply([
                                    transforms.ColorJitter(brightness=0.5,
                                                            contrast=0.5,
                                                            saturation=0.5,
                                                            hue=0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.GaussianBlur(kernel_size=9),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])

    #######

    print('Initialising datasets...\n')
        # Get dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'STL10')
    os.makedirs(data_path, exist_ok=True)
    full_dataset = STL10(root=data_path, split='unlabeled', download=True,
            transform=ContrastiveTransformations(contrast_transforms))

    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size

    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4)

    validation_loader = DataLoader(validation_dataset, batch_size=96, shuffle=False, num_workers=4)

    #######

    class ResNetWithProjectionHead(torch.nn.Module):
        def __init__(self, base_model):
            super(ResNetWithProjectionHead, self).__init__()
            # Remove the fully connected layer of ResNet to maintain spatial dimensions
            self.base_model = torch.nn.Sequential(*(list(base_model.children())[:-2]))
            # Add any additional layers here if necessary, ensuring they maintain spatial dimensions

        def forward(self, x):
            x = self.base_model(x)
            # No flattening or global pooling should be done here if you want to maintain spatial dimensions
            return x


    print('Initialising model...\n')
    resnet_model = models.resnet18(pretrained=True) # try pretrained true 
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 128)  # Adjust according to the in_features
    model = ResNetWithProjectionHead(resnet_model)
    model.to(device)

    #######

    print('Pre training set up...\n')
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    contrastive_loss = NTXentLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)  # T_max is epochs

    #######

    print('Starting training...\n')
    for epoch in range(10):
        print(f'Starting epoch {epoch+1}\n')
        
        model.train()
        train_loss = 0
        train_epoch_start = time.time()
        for batch_idx, (images, _) in enumerate(train_loader):

            # Move images to the device
            images = torch.cat(images, dim=0).to(device)

            optimizer.zero_grad()
            features = model(images)

            loss = contrastive_loss(features)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        train_epoch_end = time.time()

        model.eval()
        validation_loss = 0
        with torch.no_grad():  
            for batch_idx, (images, _) in enumerate(validation_loader):
                images = torch.cat(images, dim=0).to(device)
                features = model(images)

                loss = contrastive_loss(features)
                validation_loss += loss.item()

        print(f"Epoch {epoch+1} complete\n Average Train Loss: {train_loss / len(train_loader)}\n Time taken: {train_epoch_end}\n Average Validation Loss: {validation_loss / len(validation_loader)}")

    print('Training done')
    torch.save(model.state_dict(), "res18CLpretrain.pt")
    print("Model saved.")

if __name__ == "__main__":
    pre_train()