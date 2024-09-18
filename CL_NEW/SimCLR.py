
import torch
import time
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from extras import *
from extras_modified import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


class BaseEncoder(nn.Module):
    """use resnet18 as base encoder to extract features from input images.
       Later reuse in downstream tasks
    """
    def __init__(self, pretrained=False):
        super(BaseEncoder, self).__init__()
        self.resnet = resnet18(pretrained=pretrained) 
        self.resnet.fc = torch.nn.Identity()  

    def forward(self, x):
        x = self.resnet(x)
        return x


class ProjectionHead(nn.Module):
    """
    This ProjectionHead Class is used in contrastive learning, transform the feature vectors extracted from resnet to space.
    """
    def __init__(self, feature_dim=512, projection_dim=128):  
        super(ProjectionHead, self).__init__()
        # self.reduction = nn.Linear(feature_dim, projection_dim, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
    def forward(self, x):
        return self.projection(x)
    
    
class ResNetWithProjectionHead(nn.Module):
    ""
    def __init__(self, base_encoder, projection_head):
        super(ResNetWithProjectionHead, self).__init__()
        self.base_encoder = base_encoder
        self.projection_head = projection_head

    def forward(self, x):
        features = self.base_encoder(x)
        projections = self.projection_head(features)
        return projections
    

def pretrain_model(batch_size, learning_rate, weight_decay):
        
        print('Starting the script...\n')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'Data')  # Update this path to your 'data' directory
        best_validation_loss = float('inf')
        old_validation_loss = float('inf')
        epochs_no_improve = 0
        n_epochs_stop = 5  # Number of epochs to stop after no improvement

        # resize to 224x224 and convert to tensor
        target_size = (224, 224)
        initial_transform = transforms.Compose([
            transforms.Resize(target_size), 
            transforms.ToTensor(),
        ])

        print('Initialising datasets...\n')

        # Use ImageFolder to load the dataset
        initial_dataset = ImageFolder(root=data_path, transform=initial_transform)
        initial_dataloader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        def compute_mean_std(dataloader):
            channels_sum, channels_squared_sum, num_batches = 0, 0, 0

            for data, _ in dataloader:
                channels_sum += torch.mean(data, dim=[0, 2, 3])
                channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
                num_batches += 1

            mean = channels_sum / num_batches

            # Variance = E[X^2] - (E[X])^2
            std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

            return mean, std
        
        mean, std = compute_mean_std(initial_dataloader)


        # Define transformations with normalization using computed mean and std
        transform_with_normalization = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1)
                ], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        full_dataset = ImageFolder(root=data_path, transform=transform_with_normalization)

        train_size = int(0.8 * len(full_dataset))
        validation_size = len(full_dataset) - train_size
        train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # batch_size as large as possible 
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print('Initialising model...\n')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        base_encoder = BaseEncoder(pretrained=False)
        projection_head = ProjectionHead(feature_dim=512, projection_dim=128).to(device)
        model = ResNetWithProjectionHead(base_encoder, projection_head)
        model = model.to(device)
        

        print('Pre training set up...\n')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        contrastive_loss = NTXentLoss()
        # Initialize the scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)


        print('Starting training...\n')
        with open('resnet18_pretrain_modify_weights.txt', 'a') as f:
            print(f'Batch size: {batch_size}, Learning rate: {learning_rate}, Weight decay: {weight_decay}', file=f, flush=True)
            print(f'Dataset - Mean: {mean}, Standard Deviation: {std} \n', file=f, flush=True)

            for epoch in range(20):
                print(f'Starting epoch {epoch+1}, train size: {train_size}, validation size: {validation_size}', file=f, flush=True)
                
                model.train()
                train_loss = 0
                train_epoch_start = time.time()
                for i, (images, _) in enumerate(train_loader):  # Ignore labels
                    optimizer.zero_grad()

                    # forward pass
                    # images = torch.cat(images, dim=0).to(device)
                    images = images.to(device)
                    features = model(images)

                    loss = contrastive_loss(features)
                    train_loss += loss.item()

                    #backwards
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 99:
                        print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: { train_loss/ (i+1):.3f}', file=f, flush=True)
                scheduler.step()
                train_epoch_end = time.time()

                # visualize and validation loop
                model.eval()  # Evaluation mode
                embeddings = []
                labels_list = []
                validation_loss = 0

                if not os.path.exists('./visualisations_modify'):
                    os.makedirs('./visualisations_modify')

                with torch.no_grad():
                    for images, labels in validation_loader:  # Now we also take labels
                        images = images.to(device)
                        labels = labels.to(device)
                        output = model(images)
                        embeddings.append(output.cpu().numpy())
                        labels_list.append(labels.cpu().numpy())  # Store labels

                        loss = contrastive_loss(output)
                        validation_loss += loss.item()

                embeddings = np.concatenate(embeddings, axis=0)
                labels_list = np.concatenate(labels_list, axis=0)  

                # Dimensionality Reduction with t-SNE
                tsne = TSNE(n_components=2, random_state=123)
                reduced_embeddings = tsne.fit_transform(embeddings)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels_list, alpha=0.5)  # Color by label
                plt.legend(*scatter.legend_elements(), title="Classes")
                plt.title('t-SNE of Image Embeddings')
                plt.xlabel('t-SNE dimension 1')
                plt.ylabel('t-SNE dimension 2')
                plt.savefig(f'./visualisations_modify/tsne_embeddings_multi_{epoch+1}_{batch_size}_{learning_rate}.png', dpi=300)

                avg_validation_loss = validation_loss / len(validation_loader)
                print(f"Epoch {epoch+1} complete\n Average Train Loss: {train_loss / len(train_loader)}\n Time taken: {train_epoch_end-train_epoch_start} Average Validation Loss: {avg_validation_loss} \n", file=f, flush=True)

                # Check for early stopping
                if avg_validation_loss < best_validation_loss:
                    old_validation_loss = best_validation_loss
                    best_validation_loss = avg_validation_loss
                    torch.save(model.state_dict(), f"resnet18_pretrain_modify_weights_{batch_size}_{learning_rate}.pt")
                    epochs_no_improve = 0
                    print(f"Validation loss decreased ({old_validation_loss:.6f} --> {best_validation_loss:.6f}).  Saving model ...\n", file=f, flush=True)
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss did not decrease, count: {epochs_no_improve} \n", file=f, flush=True)
                    if epochs_no_improve == n_epochs_stop:
                        print(f'\nEarly stopping!', file=f, flush=True)
                        break

            print(f'Best validation loss: {best_validation_loss:.6f} \n', file=f, flush=True)
            print('Training done')


if __name__ == "__main__":
    for batch_size in [64, 96, 128]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            for weight_decay in [learning_rate/10]:
                pretrain_model(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
