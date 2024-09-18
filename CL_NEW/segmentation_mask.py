import torch
from torch import optim
import torchvision
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import time 
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels // 2, num_classes, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(512, num_classes, kernel_size=1)
        # self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True) 
  
    def forward(self, x):
        output = self.upsample(x)
        output = self.conv(output)
        return output
    

class ResnetwithSegHead(torch.nn.Module):
    def __init__(self, backbone, segmentation_head):
        super(ResnetwithSegHead, self).__init__()
        self.backbone = backbone
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.segmentation_head = segmentation_head

    def forward(self, x):
        features = self.features(x)
        # features = self.backbone(x)
        output = self.segmentation_head(features)
        # Upsample the output to match the target masks' size
        output = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
        return output

class Transform:
    def __call__(self, image, mask):
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224),  interpolation=TF.InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, mask


# convert normalized images to original
def denormalize(tensor): 
    tensor = tensor.clone().detach()  
    for t, m, s in zip(tensor, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])):
        t.mul_(s).add_(m)  # Multiply by std dev and then add the mean
    return tensor


if __name__ == '__main__':
    
    #data preparation
    print('Preparing dataset...')

    batch_size = 32

    transform = Transform()

    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # pets_path = os.path.join(base_dir, 'OxfordPets')

    pets_dataset = torchvision.datasets.OxfordIIITPet(root='./OxfordPets', download=True, transforms=transform, target_types='segmentation')

    develop_size = int(len(pets_dataset) * 0.8) # Test at 20%
    develop_set, test_set = random_split(pets_dataset, [develop_size, int(len(pets_dataset) - develop_size)])
    train_size = int(len(pets_dataset) * 0.8 * 0.9) # Train at 72%, Val at 8%
    train_set, val_set = random_split(develop_set, [train_size, int(len(develop_set) - train_size)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


    #traininng 
    # Check if CUDA is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    num_classes = 3

    backbone = resnet18(weights=None)
    backbone.fc = torch.nn.Identity()

    filtered_state_dict = torch.load('resnet_weights.pt')

    # Load the filtered state dict into the model, missing keys are ignored
    backbone.load_state_dict(filtered_state_dict, strict=False)

    # segmentation_head = SegmentationHead(in_channels=512, num_classes=num_classes)
    segmentation_head = SegmentationHead(in_channels=512, num_classes=num_classes)
    model = ResnetwithSegHead(backbone, segmentation_head)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0) 

    # Freezing layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    print('Model training...')
    
    with open('resnet50_segmentation_default_weights.txt', 'a') as f:
        for epoch in range(2):
            print(f'Starting epoch {epoch+1}\n', file=f, flush=True)
            
            model.train()
            train_loss = 0
            train_epoch_start = time.time()
            for i, (images, targets) in enumerate(train_loader):
                
                # Move images to the device
                images = images.to(device)
                
                print(targets)
                print(targets.size())
                print(torch.unique(targets))
                # if targets.size(1) == 1:
                #     targets = targets.squeeze(1).long().to(device)

                print(targets)
                targets_converted = (targets*255 -1).long()

                targets_converted = targets_converted.squeeze(1).long().to(device)

                # print(targets_converted)
                # print(torch.unique(targets_converted))

                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, targets_converted)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                if i % 10 == 9:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: { train_loss/ (i+1):.3f}', file=f, flush=True)

            scheduler.step()

            train_epoch_end = time.time()

            model.eval()
            validation_loss = 0
            with torch.no_grad():  
                for batch_idx, (images, targets) in enumerate(val_loader):
                    images = images.to(device)
                    if targets.size(1) == 1:
                        targets = targets.squeeze(1).long().to(device)
            
                    features = model(images)

                    loss = criterion(features, targets)
                    validation_loss += loss.item()

            print(f"Epoch {epoch+1} complete\n Average Train Loss: {train_loss / len(train_loader)}\n Time taken: {train_epoch_end - train_epoch_start}\n Average Validation Loss: {validation_loss / len(val_loader)}", file=f, flush=True)
            
            # test loop starts
            model.eval()
            test_loss = 0
            with torch.no_grad():  
                for batch_idx, (images, targets) in enumerate(test_loader):
                    images = images.to(device)
                
                    if targets.size(1) == 1:
                        targets = targets.squeeze(1).long().to(device)

                    features = model(images)

                    loss = criterion(features, targets)
                    test_loss += loss.item()
            print(f"Average Test Loss: {test_loss / len(test_loader)} \n", file=f, flush=True)


    print('Training done')
    torch.save(model.state_dict(), 'saved_CL_segmentation_model.pt')
    print('Model saved.')





    

    
