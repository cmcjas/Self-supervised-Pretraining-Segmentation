import torch
from torch import optim
import torchvision
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import time 

def CL_seg_train():

    # Check if CUDA is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #######

    print('Starting the script...')

    num_classes = 37

    model = resnet18(pretrained=False)
    model.fc = torch.nn.Identity()

    # Load the state dict from the pretraining
    pretrained_state_dict = torch.load('res18CLpretrain.pt')

    # Filter out the weights for the projection head
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('projection') and not k.startswith('reduction')}

    # Load the filtered state dict into the model, missing keys are ignored
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0) 

    #######

    class SegmentationHead(torch.nn.Module):
        def __init__(self, in_channels, num_classes):
            super(SegmentationHead, self).__init__()
            # Example segmentation head: a simple upsample followed by a 1x1 conv to get class scores
            self.upsample = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = torch.nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

        def forward(self, x):
            x = self.upsample(x)
            x = self.conv(x)
            return x
        
    #######

    # Changing the segmentation head

    model.segmentation_head = SegmentationHead(in_channels=512, num_classes=num_classes)

    # Freezing layers

    for param in model.parameters():
        param.requires_grad = True

    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    #######

    print('Preping dataser...')

    batch_size = 24
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pets_path = os.path.join(base_dir, 'OxfordPets')

    transform = transforms.Compose(
        [transforms.Resize((224,224)),
            transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    pets_dataset = torchvision.datasets.OxfordIIITPet(root=pets_path, download=True, transform=transform)

    develop_size = int(len(pets_dataset) * 0.8) # Test at 20%
    develop_set, test_set = random_split(pets_dataset, [develop_size, int(len(pets_dataset) - develop_size)])
    train_size = int(len(pets_dataset) * 0.8 * 0.9) # Train at 72%, Val at 8%
    train_set, val_set = random_split(develop_set, [train_size, int(len(develop_set) - train_size)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4) 

    #######

    print('Model traning...')

    for epoch in range(20):
        print(f'Starting epoch {epoch+1}\n')
        
        model.train()
        train_loss = 0
        train_epoch_start = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):

            # Move images to the device
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            features = model(images)

            loss = criterion(features, targets)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()


        train_epoch_end = train_epoch_start - time.time()

        model.eval()
        validation_loss = 0
        with torch.no_grad():  
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = targets.to(device)

                features = model(images)
                predicted_labels = torch.argmax(features, dim=1)

                loss = criterion(features, targets)
                validation_loss += loss.item()
        
        print(predicted_labels)


        print(f"Epoch {epoch+1} complete\n Average Train Loss: {train_loss / len(train_loader)}\n Time taken: {train_epoch_end}\n Average Validation Loss: {validation_loss / len(val_loader)}")


    # model.train()
    # for images, targets in train_loader:
    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()

    #######

    print('Training done')
    torch.save(model.state_dict(), 'saved_CL_segmentation_model.pt')
    print('Model saved.')

if __name__ == '__main__':
    CL_seg_train()