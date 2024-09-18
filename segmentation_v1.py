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


def iou_score(output, target):
    # output = output < 0.5  # Apply a threshold to the outputs
    # target = target < 0.5  # Apply a threshold to the targets

    output = torch.argmax(output, dim=1)
    target = target.long()


    intersection = (output & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (output | target).float().sum((1, 2))         # Will be zero if both are 0

    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our division to avoid 0/0

    return iou.mean()  # Average across the batch


# convert normalized images to original
def denormalize(tensor): 
    tensor = tensor.clone().detach()  
    for t, m, s in zip(tensor, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])):
        t.mul_(s).add_(m)  # Multiply by std dev and then add the mean
    return tensor

def save_sample_images(images, gt_masks, pred_masks, epoch):
    if not os.path.exists('./images_default'):
        os.makedirs('./images_default')
    base_path = './images_default'
    for idx in range(len(images[0])):
        img_tensor = denormalize(images[idx])
        im = transforms.ToPILImage()(img_tensor)
        im.save(os.path.join(base_path, f'image_{epoch}_{idx}.png'))

        gt_mask = transforms.ToPILImage()(gt_masks[idx])
        gt_mask.save(os.path.join(base_path, f'target_{epoch}_{idx}.png'))

        pred_mask = transforms.ToPILImage()(pred_masks[idx] / 2.0)
        pred_mask.save(os.path.join(base_path, f'pred_{epoch}_{idx}.png'))

def CL_seg_train():

    # Check if CUDA is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #######

    print('Starting the script...')

    num_classes = 37

    model = resnet18(pretrained=False)
    model.fc = torch.nn.Identity()
    model.load_state_dict(torch.load('resnet18_pretrain_default_weights_64_0.0001.pt'), strict=False)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
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

    validation_size = len(val_set)
    test_size = len(test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4) 

    #######

    print('Model traning...')

    # Freezing layers
    for param in model.parameters():
        param.requires_grad = True
    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    sum_iou = []

    print('Model training...')
    
    with open('resnet50_segmentation_default_weights.txt', 'a') as f:
        print(f'batch size: {batch_size}, learning rate: 1e-4, weight decay: 1e-5', file=f, flush=True)
        for epoch in range(20):
            print(f'Starting epoch {epoch+1}, train size: {train_size}, validation size: {validation_size}, test size: {test_size} \n', file=f, flush=True)
            
            model.train()
            train_loss = 0
            train_epoch_start = time.time()
            for i, (images, targets) in enumerate(train_loader):
                
                # Move images to the device
                images = images.to(device)
                targets = targets.long().to(device)

                optimizer.zero_grad()

                features = model(images)

                loss = criterion(features, targets)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: { train_loss/ (i+1):.3f}', file=f, flush=True)
            scheduler.step()

            train_epoch_end = train_epoch_start - time.time()

            model.eval()
            validation_loss = 0
            total_iou = 0
            with torch.no_grad():  
                for batch_idx, (images, targets) in enumerate(val_loader):
                    images = images.to(device)
                    targets = targets.long().to(device)

                    features = model(images)

                    loss = criterion(features, targets)
                    validation_loss += loss.item()

                    # iou = iou_score(features, targets)
                    # total_iou += iou.item()
                    # sum_iou.append(total_iou / len(val_loader))

                print(f"Epoch {epoch+1} complete\n Average Train Loss: {train_loss / len(val_loader)}\n Time taken: {train_epoch_end}\n Average Validation Loss: {validation_loss / len(val_loader)}", file=f, flush=True)


            # test loop starts
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_loader):
                    images = images.to(device)
                    targets = targets.to(device)
                    targets = (targets * 225 - 1)/2

                    predictions = model(images)
                    pred = nn.Softmax(dim=1)(predictions)

                    pred_labels = pred.argmax(dim=0)
                    # Add a value 1 dimension at dim=1
                    pred_labels = pred_labels.unsqueeze(1)
                    # print("pred_labels.shape: {}".format(pred_labels.shape))
                    pred_mask = pred_labels.to(torch.float)
                        
                save_sample_images(images, targets, pred_mask, epoch)  

        # top1_iou = max(sum_iou)
        # mean_iou = sum(sum_iou) / len(sum_iou)
        # print(f'Top 1 iou: {top1_iou}, Mean iou: {mean_iou}', file=f, flush=True)

    print('Training done')
    torch.save(model.state_dict(), 'saved_CL_segmentation_model.pt')
    print('Model saved.')

if __name__ == '__main__':
    CL_seg_train()