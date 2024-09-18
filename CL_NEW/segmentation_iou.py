import torch
from torch import optim
import torchvision
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import time 
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from enum import IntEnum
import matplotlib.pyplot as plt


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels // 2, num_classes, kernel_size=3, padding=1)
  
    def forward(self, x):
        x = self.upsample(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    

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

class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 0.5
    BORDER = 1


def compute_metrics(pred_labels, true_labels):
    # Compute pixel-wise accuracy
    correct_pixels = (pred_labels == true_labels).sum().item()
    total_pixels = pred_labels.numel()
    pixel_accuracy = correct_pixels / total_pixels

    # Compute Intersection over Union (IoU)
    intersection = torch.logical_and(pred_labels == true_labels, true_labels != TrimapClasses.BACKGROUND).sum().item()
    union = torch.logical_or(pred_labels != TrimapClasses.BACKGROUND, true_labels != TrimapClasses.BACKGROUND).sum().item()
    iou_accuracy = intersection / union

    return pixel_accuracy, iou_accuracy


def test_result(masks, pred_mask, im_montage, mask_montage, pred_mask_montage, epoch, batch_size, learning_rate):

    if not os.path.exists('./visualisations_result'):
        os.makedirs('./visualisations_result')

    acc, iou = compute_metrics(pred_mask/2, masks)
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))

    # Plot each montage with titles
    axes[0].imshow(im_montage)
    axes[0].set_title('Images')
    axes[0].axis('off')  

    axes[1].imshow(mask_montage)
    axes[1].set_title('Ground Truth Masks')
    axes[1].axis('off')

    axes[2].imshow(pred_mask_montage)
    axes[2].set_title('Predicted Masks')
    axes[2].axis('off')

    axes[3].set_title(f'Pixel Accuracy: {acc}, IoU: {iou}')
    axes[3].axis('off')

    # plt.show()
    img_path = os.path.join('./visualisations_result', f'result_montage_{epoch+1}_{batch_size}_{learning_rate}.png')
    print('result montage saved.')
    plt.savefig(img_path)


def segmentation_iou(batch_size, learning_rate, weight_decay):

    SEED = 56 # You can choose any number as your seed
    # Set the random seed for reproducibility
    torch.manual_seed(SEED)

    best_pixel_accuracy = 0.0
    old_pixel_accuracy = 0.0
    epochs_no_improve = 0
    n_epochs_stop = 5  # Number of epochs to stop after no improvement
    
    #data preparation
    print('Preparing dataset...')

    batch_size = batch_size
    transform = Transform()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pets_path = os.path.join(base_dir, 'OxfordPets')

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

    num_classes = 37

    backbone = resnet18(weights=None)
    filtered_state_dict = torch.load('resnet18_pretrain_modify_weights_128_0.0001.pt')

    # Load the filtered state dict into the model, missing keys are ignored
    backbone.load_state_dict(filtered_state_dict, strict=False)
    backbone.fc = torch.nn.Identity()

    segmentation_head = SegmentationHead(in_channels=512, num_classes=num_classes)
    model = ResnetwithSegHead(backbone, segmentation_head)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0) 

    # Freezing layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.segmentation_head.parameters():
        param.requires_grad = True

    print('Model training...')
    
    with open('resnet18_segmentation_weights_1.txt', 'a') as f:
        print(f'batch size: {batch_size}, learning rate: {learning_rate}, weight decay: {weight_decay}', file=f, flush=True)
        for epoch in range(20):
            print(f'Starting epoch {epoch+1}\n', file=f, flush=True)
            
            model.train()
            train_loss = 0
            train_epoch_start = time.time()
            for i, (images, targets) in enumerate(train_loader):
                
                # Move images to the device
                images = images.to(device)
                
                # print(targets)
                # print(targets.size())
                # print(torch.unique(targets))
                # # if targets.size(1) == 1:
                # #     targets = targets.squeeze(1).long().to(device)

                # print(targets)
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

                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: { train_loss/ (i+1):.3f}', file=f, flush=True)

            scheduler.step()

            train_epoch_end = time.time()

            # evaluation mode
            model.eval()
            validation_loss = 0
            with torch.no_grad():  
                for batch_idx, (images, targets) in enumerate(val_loader):
                    images = images.to(device)
                    targets_converted = (targets*255 -1).long()
                    targets_converted = targets_converted.squeeze(1).long().to(device)
            
                    features = model(images)

                    loss = criterion(features, targets_converted)
                    validation_loss += loss.item()

            print(f"Epoch {epoch+1} complete\n Average Train Loss: {train_loss / len(train_loader)}\n Time taken: {train_epoch_end - train_epoch_start}\n Average Validation Loss: {validation_loss / len(val_loader)} \n", file=f, flush=True)
            
            # test loop starts
            model.eval()
            avg_pixel_accuracy = 0
            with torch.no_grad():  
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    predictions = model(images)
                    pred = nn.Softmax(dim=1)(predictions)
                    pred_labels = pred.argmax(dim=1)
                    # Add a value 1 dimension at dim=1
                    pred_labels = pred_labels.unsqueeze(1)
                    pred_mask = pred_labels.to(torch.float)

                    images_grid = torchvision.utils.make_grid(images[:16], nrow=8)
                    img_tensor = denormalize(images_grid)
                    im_montage = transforms.ToPILImage()(img_tensor)

                    masks = (masks * 255 - 1)/2
                    masks_grid = torchvision.utils.make_grid(masks[:16], nrow=8)
                    mask_montage = transforms.ToPILImage()(masks_grid)

                    pred_mask_grid = torchvision.utils.make_grid(pred_mask[:16], nrow=8)
                    pred_mask_montage = transforms.ToPILImage()(pred_mask_grid/2)
                    
                    pixle_accuracy, iou_accuracy = compute_metrics(pred_mask/2, masks)
                    avg_pixel_accuracy = pixle_accuracy 
                
            test_result(masks, pred_mask, im_montage, mask_montage, pred_mask_montage, epoch, batch_size, learning_rate)
            print(f'IoU accuracy: {iou_accuracy:.6f}', file=f, flush=True)
            # Check for early stopping
            if avg_pixel_accuracy > best_pixel_accuracy:
                old_pixel_accuracy = best_pixel_accuracy
                best_pixel_accuracy= avg_pixel_accuracy
                torch.save(model.state_dict(), f"CL_segmentation_{batch_size}_{learning_rate}.pt")
                epochs_no_improve = 0
                print(f"Pixel accuracy increased ({old_pixel_accuracy:.6f} --> {best_pixel_accuracy:.6f}).  Saving model ...\n", file=f, flush=True)
            else:
                epochs_no_improve += 1
                print(f"Pixel accuracy did not improve, count: {epochs_no_improve} \n", file=f, flush=True)
                if epochs_no_improve == n_epochs_stop:
                    print(f'Early stopping!', file=f, flush=True)
                    print(f'Best pixel accuracy: {best_pixel_accuracy:.6f} \n', file=f, flush=True)
                    break

            print(f'Best pixel accuracy: {best_pixel_accuracy:.6f} \n', file=f, flush=True)
            print('Training done')


if __name__ == "__main__":
        for batch_size in [16, 32, 64]:
            for learning_rate in [0.001, 0.0001, 0.00001]:
                for weight_decay in [learning_rate/10]:
                    segmentation_iou(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)






    

    
