import torch, timm
import os
import PIL
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.NEAREST
except:
    import PIL
    interpolation = PIL.Image.NEAREST

from enum import IntEnum
import time

class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2

# Convert a pytorch tensor into a PIL image
t2img = transforms.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = transforms.ToTensor()

# Convert a float trimap ({1, 2, 3} / 255.0) into a float tensor with
# pixel values in the range 0.0 to 1.0 so that the border pixels
# can be properly displayed.
def trimap2f(trimap):
    return (img2t(trimap) * 255.0 - 1) / 2

def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def tensor_duble(t):
    return t * 2

def compute_metrics(pred_labels, true_labels):
    # Compute pixel-wise accuracy
    correct_pixels = (pred_labels == true_labels).sum().item()
    total_pixels = pred_labels.numel()
    pixel_accuracy = correct_pixels / total_pixels

    # Compute Intersection over Union (IoU)
    intersection = torch.logical_and(pred_labels == true_labels, true_labels != TrimapClasses.BACKGROUND).sum().item()
    union = torch.logical_or(pred_labels != TrimapClasses.BACKGROUND, true_labels != TrimapClasses.BACKGROUND).sum().item()
    iou = (intersection + 1e-6) / (union + 1e-6)

    return pixel_accuracy, iou

class MaskingModel(nn.Module):
    def __init__(self, num_classes):
        super(MaskingModel, self).__init__()
        # model = timm.create_model('resnet50')
        # state = torch.load('./Simon/pretrain/res50_withdecoder_1kpretrained_spark_style.pth', 'cpu')
        model = timm.create_model('resnet18')
        state = torch.load('./resnet18_pretrain_stl10_weights_128_0.0001.pt', 'cpu')
        model.load_state_dict(state.get('module', state), strict=False)
        self.pretrain = nn.Sequential(*list(model.children())[:-2])
        # self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=4, padding=2, stride=2)
        # self.conv1 = nn.Conv2d(512, num_classes, kernel_size=4, padding=2, stride=2)
        # self.upsample = nn.Upsample(scale_factor=56, mode='bilinear', align_corners=True)
        self.upconv = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, num_classes, kernel_size=3, padding=1) 
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.pretrain(x)
        # output = self.conv1(features)
        # output = self.upsample(output)
        output = self.upconv(features)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.upsample(output)
        
        return output
    
def train_model(batch_size, data_size):

    # Assuming you have already loaded and preprocessed the dataset
    num_epochs = 10
    batch_size = batch_size

    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=interpolation), 
                                    # transforms.RandomResizedCrop(224, scale=(0.67, 1.0), interpolation=interpolation),  
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    target_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=interpolation),
                                        # transforms.RandomResizedCrop(224, scale=(0.67, 1.0), interpolation=interpolation),
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Lambda(tensor_trimap)
                                        ])

    # Get dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # print(base_dir)

    # Oxford IIIT Pets Segmentation dataset loaded via torchvision.
    pets_path_train = os.path.join(base_dir, 'OxfordPets', 'train')
    pets_train_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_train, split="trainval", target_types="segmentation", download=True, transform=transform, target_transform=target_transform)

    # # Split the dataset into training and validation sets with part of original dataset
    # subset_indices = int(len(pets_train_orig) * 0.2)
    # real_set, aban_set = random_split(pets_train_orig, [subset_indices, int(len(pets_train_orig) - subset_indices)])

    # train_size = int(len(real_set) * 0.9) # Train at 90%, Val at 10%
    # train_set, val_set = random_split(real_set, [train_size, int(len(real_set) - train_size)])

    # Split the dataset into training and validation sets with full original dataset
    train_size = int(len(pets_train_orig) * data_size) # Train at 90%, Val at 10%
    train_set, val_set = random_split(pets_train_orig, [train_size, int(len(pets_train_orig) - train_size)])

    # Load dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MaskingModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")

    sum_iou = []

    # Training loop
    with open('resnet18_segmentation_weights.txt', 'a') as f:
        print(f"Training ({data_size*100}%) with {num_epochs} epochs, batch size: {batch_size}, learning rate: 0.001", file=f, flush=True)
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            acc = 0
            for images, targets in train_loader:
                acc += 1
                images = images.to(device)
                targets = targets.float().to(device)
                targets = targets.squeeze(1).long()

                optimizer.zero_grad()
                outputs = model(images)
                outputs = outputs.float()  # Convert outputs to Float
                # print(outputs.size())
                # print(outputs.shape)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                step_end = time.time()

                if acc in {1, len(train_loader) / 2,  (len(train_loader) + 1) / 2, len(train_loader)}:
                    print(f"Step [{acc}/{len(train_loader)}], Loss: {loss.item():.4f}, Time taken: {step_end - start_time:.2f}s")
            end_time = time.time()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time taken: {end_time - start_time:.2f}s", file=f, flush=True)

            # Validation loop
            model.eval()
            with torch.no_grad():
                total_loss = 0
                total_iou = 0
                total_pixel = 0
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.float().to(device)
                    targets = targets.squeeze(1).long()

                    outputs = model(images)
                    outputs = outputs.float()  # Convert outputs to Float
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()

                    outputs = nn.Softmax(dim=1)(outputs)

                    outputs = outputs.argmax(dim=1)

                    # print(outputs.size())
                    # print(targets.size())

                    # Compute the IoU
                    pixel, iou = compute_metrics(outputs, targets)  # Apply a threshold to the outputs
                    total_iou += iou
                    total_pixel += pixel

                avg_loss = total_loss / len(val_loader)
                avg_iou = total_iou / len(val_loader)
                avg_pixel = total_pixel / len(val_loader)
                sum_iou.append(avg_iou)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}, Validation IoU: {avg_iou:.4f}, Pixel Accuracy: {avg_pixel:.4f} \n", file=f, flush=True)

        top1_iou = max(sum_iou)
        mean_iou = sum(sum_iou) / len(sum_iou)
        print(f"Top-1 IoU: {top1_iou:.4f}, Mean IoU: {mean_iou:.4f}", file=f, flush=True)
        
        # Save the trained model
        torch.save(model.state_dict(), f"./segmentation_{batch_size}_{data_size}_0.001_model.pth")
        print("Model saved successfully!\n", file=f, flush=True)

if __name__ == "__main__":
    for batch_size in [8, 16, 32]:
        for data_size in [0.9, 0.5, 0.2]:
            train_model(batch_size=batch_size, data_size=data_size)