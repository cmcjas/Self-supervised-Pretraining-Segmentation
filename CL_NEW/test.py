import torch, timm
import os
import PIL
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

from enum import IntEnum

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

def denormalize(tensor): 
    tensor = tensor.clone().detach()  
    for t, m, s in zip(tensor, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])):
        t.mul_(s).add_(m)  # Multiply by std dev and then add the mean
    return tensor

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

def viz():

    SEED = 56 # You can choose any number as your seed
    # Set the random seed for reproducibility
    torch.manual_seed(SEED)

    batch_size = 64

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
    pets_path_test = os.path.join(base_dir, 'OxfordPets', 'test')
    pets_test_orig = torchvision.datasets.OxfordIIITPet(root=pets_path_test, split="test", target_types="segmentation", download=True, transform=transform, target_transform=target_transform)

    test_loader = DataLoader(pets_test_orig, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MaskingModel(num_classes=3).to(device)
    # Load the trained model
    model.load_state_dict(torch.load("./segmentation_ver3_stl10_8_0.001_model.pth"))
    # model.load_state_dict(torch.load("./Simon/spark_seg/model_res/segmentation_model_bs08_lr001.pth"))
    model.eval()

    # # Get a sample image and its corresponding mask
    # sample_image, sample_mask = next(iter(test_loader))
    # sample_image = sample_image.to(device)
    # sample_mask = sample_mask.to(device)

    # # Perform inference
    # with torch.no_grad():
    #     output_mask = model(sample_image)

    # print(model)

    # sample_image = sample_image[0]
    # # sample_image = sample_image / 2
    # # Denormalize the image
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # for i in range(3):
    #     sample_image[i] = sample_image[i] * std[i] + mean[i]
    # sample_image = t2img(sample_image)
    # # sample_image.show()

    # sample_mask = sample_mask[0].float()
    # sample_mask = sample_mask / 2
    # sample_mask = t2img(sample_mask)
    # # sample_mask.show()

    # # print(output_mask.size())
    # output_mask = nn.Softmax(dim=1)(output_mask)
    # output_mask = output_mask.argmax(dim=1)
    # output_mask = output_mask[0].float()
    # output_mask = output_mask / 2
    # # print(output_mask)
    # output_mask = t2img(output_mask)
    # # output_mask.show()

    # # Concatenate the three images horizontally
    # label_margin = 10

    # # Create a blank image to hold the concatenated images and labels
    # result_image = PIL.Image.new('RGB', (sample_image.width * 3, sample_image.height + label_margin * 2))

    # # Paste the sample image
    # sample_image_with_label = PIL.Image.new('RGB', (sample_image.width, sample_image.height + label_margin))
    # sample_image_with_label.paste(sample_image, (0, label_margin))
    # sample_image_label = PIL.ImageDraw.Draw(sample_image_with_label)
    # sample_image_label.text((0, 0), "Ground Truth")
    # result_image.paste(sample_image_with_label, (0, label_margin))

    # # Paste the sample mask with label
    # sample_mask_with_label = PIL.Image.new('RGB', (sample_mask.width, sample_mask.height + label_margin))
    # sample_mask_with_label.paste(sample_mask, (0, label_margin))
    # sample_mask_label = PIL.ImageDraw.Draw(sample_mask_with_label)
    # sample_mask_label.text((0, 0), "Truth Mask")
    # result_image.paste(sample_mask_with_label, (sample_image.width, label_margin))

    # # Paste the output mask with label
    # output_mask_with_label = PIL.Image.new('RGB', (output_mask.width, output_mask.height + label_margin))
    # output_mask_with_label.paste(output_mask, (0, label_margin))
    # output_mask_label = PIL.ImageDraw.Draw(output_mask_with_label)
    # output_mask_label.text((0, 0), "Predicted Mask")
    # result_image.paste(output_mask_with_label, (sample_image.width * 2, label_margin))

    # result_image.show()
    # # Save the result image
    # result_image.save("./Simon/spark_seg/result_image.jpg")
    # print("Result image saved.")

    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)

    predictions = model(images)
    pred_prob = nn.Softmax(dim=1)(predictions)
    pred_labels = pred_prob.argmax(dim=1)
    pred_mask = pred_labels.to(torch.float)
    pred_mask = pred_mask.unsqueeze(dim=1)
    
    # print(images.size())
    # print(pred_mask.size())


    #visualization

    images_grid = torchvision.utils.make_grid(images[:16], nrow=8)
    img_tensor = denormalize(images_grid)
    im_montage = transforms.ToPILImage()(img_tensor)

    
    masks = masks / 2
    masks_grid = torchvision.utils.make_grid(masks[:16], nrow=8)
    mask_montage = transforms.ToPILImage()(masks_grid)


    pred_mask_grid = torchvision.utils.make_grid(pred_mask[:16], nrow=8)
    pred_mask_montage = transforms.ToPILImage()(pred_mask_grid/2)


    acc, iou = compute_metrics(masks * 2, pred_mask)

    print(f'Pixel Accuracy: {acc}, IoU: {iou}')

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
    img_path = os.path.join(base_dir, 'result_ver3_stl10_8_0.001.png')
    print('result images saved.')
    plt.savefig(img_path)

if __name__ == "__main__":
    viz()