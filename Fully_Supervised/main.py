# Code adapted from segnet example:
# https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb

from model import ResNetSegmentation
from data import load_datasets
import torch
import os
from data import load_datasets
from model import ResNetSegmentation
from utils import to_device, save_test_images, compute_metrics
from torch import nn

def test_model(model, test_loader, test_data, image_save_path):
    """
    Evaluates the trained model on a given dataset using pixel accuracy and Intersection over Union (IoU).

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): The DataLoader providing the dataset for evaluation.

    Prints:
        Average pixel accuracy and IoU across the dataset.
    """
    model.eval()
    total_pixel_accuracy = 0
    total_iou_accuracy = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in test_loader:

            predictions = model(to_device(inputs))
            labels = to_device(targets)
            pred = nn.Softmax(dim=1)(predictions)
            pred_labels = pred.argmax(dim=1)
            pred_labels = pred_labels.unsqueeze(1)
            
            pixel_accuracy, iou_accuracy = compute_metrics(pred_labels, labels)
            
            total_pixel_accuracy += pixel_accuracy
            total_iou_accuracy += iou_accuracy
            count += 1

    average_pixel_accuracy = total_pixel_accuracy / count
    average_iou_accuracy = total_iou_accuracy / count
    print(f'Test model, Average Accuracy[Pixel: {average_pixel_accuracy:.4f}, IoU: {average_iou_accuracy:.4f}]')
    print("")

    # Save images
    test_images, test_targets = test_data

    test_preds = model(to_device(test_images))
    test_preds = nn.Softmax(dim=1)(test_preds)
    test_preds = test_preds.argmax(dim=1)
    test_preds = test_preds.unsqueeze(1)

    save_test_images(test_images, test_targets, test_preds, image_save_path, average_pixel_accuracy, average_iou_accuracy)

def train_model(model, loader, optimizer):
    """
    Trains a model for one epoch using a given data loader and optimizer.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        loader (DataLoader): The DataLoader providing training data.
        optimizer (Optimizer): The optimizer to use for adjusting the weights.

    Returns:
        float: The average loss computed over the epoch.
    """
    to_device(model.train())
    criterion = nn.CrossEntropyLoss(reduction='mean')

    running_loss = 0.0
    running_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)
        
        targets = targets.squeeze(dim=1) # Channel dimension from (NCHW) to (NHW)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        running_samples += targets.size(0)
        running_loss += loss.item()

    return running_loss / (batch_idx+1)

def train_loop(model, train_loader, test_data, epochs, optimizer, scheduler, model_save_path):
    """
    Executes the training process for the specified number of epochs and evaluates the model on test data.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        test_data (tuple): A tuple containing test inputs and labels.
        epochs (int): Number of epochs to train the model.
        optimizer (Optimizer): Optimizer for adjusting the model weights.
        scheduler (lr_scheduler): Learning rate scheduler.
        model_save_path (str): Path to save the trained model.
        image_save_path (str): Path to save visualisation results.
        visualise_results (bool): Flag to determine whether to save visualisation results or not.
    """
    test_inputs, test_targets = test_data
    iou_accuracies = []

    for i in range(epochs):
        loss = train_model(model, train_loader, optimizer)
        with torch.inference_mode():
            to_device(model.eval())
            predictions = model(to_device(test_inputs))
            test_pets_labels = to_device(test_targets)
            pred = nn.Softmax(dim=1)(predictions)
            pred_labels = pred.argmax(dim=1)
            pred_labels = pred_labels.unsqueeze(1)
            
            pixel_accuracy, iou_accuracy = compute_metrics(pred_labels, test_pets_labels)
            title = f'Epoch: {i+1:02d}, Loss: {loss:.4f}, Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}]'
            print(title)
        
            iou_accuracies.append(iou_accuracy)

        if scheduler is not None:
            scheduler.step()
        if (i == epochs - 1):
            average_iou = sum(iou_accuracies) / len(iou_accuracies)
            max_iou = max(iou_accuracies)
            print(f"Finished training: Average IoU = {average_iou:.4f}, Max IoU = {max_iou:.4f}")
            torch.save(model.state_dict(), model_save_path)
            print("Model saved.")
            print("")


def main():
    """
    Main execution function for training and evaluating a ResNet-based image segmentation model.
    
    This function sets up the training and testing environments, iterates through combinations of batch sizes and learning rates, trains the ResNet model, saves the trained models, and evaluates them using a test dataset. It also handles the setup for visualization of results if enabled.
    
    Models trained include ResNet18 and ResNet50, which are selected based on the 'isResNet18' flag.
    
    Args:
        None
    
    Returns:
        None
    """
    working_dir = os.path.dirname(os.path.abspath(__file__))
    isResNet18 = True # False for ResNet50

    for batch_size in [16]:      
        for learning_rate in [0.001]:
            image_save_path = os.path.join(
                working_dir,
                "resnet18_training_images" if isResNet18 else "resnet50_training_images",
                f"result_image_bs{batch_size}_lr{str(learning_rate).replace('.', '')}"
            )
            model_save_path = os.path.join(working_dir, f"{'resnet18' if isResNet18 else 'resnet50'}_bs{batch_size}_lr{str(learning_rate).replace('.', '')}.pt")
            pets_train_loader, pets_test_loader = load_datasets(working_dir, batch_size)
            (test_pets_inputs, test_pets_targets) = next(iter(pets_test_loader))  
            model = ResNetSegmentation(n_classes=3, isResNet18=isResNet18) 
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
            print(f"Training with: {'ResNet18' if isResNet18 else 'ResNet50'}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
            train_loop(model, pets_train_loader, (test_pets_inputs, test_pets_targets), 10, optimizer, scheduler, model_save_path)
    
            ### To only test model, comment out lines 168-169, and uncomment lines 172-178 (Must have weights in folder)
            # print(f"Testing with: {'ResNet18' if isResNet18 else 'ResNet50'}")
            # if isResNet18:
            #    model.load_state_dict(torch.load(f'resnet18_bs16_lr0001.pt'))
            #    image_save_path = os.path.join(working_dir, "resnet18_training_images", "result_image_bs16_lr0001")
            # else:
            #    model.load_state_dict(torch.load(f'resnet50_bs16_lr00001.pt'))
            #    image_save_path = os.path.join(working_dir, "resnet50_training_images", "result_image_bs16_lr00001")
            test_model(model, pets_test_loader, (test_pets_inputs, test_pets_targets), image_save_path)

if __name__ == '__main__':
    main()
