import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import os
import numpy as np
from PIL import Image
import torch.optim as optim

class PlanetsMoonsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.imgs = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(root_dir, target_class)
            for root, _, fnames in os.walk(target_dir):
                for fname in sorted(fnames):
                    if fname.endswith('.jpg'):
                        path = os.path.join(root, fname)
                        item = path, class_index
                        self.imgs.append(item)
                        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, class_index = self.imgs[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        # Since objects cover the full image, we create a bounding box covering the whole image
        # Box format: [xmin, ymin, xmax, ymax]
        box = torch.tensor([0, 0, 256, 144], dtype=torch.float32)
        # As there is only one class per image, we use a tensor of zeros to indicate the single class
        labels = torch.tensor([class_index], dtype=torch.int64)
        
        target = {}
        target['boxes'] = box.unsqueeze(0)  # Add an extra dimension as FasterRCNN expects a batch of boxes
        target['labels'] = labels
        
        return image, target

# Rest of the PlanetsMoonsDataset class code...

# Function to get the object detection model
def get_model(num_classes):
    # Load a pre-trained model for object detection
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Main code
if __name__ == "__main__":
    # Define the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Set the root_dir to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Planets_and_Moons')  # Adjust the relative path to the data as needed

    # Use our dataset and defined transformations
    dataset = PlanetsMoonsDataset(root_dir=data_dir, transform=T.ToTensor())
    
    # Define the dataloader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # Get the model using our helper function and move it to the right device
    num_classes = len(dataset.classes) + 1 # +1 for the background class
    model = get_model(num_classes).to(device)
    
    # Parameters
    num_epochs = 10
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Iteration #{i} loss: {losses.item()}")
            i += 1

        # Update the learning rate
        lr_scheduler.step()

        # Save the model after each epoch
        torch.save(model.state_dict(), f'fasterrcnn_resnet50_fpn_epoch{epoch}.pth')

    print("Training finished!")