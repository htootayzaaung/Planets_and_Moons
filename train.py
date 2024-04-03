import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models.mobilenet as mobilenet
import torchvision.ops.feature_pyramid_network as fpn
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN, AnchorGenerator, GeneralizedRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {}  # Mapping class names to integers
        self.img_labels = []
        
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            self.class_to_idx[class_name] = idx  # Assign an integer label to each class name
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_name)
                    self.img_labels.append((img_path, class_name))
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path, class_name = self.img_labels[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]  # Convert class name to its corresponding integer label
        label = torch.tensor(label)
        return image, label

def mobilenet_v3_large_fpn_backbone(pretrained=True, trainable_layers=3):
    # Load a pre-trained MobileNetV3-Large model
    mobilenet_backbone = mobilenet.mobilenet_v3_large(pretrained=pretrained)
    # Select layers that produce the features at different scales
    return_layers = {'0': '0', '1': '1', '11': '2', '13': '3'}

    # Create an IntermediateLayerGetter that extracts features from the selected layers
    in_features = mobilenet_backbone.features[:14]
    out_channels = 960  # The number of output channels from the last layer in the selected layers
    backbone = IntermediateLayerGetter(in_features, return_layers)

    # Define the sizes of the feature maps generated by the FPN
    in_channels_list = [16, 24, 96, out_channels]
    out_channels = 256
    extra_blocks = fpn.LastLevelMaxPool()

    # Create the FPN on top of the backbone
    backbone_fpn = backbone_utils.BackboneWithFPN(backbone, in_channels_list, out_channels, extra_blocks)

    # Make the specified number of layers trainable
    assert trainable_layers <= 4 and trainable_layers >= 0
    layers_to_train = ['features'] if trainable_layers == 4 else ['features.' + str(i) for i in range(14 - trainable_layers * 3, 14)]
    for name, parameter in mobilenet_backbone.named_parameters():
        # Freeze layers that are not in the layers_to_train list
        parameter.requires_grad = name.startswith(tuple(layers_to_train))

    return backbone_fpn

def get_model(num_classes):
    # Create the custom backbone with FPN
    backbone = mobilenet_v3_large_fpn_backbone(pretrained=True, trainable_layers=3)
    backbone.out_channels = 256
    
    # Create an anchor generator for the RPN that will be used by the Faster R-CNN model
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    
    # Even without resizing, normalization is important for model training
    # These are standard ImageNet means and standard deviations
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    # Since all your images are already at the desired size, we skip the resizing step
    transform = GeneralizedRCNNTransform(min_size=144, max_size=256, image_mean=image_mean, image_std=image_std, size_divisible=32)
    
    # Initialize the Faster R-CNN model with the custom backbone
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=None,  # Using the default BoxRoIPool
                       transform=transform)

    return model

def main():
    # Dataset and DataLoader setup
    dataset_root_dir = os.path.dirname(os.path.abspath(__file__))
    transforms = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = CustomDataset(root_dir=dataset_root_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Model initialization
    num_classes = 12  # Include background as a class
    model = get_model(num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Optimizer setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            images = list(image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for image in images)
            targets = [{k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    main()