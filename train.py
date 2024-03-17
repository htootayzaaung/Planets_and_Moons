import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import os
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
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)

        box = torch.tensor([[0, 0, 256, 144]], dtype=torch.float32)
        labels = torch.tensor([class_index], dtype=torch.int64)

        target = {'boxes': box, 'labels': labels}
        return image, target

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

if __name__ == "__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '')  # Relative path to data directory

    dataset = PlanetsMoonsDataset(root_dir=data_dir, transform=None)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    num_classes = len(dataset.classes) + 1
    model = get_model(num_classes).to(device)
    
    num_epochs = 10
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        torch.save(model.state_dict(), f'fasterrcnn_resnet50_fpn_epoch{epoch}.pth')

    print("Training finished!")
