import os
import glob
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torch.optim as optim

INPUT_IMG_SZ = 112
IMG_DIR = "data/cat_dog_dataset/images"
ANNOTATION_DIR = "data/cat_dog_dataset/annotations"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)  # Default to -1 if unknown label
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)
        # TODO: tmp``
        label = objects[0]["label"] if objects else -1
        bbox = objects[0]["bbox"] if objects else [0, 0, 0, 0]

        # Scale bounding boxes
        scaler_x, scaler_y = width / INPUT_IMG_SZ, height / INPUT_IMG_SZ
        bbox = [
            bbox[0] / scaler_x,
            bbox[1] / scaler_y,
            bbox[2] / scaler_x,
            bbox[3] / scaler_y,
        ]

        if self.transform:
            image = self.transform(image)

        # Dummy YOLO target: [x, y, w, h, obj, class0, class1]
        x_center = (bbox[0] + bbox[2]) / 2 / INPUT_IMG_SZ
        y_center = (bbox[1] + bbox[3]) / 2 / INPUT_IMG_SZ
        box_w = (bbox[2] - bbox[0]) / INPUT_IMG_SZ
        box_h = (bbox[3] - bbox[1]) / INPUT_IMG_SZ
        obj = 1
        class_vec = [1, 0] if label == 0 else [0, 1]
        target = torch.tensor([x_center, y_center, box_w, box_h, obj] + class_vec)

        return image, target

    def __len__(self):
        return len(self.img_files)


class YOLOv1Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 7),  # [x, y, w, h, obj, class0, class1]
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def yolo_loss(pred, target):
    return nn.MSELoss()(pred, target)


def train():
    transform = T.Compose([T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)), T.ToTensor()])
    dataset = CatDogDataset(IMG_DIR, ANNOTATION_DIR, transform)

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = YOLOv1Tiny().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = yolo_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                loss = yolo_loss(preds, targets)
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}"
        )

    torch.save(model.state_dict(), "cat_dog_detector.pth")
    print("Model saved to cat_dog_detector.pth")


if __name__ == "__main__":
    train()
