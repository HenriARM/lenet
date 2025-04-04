import os
import glob
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image, ImageOps
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches


GRID_SIZE = 7
INPUT_IMG_SZ = 112
IMG_DIR = "data/cat_dog_dataset/images"
ANNOTATION_DIR = "data/cat_dog_dataset/annotations"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(dataset, test_size=0.2):
    all_labels = []
    for ann_path in dataset.ann_files:
        tree = ET.parse(ann_path)
        root = tree.getroot()
        obj_name = root.find("object/name").text
        label = dataset.label_map[obj_name]
        all_labels.append(label)

    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=all_labels,
        random_state=42,
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return train_ds, val_ds


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

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        # will resize the image down to fit within the INPUT_IMG_SZxINPUT_IMG_SZ box
        # while preserving aspect ratio, and then pad any remaining space with black (or any color you specify).
        image = ImageOps.pad(image, (INPUT_IMG_SZ, INPUT_IMG_SZ))  # pad to 112x112

        width, height, objects = self.parse_annotation(ann_path)

        if self.transform:
            image = self.transform(image)

        target = torch.zeros((GRID_SIZE, GRID_SIZE, 7))
        cell_size_x = INPUT_IMG_SZ / GRID_SIZE
        cell_size_y = INPUT_IMG_SZ / GRID_SIZE

        for obj in objects:
            label = obj["label"]
            bbox = obj["bbox"]

            # Scale bbox to model input size
            bbox = [
                bbox[0] * INPUT_IMG_SZ / width,
                bbox[1] * INPUT_IMG_SZ / height,
                bbox[2] * INPUT_IMG_SZ / width,
                bbox[3] * INPUT_IMG_SZ / height,
            ]

            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            i = int(y_center // cell_size_y)
            j = int(x_center // cell_size_x)

            x_cell = (x_center - j * cell_size_x) / cell_size_x
            y_cell = (y_center - i * cell_size_y) / cell_size_y

            box_w /= INPUT_IMG_SZ
            box_h /= INPUT_IMG_SZ

            target[i, j, 0:4] = torch.tensor([x_cell, y_cell, box_w, box_h])
            target[i, j, 4] = 1.0
            target[i, j, 5 + label] = 1.0

        return image, target

    def __len__(self):
        return len(self.img_files)


class YOLOv1Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 343),  # 7x7 grid * 7 values per cell
            nn.Sigmoid(),  # squash to [0, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, GRID_SIZE, GRID_SIZE, 7)
        # print("Output shape:", x.shape)
        return x


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union != 0 else 0


def visualize_predictions(images, targets, preds, epoch):
    os.makedirs("epoch_vis", exist_ok=True)
    for idx in range(min(4, len(images))):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if targets[idx][i, j, 4] > 0:
                    x, y, w, h = targets[idx][i, j, :4].tolist()
                    cx = (j + x) * INPUT_IMG_SZ / GRID_SIZE
                    cy = (i + y) * INPUT_IMG_SZ / GRID_SIZE
                    bw = w * INPUT_IMG_SZ
                    bh = h * INPUT_IMG_SZ
                    rect = patches.Rectangle(
                        (cx - bw / 2, cy - bh / 2),
                        bw,
                        bh,
                        linewidth=2,
                        edgecolor="g",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                if preds[idx][i, j, 4] > 0.5:
                    x, y, w, h = preds[idx][i, j, :4].tolist()
                    cx = (j + x) * INPUT_IMG_SZ / GRID_SIZE
                    cy = (i + y) * INPUT_IMG_SZ / GRID_SIZE
                    bw = w * INPUT_IMG_SZ
                    bh = h * INPUT_IMG_SZ
                    rect = patches.Rectangle(
                        (cx - bw / 2, cy - bh / 2),
                        bw,
                        bh,
                        linewidth=2,
                        edgecolor="r",
                        linestyle="--",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

        plt.axis("off")
        plt.savefig(f"epoch_vis/epoch_{epoch}_img_{idx}.png")
        plt.close()


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


def yolo_loss(pred, target, lambda_coord=5, lambda_noobj=0.5):
    obj_mask = target[..., 4] == 1
    noobj_mask = target[..., 4] == 0
    coord_loss = (
        lambda_coord
        * ((pred[..., 0:2][obj_mask] - target[..., 0:2][obj_mask]) ** 2).sum()
    )
    wh_loss = (
        lambda_coord
        * (
            (
                torch.sqrt(pred[..., 2:4][obj_mask] + 1e-6)
                - torch.sqrt(target[..., 2:4][obj_mask] + 1e-6)
            )
            ** 2
        ).sum()
    )
    obj_loss = ((pred[..., 4][obj_mask] - target[..., 4][obj_mask]) ** 2).sum()
    noobj_loss = (
        lambda_noobj
        * ((pred[..., 4][noobj_mask] - target[..., 4][noobj_mask]) ** 2).sum()
    )
    class_loss = ((pred[..., 5:][obj_mask] - target[..., 5:][obj_mask]) ** 2).sum()
    return coord_loss + wh_loss + obj_loss + noobj_loss + class_loss


def train():
    transform = T.Compose([T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)), T.ToTensor()])
    dataset = CatDogDataset(IMG_DIR, ANNOTATION_DIR, transform)

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = stratified_split(dataset)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = YOLOv1Tiny().to(device)
    count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses, val_ious = [], [], []

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
        all_ious = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                val_loss += yolo_loss(preds, targets).item()
                for b in range(len(imgs)):
                    for i in range(GRID_SIZE):
                        for j in range(GRID_SIZE):
                            if targets[b, i, j, 4] == 1 and preds[b, i, j, 4] > 0.5:
                                tx, ty, tw, th = targets[b, i, j, :4]
                                px, py, pw, ph = preds[b, i, j, :4]
                                t_box = [
                                    (j + tx.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    - tw.item() * INPUT_IMG_SZ / 2,
                                    (i + ty.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    - th.item() * INPUT_IMG_SZ / 2,
                                    (j + tx.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    + tw.item() * INPUT_IMG_SZ / 2,
                                    (i + ty.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    + th.item() * INPUT_IMG_SZ / 2,
                                ]
                                p_box = [
                                    (j + px.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    - pw.item() * INPUT_IMG_SZ / 2,
                                    (i + py.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    - ph.item() * INPUT_IMG_SZ / 2,
                                    (j + px.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    + pw.item() * INPUT_IMG_SZ / 2,
                                    (i + py.item()) * INPUT_IMG_SZ / GRID_SIZE
                                    + ph.item() * INPUT_IMG_SZ / 2,
                                ]
                                all_ious.append(iou(t_box, p_box))

            visualize_predictions(imgs, targets, preds, epoch)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_ious.append(avg_iou)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val IoU: {avg_iou:.4f}"
        )

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.plot(val_ious, label="Val IoU")
        plt.legend()
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.savefig(f"training_progress_epoch_{epoch+1}.png")
        plt.close()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "cat_dog_detector.pth")
            print("Model saved (new best)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(), "cat_dog_detector.pth")
    print("Model saved to cat_dog_detector.pth")


if __name__ == "__main__":
    train()
