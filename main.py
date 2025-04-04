import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix


# Define transformations
def get_transforms():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return transform


def load_data():
    transform = get_transforms()
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return trainloader, valloader, testloader


def load_cifar100():
    transform = get_transforms()

    # Define 20 selected classes from CIFAR-100
    selected_classes = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
    ]

    # Get full CIFAR-100 dataset
    full_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # Map class names to indices
    class_to_idx = {name: idx for idx, name in enumerate(full_dataset.classes)}
    selected_indices = [class_to_idx[cls] for cls in selected_classes]

    # Filter dataset to include only selected classes
    train_subset = [
        (img, selected_indices.index(label))
        for img, label in full_dataset
        if label in selected_indices
    ]
    test_subset = [
        (img, selected_indices.index(label))
        for img, label in test_dataset
        if label in selected_indices
    ]

    # Convert to PyTorch Dataset
    trainset = torch.utils.data.TensorDataset(
        torch.stack([img for img, _ in train_subset]),
        torch.tensor([label for _, label in train_subset]),
    )
    testset = torch.utils.data.TensorDataset(
        torch.stack([img for img, _ in test_subset]),
        torch.tensor([label for _, label in test_subset]),
    )

    # Create validation set (20% of trainset)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    batch_size = 32
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, valloader, testloader


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # No softmax as it's included in CrossEntropyLoss


class LeNet5_Dropout(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Model Variant 2: Increase Filters + LeakyReLU
class LeNet5_LeakyReLU(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_LeakyReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)  # Increased filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # Increased filters
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Adapted LeNet5_LeakyReLU for CIFAR-100 (20 classes)
class LeNet5_CIFAR100(nn.Module):
    def __init__(self, num_classes=20):
        super(LeNet5_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)  # Increased filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # Increased filters
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, device, trainloader, valloader, criterion, optimizer, epochs=10):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_losses.append(running_loss / len(trainloader))
        train_accs.append(100 * correct / total)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_losses.append(val_loss / len(valloader))
        val_accs.append(100 * correct / total)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%"
        )

    return train_losses, val_losses, train_accs, val_accs


def plot_confusion_matrix(y_true, y_pred, classes=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")


def test_model(model, device, testloader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    plot_confusion_matrix(all_labels, all_preds, classes=cifar10_classes)
    return accuracy


def save_model(model, path="lenet5.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # trainloader, valloader, testloader = load_data()
    trainloader, valloader, testloader = load_cifar100()

    # model = LeNet5().to(device)
    # model = LeNet5_LeakyReLU().to(device)
    # model = LeNet5_Dropout().to(device)
    model = LeNet5_CIFAR100().to(device)

    test = False
    if test:
        # load model and test
        model.load_state_dict(torch.load("lenet5.pth"))
        test_model(model, device, testloader)
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accs, val_accs = train_model(
        model, device, trainloader, valloader, criterion, optimizer, epochs=10
    )
    # save_model(model)
    save_model(model, "lenet5_cifar100.pth")
    # save_model(model, "lenet5_dropout.pth")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    plt.savefig("training.png")

    plt.show()


def fine_tune_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, valloader, testloader = load_data()

    model = LeNet5_CIFAR100(num_classes=20).to(device)  # Load CIFAR-100 model
    model.load_state_dict(
        torch.load("data/weights/lenet5_cifar100.pth", map_location=device),
        strict=False,
    )  # Load weights

    # Modify last layer to 10 classes
    model.fc3 = nn.Linear(84, 10).to(device)

    # Reduce learning rate to half (0.0005)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_losses, val_losses, train_accs, val_accs = train_model(
        model, device, trainloader, valloader, criterion, optimizer, epochs=10
    )
    save_model(model, "lenet5_cifar10_pretrained.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    plt.savefig("training.png")

    plt.show()


if __name__ == "__main__":
    # fine_tune_cifar10()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, valloader, testloader = load_data()
    # model = LeNet5_LeakyReLU().to(device)
    # model.load_state_dict(torch.load("data/weights/lenet5_leaky.pth"))

    # model = LeNet5_CIFAR100(num_classes=20).to(device)
    # model.fc3 = nn.Linear(84, 10).to(device)
    # model.load_state_dict(torch.load("data/weights/lenet5_cifar10_pretrained.pth"))

    # model = LeNet5().to(device)
    # model.load_state_dict(torch.load("data/weights/lenet5.pth"))

    model = LeNet5_Dropout().to(device)
    model.load_state_dict(torch.load("data/weights/lenet5_dropout.pth"))

    test_model(model, device, testloader)

