import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.annotations.iloc[idx, 0]}.jpg")
        image = Image.open(img_name)
        labels = self.annotations.iloc[idx, 1:].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=20),
    transforms.ToTensor()
])

train_dataset = SkinLesionDataset(csv_file='dataset/ISIC_2019_Training_GroundTruth.csv',
                                  root_dir='./dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
                                  transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class SkinLesionClassifier(nn.Module):
    def __init__(self):
        super(SkinLesionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SkinLesionClassifier().to(device)

class_weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
class_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(weight=class_weight)

optimizer = optim.Adam(model.parameters(), lr=0.001)


def calculate_accuracy(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(outputs == labels)
    accuracy = correct.float() / labels.numel()
    return accuracy.item()


if __name__ == "__main__":
    num_epochs = 10
    train_acc, val_acc = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for i, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(torch.sigmoid(outputs), labels)
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        train_acc.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                val_running_accuracy += calculate_accuracy(torch.sigmoid(outputs), labels)

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracy = val_running_accuracy / len(val_loader)
        val_acc.append(val_epoch_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

    torch.save(model.state_dict(), "skin_lesion_classifier24.pth")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    accuracy = calculate_accuracy(torch.tensor(all_preds_array), torch.tensor(all_labels_array))
    print(f"Overall Accuracy: {accuracy:.4f}")

    mlb = MultiLabelBinarizer()
    all_labels_binarized = mlb.fit_transform(all_labels_array)
    all_preds_binarized = mlb.transform(all_preds_array)

    cm = confusion_matrix(all_labels_binarized.argmax(axis=1), all_preds_binarized.argmax(axis=1))


def plot_accuracy(train_acc, val_acc):
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    plot_accuracy(train_acc, val_acc)

    plot_confusion_matrix(cm, mlb.classes_)
