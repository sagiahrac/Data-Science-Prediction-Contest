import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

from helpers.getmodel import get_data

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
NUM_SAMPLES_PER_EPOCH = BATCH_SIZE * 1000


class CustomImageDataset(Dataset):
    label_to_idx = {
        "cakes_cupcakes_snack_cakes": 0,
        "candy": 1,
        "chips_pretzels_snacks": 2,
        "chocolate": 3,
        "cookies_biscuits": 4,
        "popcorn_peanuts_seeds_related_snacks": 5,
    }
    
    classes = sorted(label_to_idx, key=label_to_idx.get)

    def __init__(self, root_dir, X, y, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.X = X
        self.y = y
        self.nfeatures = X.shape[1] - 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        snack_id = self.X.iloc[idx, 0]
        features = self.X.iloc[idx, 1:].to_numpy()
        label = self.y.iloc[idx]
        if 'test' not in self.root_dir:
            img_path = os.path.join(self.root_dir, label, f"{snack_id}.jpg")
        else:
            img_path = os.path.join(self.root_dir, f"{snack_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return features, image, self.label_to_idx[label]


def get_datasets(X_train, y_train, X_val, y_val):
    root_dir = "data/snacks_images/train"
    root_dir = "data/snacks_images/test"
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_ds = CustomImageDataset(
        root_dir=root_dir, X=X_train, y=y_train, transform=data_transforms["train"]
    )
    val_ds = CustomImageDataset(
        root_dir=root_dir, X=X_val, y=y_val, transform=data_transforms["val"]
    )

    image_datasets = {"train": train_ds, "val": val_ds}
    return image_datasets


def get_dataloaders(image_datasets):
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, num_workers=4)
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_batch_example(dataloaders, class_names):
    # Get a batch of training data
    _, inputs, classes = next(iter(dataloaders["train"]))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


def train_model(
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    checkpoints_dir="checkpoints/resnet18",
):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    best_model_params_path = os.path.join(checkpoints_dir, "best_model_params.pt")

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            loop = tqdm(dataloaders[phase])
            for features, image, labels in loop:
                image = image.to(device)
                labels = labels.to(device)
                # features = features.to(torch.float32).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(image)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * image.size(0)
                running_corrects += torch.sum(preds == labels.data)

                loop.set_description(
                    f"[{phase.capitalize()}] Epoch [{epoch}/{num_epochs-1}]"
                )
                loop.set_postfix(
                    loss=loss.item(),
                    acc=torch.sum(preds == labels.data).item() / len(labels.data),
                )
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / (len(dataloaders[phase]) * 10)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * 10)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (features, images, labels) in enumerate(dataloaders["val"]):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = (outputs > 0).float()

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(images.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def init_and_train_model(dataloaders):
    model_ft = models.resnet18(weights="IMAGENET1K_V1")
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in list(model_ft.fc.parameters())[-10:]:
        param.requires_grad = True
    # model_ft = ResNet18ForSnacks()

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 6.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
    )


class ResNet18ForSnacks(nn.Module):
    def __init__(self):
        super(ResNet18ForSnacks, self).__init__()
        self.base_model = models.resnet18(weights="IMAGENET1K_V1")
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in list(self.base_model.fc.parameters())[-10:]:
            param.requires_grad = True
        self.base_model.fc = self.fc = nn.Linear(self.base_model.fc.in_features, 256)
        self.base_model = self.base_model.to(device)

        self.bn1 = nn.BatchNorm1d(172 + 256)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(172 + 256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 6)

    def forward(self, features, images):
        images = self.base_model(images)
        x = torch.cat((features, images), axis=1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = get_data()
    image_datasets = get_datasets(X_train, y_train, X_val, y_val)
    dataloaders, dataset_sizes, class_names = get_dataloaders(image_datasets)
    init_and_train_model(dataloaders)
