import os
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

data_dir = 'data/train'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


full_dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
