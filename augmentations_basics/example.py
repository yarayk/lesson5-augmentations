import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from datasets import CustomImageDataset
from utils import show_images, show_single_augmentation, show_multiple_augmentations
from extra_augs import (
    AddGaussianNoise, RandomErasingCustom, CutOut,
    Solarize, Posterize, AutoContrast, ElasticTransform
)
from  standard_augs import get_standard_transforms

# Загрузка датасета без аугментаций
root = '../data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))

# Берём одно изображение для демонстрации
original_img, label = dataset[0]
class_names = dataset.get_class_names()
print(f"Оригинальное изображение, класс: {class_names[label]}")

# === Демонстрация кастомных аугментаций ===
print("\n=== Демонстрация отдельных кастомных аугментаций ===")

custom_augs = [
    ("Гауссов шум", transforms.Compose([transforms.ToTensor(), AddGaussianNoise(0., 0.2)])),
    ("Random Erasing", transforms.Compose([transforms.ToTensor(), RandomErasingCustom(p=1.0)])),
    ("CutOut", transforms.Compose([transforms.ToTensor(), CutOut(p=1.0, size=(32, 32))])),
    ("Solarize", transforms.Compose([transforms.ToTensor(), Solarize(threshold=128)])),
    ("Posterize", transforms.Compose([transforms.ToTensor(), Posterize(bits=4)])),
    ("AutoContrast", transforms.Compose([transforms.ToTensor(), AutoContrast(p=1.0)])),
    ("Elastic Transform", transforms.Compose([transforms.ToTensor(), ElasticTransform(p=1.0, alpha=1, sigma=50)])),
]

for name, aug in custom_augs:
    aug_img = aug(original_img)
    show_single_augmentation(original_img, aug_img, name)

# === Демонстрация стандартных аугментаций torchvision ===
print("\n=== Визуализация стандартных аугментаций (torchvision) ===")
std_transform = get_standard_transforms()

# Создадим 8 версий изображения с разными аугментациями
grid_imgs = [std_transform(original_img) for _ in range(8)]
grid = torchvision.utils.make_grid(torch.stack(grid_imgs), nrow=4, normalize=True)

plt.figure(figsize=(8, 6))
plt.title("8 версий с разными стандартными аугментациями")
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()