import random
import torch
from torchvision import transforms
from datasets import CustomImageDataset
from utils import show_single_augmentation, show_images

def main():
    dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))
    class_names = dataset.get_class_names()

    samples = []
    used = set()
    for img, label in dataset:
        if label not in used:
            samples.append((img, label))
            used.add(label)
        if len(samples) == 5:
            break

    std_augs = {
        "HorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
        "RandomCrop": transforms.RandomCrop(200, padding=20),
        "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        "RandomRotation": transforms.RandomRotation(degrees=30),
        "RandomGrayscale": transforms.RandomGrayscale(p=1.0),
    }

    combined = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(200, padding=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(20),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()
    ])

    to_tensor = transforms.ToTensor()

    for img, label in samples:
        orig_t = to_tensor(img)
        show_single_augmentation(orig_t, orig_t, f"Оригинал ({class_names[label]})")

        for name, aug in std_augs.items():
            pipeline = transforms.Compose([aug, transforms.ToTensor()])
            aug_t = pipeline(img)
            show_single_augmentation(orig_t, aug_t, name)

        comb_t = combined(img)
        show_single_augmentation(orig_t, comb_t, "Комбинированно")

    combs = [combined(img) for img, _ in samples]
    show_images(combs, title="Комбинированные 5 примеров")

if __name__ == "__main__":
    main()
