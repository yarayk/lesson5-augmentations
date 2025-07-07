import random
import torch
from torchvision import transforms
from datasets import CustomImageDataset
from utils import show_single_augmentation, show_images
from extra_augs import AddGaussianNoise, CutOut 

class RandomGaussianBlur:
    def __init__(self, kernel_size=(5,5), sigma=(0.1,2.0), p=0.5):
        self.p = p
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    def __call__(self, img):
        return self.blur(img) if random.random() < self.p else img

class RandomPerspectiveTransform:
    def __init__(self, distortion_scale=0.5, p=0.5):
        self.p = p
        self.persp = transforms.RandomPerspective(distortion_scale=distortion_scale, p=1.0)
    def __call__(self, img):
        return self.persp(img) if random.random() < self.p else img

class RandomBrightnessContrast:
    def __init__(self, brightness=0.5, contrast=0.5, p=0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
    def __call__(self, img):
        return self.jitter(img) if random.random() < self.p else img

def main():
    dataset = CustomImageDataset('data/train', transform=None, target_size=(224,224))
    samples = []
    used = set()
    for img, lbl in dataset:
        if lbl not in used:
            samples.append((img, lbl))
            used.add(lbl)
        if len(samples) == 5: break

    custom_augs = {
        "RandomGaussianBlur": RandomGaussianBlur(p=1.0),
        "RandomPerspective": RandomPerspectiveTransform(p=1.0),
        "RandomBrightnessContrast": RandomBrightnessContrast(p=1.0)
    }

    from extra_augs import AddGaussianNoise, CutOut
    ready_augs = {
        "AddGaussianNoise": AddGaussianNoise(mean=0., std=0.1),
        "CutOut": CutOut(p=1.0, size=(50,50))
    }

    to_tensor = transforms.ToTensor()
    for img, lbl in samples:
        orig = to_tensor(img)
        show_single_augmentation(orig, orig, "Оригинал")

        for name, aug in custom_augs.items():
            aug_img = transforms.ToTensor()(aug(img))
            show_single_augmentation(orig, aug_img, name)

        for name, aug in ready_augs.items():
            aug_img = aug(orig.clone())
            show_single_augmentation(orig, aug_img, name)

    for name, aug in custom_augs.items():
        imgs = [transforms.ToTensor()(aug(img)) for img, _ in samples]
        show_images(imgs, title=name + " – 5 примеров")

if __name__ == "__main__":
    main()
