import os
from PIL import Image
from torchvision import transforms
import torch

class AugmentationPipeline:
    def __init__(self):
        self.augs = {}

    def add_augmentation(self, name: str, aug):
        self.augs[name] = aug

    def remove_augmentation(self, name: str):
        self.augs.pop(name, None)

    def apply(self, img: Image.Image) -> torch.Tensor:
        ts = transforms.Compose(list(self.augs.values()) + [transforms.ToTensor()])
        return ts(img)

    def get_augmentations(self):
        return list(self.augs.keys())

def process_profile(name, pipeline: AugmentationPipeline, input_dir='data/train', out_root='results'):
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    for cls in os.listdir(input_dir):
        cls_in = os.path.join(input_dir, cls)
        cls_out = os.path.join(out_dir, cls)
        os.makedirs(cls_out, exist_ok=True)

        for fname in os.listdir(cls_in):
            if not fname.lower().endswith(('.jpg', '.png', 'jpeg')):
                continue
            img = Image.open(os.path.join(cls_in, fname)).convert('RGB')
            out_t = pipeline.apply(img) 
            pil = transforms.ToPILImage()(out_t)
            pil.save(os.path.join(cls_out, fname))

if __name__ == '__main__':
    light = AugmentationPipeline()
    light.add_augmentation('flip', transforms.RandomHorizontalFlip(p=0.5))

    medium = AugmentationPipeline()
    medium.add_augmentation('flip', transforms.RandomHorizontalFlip(p=0.5))
    medium.add_augmentation('crop', transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
    medium.add_augmentation('jitter', transforms.ColorJitter(0.2,0.2,0.2,0.05))

    heavy = AugmentationPipeline()
    heavy.add_augmentation('flip', transforms.RandomHorizontalFlip(p=0.5))
    heavy.add_augmentation('crop', transforms.RandomResizedCrop(224, scale=(0.7, 1.0)))
    heavy.add_augmentation('jitter', transforms.ColorJitter(0.4,0.4,0.4,0.1))
    heavy.add_augmentation('rotate', transforms.RandomRotation(30))
    heavy.add_augmentation('gray', transforms.RandomGrayscale(p=0.2))

    for name, pipeline in [('light', light), ('medium', medium), ('heavy', heavy)]:
        print(f"Processing profile '{name}', augs:", pipeline.get_augmentations())
        process_profile(name, pipeline)
