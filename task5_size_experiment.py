import os
import time
from PIL import Image
import psutil
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def measure(size, dataset, num_samples=100):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((size, size))
    aug = transforms.Compose([resize, transforms.RandomHorizontalFlip(), to_tensor])

    proc = psutil.Process(os.getpid())

    mem_before = proc.memory_info().rss / 1024**2  
    t0 = time.time()

    for i in range(min(num_samples, len(dataset.images))):
        img_path = dataset.images[i]
        img = Image.open(img_path).convert('RGB')
        _ = aug(img)

    t1 = time.time()
    mem_after = proc.memory_info().rss / 1024**2

    return t1 - t0, mem_after - mem_before

def main():
    from datasets import CustomImageDataset
    dataset = CustomImageDataset('data/train', transform=None, target_size=(224,224))

    sizes = [64, 128, 224, 512]
    times, mems = [], []

    for s in sizes:
        print(f"Measuring size {s}x{s}...")
        t, m = measure(s, dataset, num_samples=100)
        print(f"  time: {t:.2f}s, mem Δ: {m:.1f} MB")
        times.append(t)
        mems.append(m)

    os.makedirs('results', exist_ok=True)
    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.title("Время обработки 100 изображений")
    plt.xlabel("Размер (px)")
    plt.ylabel("Время (сек)")
    plt.grid(True)
    plt.savefig('results/time_vs_size.png')
    plt.figure()
    plt.plot(sizes, mems, marker='o', color='orange')
    plt.title("Изменение памяти при обработке")
    plt.xlabel("Размер (px)")
    plt.ylabel("Δ Память (MB)")
    plt.grid(True)
    plt.savefig('results/memory_vs_size.png')

    print("\n Графики сохранены в `results/`")

if __name__ == "__main__":
    main()
