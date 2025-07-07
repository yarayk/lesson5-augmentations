import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    root = 'data/train'
    class_counts = {}
    widths = []
    heights = []

    for class_name in os.listdir(root):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        class_counts[class_name] = len(files)
        for fname in files:
            path = os.path.join(class_dir, fname)
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

    print("Количество изображений по классам:")
    for cls, cnt in class_counts.items():
        print(f"{cls}: {cnt}")

    widths_arr = np.array(widths)
    heights_arr = np.array(heights)
    print("\n Размеры изображений (ширина x высота):")
    print(f"Ширина — min: {widths_arr.min()}, max: {widths_arr.max()}, avg: {widths_arr.mean():.1f}")
    print(f"Высота — min: {heights_arr.min()}, max: {heights_arr.max()}, avg: {heights_arr.mean():.1f}")

    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.hist(widths_arr, bins=20, alpha=0.6, label='width')
    plt.hist(heights_arr, bins=20, alpha=0.6, label='height')
    plt.title("Распределение размеров изображений")
    plt.xlabel("Pixels")
    plt.ylabel("Количество")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/size_distribution.png')
    plt.close()

    plt.figure(figsize=(8,5))
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, counts, color='skyblue')
    plt.yticks(y_pos, classes)
    plt.title("Количество изображений по классам")
    plt.xlabel("Количество")
    plt.tight_layout()
    plt.savefig('results/counts_per_class.png')
    plt.close()

    print("\n Графики сохранены в папке `results/`")

if __name__ == "__main__":
    main()
