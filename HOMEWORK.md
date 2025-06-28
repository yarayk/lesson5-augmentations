# Домашнее задание к уроку 5: Аугментации и работа с изображениями

## Датасет

Используйте датасет с изображениями героев, структура папок:
```
data/
├── train/
│   ├── hero1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── hero2/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Для загрузки используйте класс `CustomImageDataset` из `datasets.py`.

---

## Задание 1: Стандартные аугментации torchvision (15 баллов)

1. Создайте пайплайн стандартных аугментаций torchvision (например, RandomHorizontalFlip, RandomCrop, ColorJitter, RandomRotation, RandomGrayscale).
2. Примените аугментации к 5 изображениям из разных классов (папка train).
3. Визуализируйте:
   - Оригинал
   - Результат применения каждой аугментации отдельно
   - Результат применения всех аугментаций вместе

---

## Задание 2: Кастомные аугментации (20 баллов)

1. Реализуйте минимум 3 кастомные аугментации (например, случайное размытие, случайная перспектива, случайная яркость/контрастность).
2. Примените их к изображениям из train.
3. Сравните визуально с готовыми аугментациями из extra_augs.py.

---

## Задание 3: Анализ датасета (10 баллов)

1. Подсчитайте количество изображений в каждом классе.
2. Найдите минимальный, максимальный и средний размеры изображений.
3. Визуализируйте распределение размеров и гистограмму по классам.

---

## Задание 4: Pipeline аугментаций (20 баллов)

1. Реализуйте класс AugmentationPipeline с методами:
   - add_augmentation(name, aug)
   - remove_augmentation(name)
   - apply(image)
   - get_augmentations()
2. Создайте несколько конфигураций (light, medium, heavy).
3. Примените каждую конфигурацию к train и сохраните результаты.

---

## Задание 5: Эксперимент с размерами (10 баллов)

1. Проведите эксперимент с разными размерами изображений (например, 64x64, 128x128, 224x224, 512x512).
2. Для каждого размера измерьте время загрузки и применения аугментаций к 100 изображениям, а также потребление памяти.
3. Постройте графики зависимости времени и памяти от размера.

---

## Задание 6: Дообучение предобученных моделей (25 баллов)

1. Возьмите одну из предобученных моделей torchvision (например, resnet18, efficientnet_b0, mobilenet_v3_small).
2. Замените последний слой на количество классов вашего датасета.
3. Дообучите модель на train, проверьте качество на val.
4. Визуализируйте процесс обучения (loss/accuracy).

**Пример кода для ResNet18:**
```python
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset

# Подготовка датасета
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = CustomImageDataset('data/train', transform=transform)
val_dataset = CustomImageDataset('data/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Загрузка предобученной модели
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.get_class_names()))

# Обучение (пример)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(3):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} done!')
```

---

## Требования к коду
- Используйте только папку train для обучения (val/test — только для проверки качества).
- Код должен быть хорошо документирован.
- Все результаты и графики сохраняйте в папку results/.

## Срок сдачи
До начала занятия 7.

Удачи в выполнении задания! 🚀 