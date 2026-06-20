"""
dataset.py — PyTorch Dataset for deepfake detection.

Folder structure:
    root/
        real/  → label 0
        fake/  → label 1
"""

import os
from PIL import Image
from torch.utils.data import Dataset


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class DeepfakeDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples = []   # list of (path, label)

        for label_name, label_idx in [('real', 0), ('fake', 1)]:
            folder = os.path.join(root, label_name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Expected folder: {folder}")
            for fname in os.listdir(folder):
                if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                    self.samples.append((os.path.join(folder, fname), label_idx))

        print(f"[Dataset] {root}: {len(self.samples)} images "
              f"({sum(1 for _, l in self.samples if l == 0)} real, "
              f"{sum(1 for _, l in self.samples if l == 1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
