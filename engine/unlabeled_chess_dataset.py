from torch.utils.data import Dataset
from PIL import Image
import os
from typing import List, Optional
import torch

class UnlabeledChessDataset(Dataset):
    def __init__(self, image_dir: str, transform: Optional[callable] = None):
        self.image_dir: str = image_dir
        self.transform: Optional[callable] = transform
        self.image_files: List[str] = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name: str = os.path.join(self.image_dir, self.image_files[idx])
        image: Image.Image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image