import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Optional
from unlabeled_chess_dataset import UnlabeledChessDataset
from dataset import ChessRecognitionDataset

class ChessDataModule(pl.LightningDataModule):
    def __init__(self, dataroot: str, batch_size: int, workers: int, unlabeled_image_dir: Optional[str] = None) -> None:
        super().__init__()
        self.dataroot: str = dataroot
        self.unlabeled_image_dir: Optional[str] = unlabeled_image_dir
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])
        self.batch_size: int = batch_size
        self.workers: int = workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.chess_train = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="train", transform=self.transform)
            self.chess_val = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="val", transform=self.transform)
        if stage == "test" or stage == "predict":
            if self.unlabeled_image_dir:
                self.chess_test = UnlabeledChessDataset(
                    image_dir=self.unlabeled_image_dir,
                    transform=self.transform)
            else:
                self.chess_test = ChessRecognitionDataset(
                    dataroot=self.dataroot,
                    split="test", transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_train, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_val, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_test, batch_size=self.batch_size,
            num_workers=self.workers)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()