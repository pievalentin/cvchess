import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from typing import Tuple, List
from utils import recognition_accuracy

class ChessResNeXt(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, decay: int = 100) -> None:
        super().__init__()

        self.lr: float = lr
        self.decay: int = decay

        backbone = models.resnext101_32x8d(weights="DEFAULT")
        num_filters: int = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor: nn.Sequential = nn.Sequential(*layers)

        num_target_classes: int = 64 * 13

        self.classifier: nn.Linear = nn.Linear(num_filters, num_target_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x).flatten(1)
        x = self.classifier(x)
        return x

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels)

    def common_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, return_accuracy: bool = False) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)

        if return_accuracy:
            y_cat = torch.argmax(y.reshape((-1, 64, 13)), dim=2)
            preds_cat = torch.argmax(logits.reshape((-1, 64, 13)), dim=2)

            return (self.cross_entropy_loss(logits, y),
                    recognition_accuracy(y_cat, preds_cat))

        return self.cross_entropy_loss(logits, y)

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.common_step(train_batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, accuracy = self.common_step(val_batch, batch_idx, return_accuracy=True)
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

    def test_step(self, test_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss = self.common_step(test_batch, batch_idx)
        self.log('test_loss', loss)

    def predict_step(self, predict_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = predict_batch
        logits = self.forward(x)
        return (logits, y)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.StepLR]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.decay)
        return [optimizer], [scheduler]