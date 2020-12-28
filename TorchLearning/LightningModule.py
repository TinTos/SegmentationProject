from pytorch_lightning import LightningModule
from torchvision import models, transforms
import torch.nn as nn
import torch
from TorchLearning.PretrainedModel import preprocess

class LitModel(LightningModule):

    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.classifier = models.resnet18(pretrained=True)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        return optimizer

    def training_step(self, batch, batch_idx, isRGB = False):
        x, y = batch
        x = preprocess(x, isRGB)
        outputs = self(x)
        loss = nn.BCEWithLogitsLoss(outputs, y)

        return loss

    def validation_step(self, batch, batch_idx, isRGB = False):
        x, y = batch
        x = preprocess(x, isRGB)
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        _, comparisonlabel = torch.max(y, 1)
        val_loss = torch.sum(preds == comparisonlabel) / batch.shape[0]

        return val_loss


    def test_step(self, batch, batch_idx, isRGB = False):
        x, y = batch
        x = preprocess(x, isRGB)
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        _, comparisonlabel = torch.max(y, 1)
        val_loss = torch.sum(preds == comparisonlabel) / batch.shape[0]

        return val_loss

