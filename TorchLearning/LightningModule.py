from pytorch_lightning import LightningModule
from torchvision import models, transforms
import torch.nn as nn
import torch
from TorchLearning.PretrainedModel import preprocess

class LitModel(LightningModule):

    def __init__(self, num_classes, learning_rate, training_size, batch_size,model=None):
        super().__init__()
        if model is 'renset34':
            self.classifier = models.resnet34(pretrained=True)
        else:
            self.classifier = models.resnet18(pretrained=True)

        self.classifier = nn.Sequential(self.classifier, nn.Linear(1000, num_classes))
        #self.classifier.load_state_dict(torch.load('LastModel'))
        #num_ftrs = self.classifier.fc.in_features
        #self.classifier.fc = nn.Linear(num_ftrs, num_classes)
        self.learning_rate = learning_rate
        self.lossobject = nn.BCEWithLogitsLoss()
        self.stepsize = 4*int(training_size // batch_size)

    def forward(self, x):
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.learning_rate)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.learning_rate, total_steps = 2 * self.stepsize),
                'interval': 'step',
                'frequency': 1
        }


        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx, isRGB = False):
        x, y = batch
        x = preprocess(x, isRGB)
        outputs = self(x)
        loss = self.lossobject(outputs, y)

        return loss

    def validation_step(self, batch, batch_idx, isRGB = False):
        x, y = batch
        x = preprocess(x, isRGB)
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        _, comparisonlabel = torch.max(y, 1)
        val_loss = torch.sum(preds == comparisonlabel) / len(batch)

        return val_loss
        #loss = nn.BCEWithLogitsLoss()(outputs, y)

        #return loss


    def test_step(self, batch, batch_idx, isRGB = False):
        x, y = batch
        x = preprocess(x, isRGB)
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        _, comparisonlabel = torch.max(y, 1)
        val_loss = torch.sum(preds == comparisonlabel) / batch.shape[0]

        return val_loss

