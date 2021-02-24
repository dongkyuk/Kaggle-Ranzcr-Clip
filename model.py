import torch
import torch.nn as nn
import timm
import neptune
import numpy as np
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from pytorch_lightning.metrics import AUROC


class RanzcrModel(LightningModule):
    def __init__(self, model: nn.Module = None):
        super().__init__()
        self.model = model
        self.criterion = nn.MultiLabelSoftMarginLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        neptune.log_metric('train loss', loss)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        return {'loss': loss, 'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        try:
            auc = self.get_mean_rocauc(y_hat, y)
        except:
            auc = 0

        self.log('test loss', avg_loss, prog_bar=True)
        self.log('test auc', auc, prog_bar=True)

        neptune.log_metric('test loss', avg_loss)
        neptune.log_metric('test auc', auc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0)

        return [optimizer], [scheduler]

    def get_mean_rocauc(self, y_hat, y):
        scores = []
        y_hat, y = y_hat.cpu().detach(), y.cpu().detach()
        for i in range(y.shape[0]):
            score = roc_auc_score(y[i, :].int(), torch.sigmoid(y_hat[i, :]))
            scores.append(score)
        avg_score = np.mean(scores)
        return avg_score


class EffnetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.classifier = nn.Linear(1000, 11)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.classifier(x)

        return x


class ResNet200DModel(nn.Module):
    def __init__(self):
        super(ResNet200DModel, self).__init__()
        self.resnet200d = timm.create_model('resnet200d', pretrained=True)
        n_features = self.resnet200d.fc.in_features
        self.resnet200d.global_pool = nn.Identity()
        self.resnet200d.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11)

    def forward(self, x):
        bs = x.size(0)
        features = self.resnet200d(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output
