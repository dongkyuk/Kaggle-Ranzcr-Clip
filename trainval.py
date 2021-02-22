import torch
from torch.utils.data import DataLoader
from torch import nn
from preprocess import transforms_train, transforms_test
import numpy as np
import neptune

def train(train_loader, model, optimizer, criterion, epoch, device):
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Log and save
        if (i+1) % 10 == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            neptune.log_metric('train loss', loss.item())
            neptune.log_metric('train accuracy', acc.item())
            print(f'Epoch {epoch} / Step: {i+1}: Train loss {loss.item():.5f}, Train Accuracy {acc.item():.5f}')


def validate(test_loader, model, criterion, epoch, device):
    model.eval()  # prep model for *evaluation*
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            outputs = outputs > 0.5
            val_acc += (outputs == targets).float().mean()*images.size(0)
            val_loss += loss.item()*images.size(0)

    val_loss = val_loss/len(test_loader.dataset)
    val_acc = val_acc/len(test_loader.dataset)
    
    print(f'Epoch {epoch}: Val loss {val_loss:.5f}, Val Accuracy {val_acc:.5f}')
    neptune.log_metric('validation loss', val_loss)
    neptune.log_metric('validation accuracy', val_acc)
    
    return val_loss, val_acc
