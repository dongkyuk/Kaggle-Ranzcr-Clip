import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
import numpy as np

def validate(test_loader, model, criterion, epoch, device):
    model.eval() # prep model for *evaluation*
    val_loss = 0 
    val_acc = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            output_prob, output_cnt = model(images)
            loss = criterion(output_prob, targets[:,:-1]) + 0.1 * nn.L1Loss()(torch.sum(output_prob, 1).squeeze(), targets[:,-1:])
            # print("output1", outputs)
            outputs = output_prob > 0.5
            # print("output2", outputs)
            val_acc += (outputs == targets[:,:-1]).float().mean()*images.size(0)
            # print("val acc", val_acc)
            val_loss += loss.item()*images.size(0)
            # print("val loss", val_loss)
    
    val_loss = val_loss/len(test_loader.dataset)
    val_acc = val_acc/len(test_loader.dataset)

    model.train()

    return val_loss, val_acc
