import torch
import torch.optim as optim
import neptune
import pandas as pd

from torch.utils.data import DataLoader
from torch import nn
from dataset import RanzcrDataset, split_dataset
from preprocess import transforms_train, transforms_test
from model import EffnetModel
from torchinfo import summary
from trainval import train, validate
from sklearn.model_selection import train_test_split

#Init Neptune
# neptune.init(project_qualified_name='simonvc/dacon-mnist',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjgwYjQ2NWYtMmY0MC00YzNjLWI1OGUtZWU4MDMzNDA2MWNhIn0=',
#              )
             
# neptune.init(project_qualified_name='dongkyuk/dacon-mnist',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTlmOGExYWUtNDRlOS00MTk1LThiOTQtOGY4MDkyZDAxZjY2In0=',
#              )

# neptune.init(project_qualified_name='dhdroid/Dacon-MNIST',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWM3ZDFmYjAtM2FlNS00YzUzLThjYTgtZjU3ZmM1MzJhOWQ4In0=',
#              )

# neptune.create_experiment()

# cuda cache 초기화
torch.cuda.empty_cache()

def model_train(fold: int) -> None:
    # Prepare Data
    df = pd.read_csv('data/split_kfold.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_val = df[df['kfold'] == fold].reset_index(drop=True)

    df_train.drop(['kfold'], axis=1).to_csv(f'data/train-kfold-{fold}.csv', index=False)
    df_val.drop(['kfold'], axis=1).to_csv(f'data/val-kfold-{fold}.csv', index=False)

    train_dataset = RanzcrDataset('data/train', f'data/train-kfold-{fold}.csv', transforms_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = RanzcrDataset('data/train', f'data/val-kfold-{fold}.csv', transforms_test)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Prepare Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EffnetModel().to(device)

    # Optimizer, loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MultiLabelSoftMarginLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    # Train
    num_epochs = 40
    best_loss = 1
    for epoch in range(num_epochs):
        # Train
        train(train_loader, model, optimizer, criterion, epoch, device)
        
        # Update learning rate
        scheduler.step()

        # Validate
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, device)

        # Save recent
        torch.save(model.state_dict(), f'checkpoints/fold-{fold}_epoch-{epoch}.pth')

        # Save best
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if is_best:
            torch.save(model.state_dict(), f'data/fold-{fold}_best.pth')

if __name__ == '__main__':
    split_dataset('data/train.csv')

    model_train(0)
    model_train(1)
    model_train(2)
    model_train(3)
    model_train(4)