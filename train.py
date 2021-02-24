import torch
import neptune
import pandas as pd
import random
import numpy as np
import os

from torch.utils.data import DataLoader
from torch import nn

from dataset import RanzcrDataset, split_dataset
from preprocess import transforms_train, transforms_test
from model import RanzcrModel, EffnetModel, ResNet200DModel
from torchinfo import summary
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# Init Neptune
# neptune.init(project_qualified_name='simonvc/dacon-mnist',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjgwYjQ2NWYtMmY0MC00YzNjLWI1OGUtZWU4MDMzNDA2MWNhIn0=',
#              )

neptune.init(project_qualified_name='dongkyuk/Ranzcr',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTlmOGExYWUtNDRlOS00MTk1LThiOTQtOGY4MDkyZDAxZjY2In0=',
             )

# neptune.init(project_qualified_name='dhdroid/Dacon-MNIST',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWM3ZDFmYjAtM2FlNS00YzUzLThjYTgtZjU3ZmM1MzJhOWQ4In0=',
#              )

neptune.create_experiment()

# cuda cache 초기화
torch.cuda.empty_cache()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def model_train(fold: int) -> None:
    # Prepare Data
    df = pd.read_csv(os.path.join(config.save_dir, 'split_kfold.csv'))
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_val = df[df['kfold'] == fold].reset_index(drop=True)

    df_train.drop(['kfold'], axis=1).to_csv(os.path.join(
        config.save_dir, f'train-kfold-{fold}.csv'), index=False)
    df_val.drop(['kfold'], axis=1).to_csv(os.path.join(
        config.save_dir, f'val-kfold-{fold}.csv'), index=False)

    train_dataset = RanzcrDataset(os.path.join(config.data_dir, 'train'), os.path.join(
        config.save_dir, f'train-kfold-{fold}.csv'), transforms_train)
    val_dataset = RanzcrDataset(
        os.path.join(config.data_dir, 'train'), os.path.join(config.save_dir, f'val-kfold-{fold}.csv'), transforms_test)

    model = RanzcrModel(ResNet200DModel())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(config.save_dir, f'{fold}'),
        filename='{epoch:02d}-{val_loss:.2f}.pth',
        save_top_k=5,
        mode='min',
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
    )

    if config.device == 'tpu':
        train_loader = DataLoader(
            train_dataset, batch_size=16, num_workers=10, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2,
                                num_workers=10, shuffle=False)
        trainer = Trainer(
            tpu_cores=8,
            num_sanity_val_steps=-1,
            deterministic=True,
            max_epochs=config.epochs,
            callbacks=[checkpoint_callback, early_stopping]
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=16, num_workers=10, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8,
                                num_workers=10, shuffle=False)
        trainer = Trainer(
            gpus=1,
            num_sanity_val_steps=-1,
            deterministic=True,
            max_epochs=config.epochs,
            callbacks=[checkpoint_callback, early_stopping]
        )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    class config:
        seed = 42
        epochs = 30
        data_dir = 'data/'
        save_dir = 'save/exp1'
        device = 'gpu'

    from pathlib import Path
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    split_dataset(os.path.join(config.data_dir, 'train.csv'), config.save_dir)

    model_train(0)
    model_train(1)
    model_train(2)
    model_train(3)
    model_train(4)
