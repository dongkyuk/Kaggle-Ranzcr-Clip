import os
from typing import Tuple, Sequence, Callable
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd
from sklearn.model_selection import GroupKFold
from PIL import Image


class RanzcrDataset(Dataset):
    def __init__(self, dir: os.PathLike, image_ids: os.PathLike, transforms: Sequence[Callable]) -> None:
        self.dir = dir
        self.transforms = transforms
        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[row[0]] = list(map(int, row[1:-1]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.jpg')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = np.array(image)
            image = self.transforms(image=image)['image']

        return image, target

def split_dataset(path: os.PathLike, dest_path: os.PathLike, num_split:int=5) -> None:
    df = pd.read_csv(path)
    kfold = GroupKFold(n_splits=num_split)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index, df['PatientID'])):
        df.loc[valid, 'kfold'] = int(fold)

    df.to_csv(os.path.join(dest_path, 'split_kfold.csv'), index=False)