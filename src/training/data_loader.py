from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset

from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size, eval_batch_size,
                 dataset_name_and_path):
        super(DataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_name_and_path = dataset_name_and_path

    def setup(self, stage: str):
        self.dataset = load_dataset(self.dataset_name_and_path)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)
