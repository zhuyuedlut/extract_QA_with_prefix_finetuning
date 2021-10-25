import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path,
                 dataset_name_and_path,
                 train_batch_size,
                 eval_batch_size,
                 ):
        super(DataModule, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.dataset_name_and_path = dataset_name_and_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset = None

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self):
        # reference https://huggingface.co/transformers/preprocessing.html
        # reference https://huggingface.co/docs/datasets/share.html
        dataset = load_dataset(self.dataset_name_and_path).flatten()

        for split in dataset.keys():
            self.dataset[split] = dataset[split].map(
                self.convert_to_feature,
                batched=True,
                remove_columns=dataset[split].column_names
            )

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)

    def convert_to_feature(self, examples):
        features = self.tokenizer(examples['question'], examples['context'], padding='longest')
        features['decoder_ids'] = self.tokenizer(examples['answers.text'], padding='longest')

        return features