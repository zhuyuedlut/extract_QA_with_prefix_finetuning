import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path,
                 dataset_name_or_path,
                 train_batch_size,
                 eval_batch_size,
                 ):
        super(DataModule, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path,
                                                       use_fast=True)

    def prepare_data(self) -> None:
        self.dataset = load_dataset(self.dataset_name_or_path)

    def setup(self):

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_feature,
                batched=True,
                batch_size=self.train_batch_size if split == 'train' else self.eval_batch_size,
                remove_columns=self.dataset[split].column_names
            )

    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'],
                          batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)

    def convert_to_feature(self, examples):
        features = self.tokenizer(examples['question'], examples['context'],
                                  padding='longest')
        start_positions, end_positions = [], []

        for i, (context, answer) in enumerate(
                zip(examples['context'], examples['answers'])):
            start_idx, end_idx = self.get_correct_alignment(context, answer)
            start_positions.append(start_idx)
            end_positions.append(end_idx)

        features.update({'start_positions': start_positions,
                         'enc_positions': end_positions})
        return features

    @staticmethod
    def get_correct_alignment(context, answer):
        target_text = answer['text'][0]
        start_idx = answer['answer_start'][0]
        while context[start_idx] == ' ' or context[start_idx] == '\t' or \
                context[start_idx] == '\r' or context[start_idx] == '\n':
            start_idx += 1
        end_idx = start_idx + len(target_text)
        if context[start_idx: end_idx] == target_text:
            return start_idx, end_idx
        elif context[start_idx - 1: end_idx - 1] == target_text:
            return start_idx - 1, end_idx - 1
        elif context[start_idx - 2: end_idx - 2] == target_text:
            return start_idx - 2, end_idx - 2
        else:
            raise ValueError()