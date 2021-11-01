import json

import torch

from tqdm import tqdm

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, AutoConfig

from src.data_proccess.pre_proccess import generate_examples_features


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path,
                 train_file,
                 val_file,
                 test_file,
                 train_batch_size,
                 eval_batch_size,
                 **kwargs
                 ):
        super(DataModule, self).__init__()

        self.model_name_or_path = model_name_or_path

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_examples, self.train_features = generate_examples_features(
                self.train_file,
                self.tokenizer,
                is_training=True,
                max_seq_length=self.config.max_position_embeddings
            )
            self.val_examples, self.val_features = generate_examples_features(
                self.val_file,
                self.tokenizer,
                is_training=False,
                max_seq_length=self.config.max_position_embeddings
            )

        if stage == 'test' or stage is None:
            self.test_examples, self.test_features = generate_examples_features(
                self.test_file,
                self.tokenizer,
                is_training=False,
                max_seq_length=self.config.max_position_embeddings
            )

    def train_dataloader(self):
        train_data = self.convert_to_input(self.train_features, is_training=True)
        return DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        val_data = self.convert_to_input(self.val_features)
        return DataLoader(val_data, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        test_data = self.convert_to_input(self.test_features)
        return DataLoader(test_data, batch_size=self.eval_batch_size)

    def convert_to_input(self, features):
        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        seq_len = all_input_ids.shape[1]
        assert seq_len <= self.config.max_position_embeddings

        all_start_positions = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_position'] for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_start_positions, all_end_positions, all_example_index)

        return data

    def generate_examples(self, data_file, is_training=False):
        with open(data_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            input_data = input_data['data']

        examples = []
        mis_match = []
        for article in tqdm(input_data):
            for para in article['paragraphs']:
                context = para['context']
                for qas in para['qas']:
                    qid = qas['id']
                    question_text = qas['question']
                    answer_text = qas['answer'][0]['text']

                    start_position = None
                    end_position = None

                    if is_training:
                        count = 0
                        search_limit = 3
                        start_position = qas['answer'][0]['answer_start']
                        end_position = start_position + len(answer_text) - 1

                        while context[start_position:end_position + 1] != answer_text and count < search_limit:
                            start_position -= 1
                            end_position -= 1
                            count += 1

                        while context[start_position] == " " or context[start_position] == "\t"\
                                or context[start_position] == "\r" or context[start_position] == "\n":
                            start_position += 1

                        if answer_text[start_position] in {"。", "，", "：", ":", ".", ","}:
                            start_position += 1

                        actual_text = "".join(answer_text[start_position: (end_position + 1)])

                        if actual_text != answer_text:
                            print(actual_text, 'V.S', answer_text)
                            mis_match += 1

                    examples.append({
                        'orig_answer_text': answer_text,
                        'qid': qid,
                        'question': question_text,
                        'answer': answer_text,
                        'start_position': start_position,
                        'end_position': end_position
                    })

    def generate_features(self, examples, is_training=False):
        for (example_index, example) in enumerate(tqdm(examples)):
            features = self.tokenizer.tokenize(
                example['question'],
                example['context'],
                padding='max_length',
                truncation='only_second',
                max_length = self.config.max_position_embeddings
            )

            start_position = None
            end_position = None

            if is_training:
                start_position = example['start_position']
                end_position = example['end_position']
                if start_position == 0:
                    start_position += 1
                if end_position >= self.config.max_position_embeddings:
                    end_position = self.config.max_position_embeddings - 1

            features.update({
                'example_index': example_index,
                'start_position': start_position,
                'end_position': end_position,
            })