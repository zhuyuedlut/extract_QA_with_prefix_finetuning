import collections
import json
import os
from tqdm import tqdm

import pytorch_lightning as pl
import torch

from transformers import BertTokenizer, AutoConfig
from torch.utils.data import DataLoader, TensorDataset

from src.utils import check_is_max_context


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.save_hyperparameters()

        self.model_name_or_path = args.model_name_or_path

        self.data_dir = args.data_dir

        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_examples = self.generate_examples(mode='train')
            self.train_dataset, self.train_features = self.generate_features(self.train_examples, mode='train')
            self.val_examples = self.generate_examples(mode='val')
            self.val_dataset, self.val_features = self.generate_features(self.val_examples, mode='val')

        if stage == 'test' or stage is None:
            self.test_examples = self.generate_examples(mode='test')
            self.test_dataset, self.test_features = self.generate_features(self.test_examples, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size)

    def generate_examples(self, mode):
        data_file = os.path.join(self.data_dir, "{}.json".format(mode))
        with open(data_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            input_data = input_data['data']

        examples = []
        mis_match = 0
        for article in tqdm(input_data):
            for para in article['paragraphs']:
                context = para['context']
                doc_tokens = self.tokenizer.tokenize(context)

                for qas in para['qas']:
                    qid = qas['id']
                    question_text = qas['question']
                    answer_text = qas['answers'][0]['text']

                    start_position = None
                    end_position = None

                    if mode == 'train':
                        answer_tokens = self.tokenizer.tokenize(answer_text)
                        start_token = answer_tokens[0]
                        if start_token in doc_tokens:
                            start_position = doc_tokens.index(start_token)
                            end_position = start_position + len(answer_tokens) - 1
                        else:
                            mis_match += 1
                            continue

                    examples.append({
                        'orig_answer_text': answer_text,
                        'qid': qid,
                        'doc_tokens': doc_tokens,
                        'question': question_text,
                        'context': context,
                        'answer': answer_text,
                        'start_position': start_position,
                        'end_position': end_position
                    })
        return examples

    def generate_features(self, examples, mode, max_question_length, doc_stride):
        unique_id = 1000000000
        cache_features_file = os.path.join(self.data_dir, 'cached_{}'.format(mode))

        if os.path.exists(cache_features_file):
            features = torch.load(cache_features_file)
        else:
            features = []
            for (example_index, example) in enumerate(examples):
                query_tokens = self.tokenizer.tokenize(example['question'])

                if len(query_tokens) > max_question_length:
                    query_tokens = query_tokens[0:max_question_length]

                # 通过index来访问token
                token_to_original_index = []
                original_to_token_index = []
                for (i, token) in enumerate(example['doc_tokens']):
                    original_to_token_index.append(i)
                    token_to_original_index.append(token)

                # context可以使用的最大的长度
                max_tokens_for_context = self.config.max_position_embeddings - len(query_tokens) - 3

                _Docspan = collections.namedtuple("DocSpan", ["start", "length"])

                doc_spans = []
                start_offset = 0
                while start_offset < len(example['doc_tokens']):
                    length = len(example['doc_tokens']) - start_offset
                    # context组成的token的长度同start_offset大于max_tokens_for_context
                    # 如果是循环的第一次执行则说明context组成的token的长度大于max_tokens_for_context
                    if length > max_tokens_for_context:
                        length = max_tokens_for_context
                    doc_spans.append(_Docspan(start=start_offset, length=length))
                    if start_offset + length == len(example['doc_tokens']):
                        break
                    start_offset += min(length, doc_stride)

                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    tokens = []
                    token_to_original_map = {}
                    token_is_max_context = {}
                    segment_ids = []
                    input_span_mask = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    input_span_mask.append(1)
                    for token in query_tokens:
                        tokens.append(token)
                        segment_ids.append(0)
                        input_span_mask.append(0)
                    tokens.append("[SEP]")
                    segment_ids.append(0)
                    input_span_mask.append(0)

                    # 从start到doc_span.length依次遍历
                    for i in range(doc_span.length):
                        split_token_index = doc_span.start + i
                        token_to_original_map[len(tokens)] = example['doc_tokens'][split_token_index]
                        is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
                        token_is_max_context[len(tokens)] = is_max_context
                        tokens.append(example['doc_tokens'][split_token_index])
                        segment_ids.append(1)
                        input_span_mask.append(1)

                    tokens.append("[SEP]")
                    segment_ids.append(1)
                    input_span_mask.append(0)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)

                    while len(input_ids) < max_question_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                        input_span_mask.append(0)

                    start_position = None
                    end_position = None

                    if mode == 'train':
                        doc_start = doc_span.start
                        doc_end = doc_span.start + doc_span.length - 1
                        out_of_span = False
                        if not (example['start_position'] >= doc_start and example['end_position'] <= doc_end):
                            out_of_span = True
                        if out_of_span:
                            start_position = 0
                            end_position = 0
                        else:
                            doc_offset = len(query_tokens) + 2
                            start_position = example['start_position'] - doc_start + doc_offset
                            end_position = example['end_position'] - doc_start + doc_offset

                        feature = {

                        }
                features.append(feature)
                unique_id += 1

        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        all_example_index = torch.tensor([f['example_index'] for f in features], dtype=torch.int)

        seq_len = all_input_ids.shape[1]
        assert seq_len <= self.config.max_position_embeddings

        if mode == 'train':
            all_start_positions = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f['end_position'] for f in features], dtype=torch.long)
        else:
            all_start_positions = torch.zeros(all_input_ids.size(0), dtype=torch.int)
            all_end_positions = torch.zeros(all_input_ids.size(0), dtype=torch.int)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_start_positions, all_end_positions,
                                all_example_index)

        return dataset, features
