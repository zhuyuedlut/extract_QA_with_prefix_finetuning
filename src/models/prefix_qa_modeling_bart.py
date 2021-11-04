import collections
import os

import pytorch_lightning as pl
import torch
from tqdm import tqdm
from transformers import AutoConfig

from src.models.prefix_bart import BartModel
from src.models.prefix_tuning import PrefixTuning

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


class PrefixBartQAModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.learning_rate = args.learning_rate

        self.pre_len = args.pre_len

        self.config = AutoConfig.from_pretrained(model_name_or_path=args.model_name_or_path)

        self.seq2seq_model = BartModel.from_pretrained(model_name_or_path=args.model_name_or_path)

        self.freeze_params(self.seq2seq_model)
        print('Freezing entire seq2seq model')

        self.prefix_model = PrefixTuning(args, self.config)

    def forward(self, x):
        input_ids, attention_mask, start_positions, end_positions = x
        batch_size, _ = input_ids.size()

        input_tokens = torch.arange(self.pre_len).long().unsqueeze(0).expand(batch_size, -1).to(self.device)

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, start_positions, end_positions, example_indices = batch

        outputs = self((input_ids, input_mask, start_positions, end_positions))
        loss = outputs[0]

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, start_positions, end_positions, example_indices = batch

        outputs = self((input_ids, input_mask, None, None))

        return {
            "start_logits": outputs["start_logits"],
            "end_logits": outputs["end_logits"],
            "example_indices": example_indices
        }

    def validation_epoch_end(self, outputs):
        all_results = []
        all_examples = []
        all_features = []

        for output in tqdm(outputs, desc="Evaluating"):
            batch_start_logits = output['start_logits']
            batch_end_logits = output['end_logits']
            example_indices = output['example_indices']

            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_example = self.trainer.datamodule.val_examples[example_index.item()]
                all_examples.append(eval_example)
                eval_feature = self.trainer.datamodule.val_features[example_index.item()]
                all_features.append(eval_feature)
                unique_id = int(eval_feature['unique_id'])
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        predictions = self.metric.generate_predictions(
            all_examples,
            all_features,
            all_results,
            n_best_size=20,
            max_answer_length=512,
        )

        input_data_file = os.path.join(self.input_dir, "{}.json".format("val"))

        result = self.metric.get_eval(input_data_file, predictions)
        print("f1 score", result['F1'])
        print("EM", result["EM"])

        self.log("f1", float(result['F1']), prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, start_positions, end_positions, example_indices = batch

        outputs = self((input_ids, input_mask, None, None))

        return {
            "start_logits": outputs["start_logits"],
            "end_logits": outputs["end_logits"],
            "example_indices": example_indices
        }

    def test_epoch_end(self, outputs) -> None:
        all_results = []
        all_examples = []
        all_features = []

        for output in tqdm(outputs, desc="Evaluating"):
            batch_start_logits = output['start_logits']
            batch_end_logits = output['end_logits']
            example_indices = output['example_indices']

            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_example = self.trainer.datamodule.test_examples[
                    example_index.item()]
                all_examples.append(eval_example)
                eval_feature = self.trainer.datamodule.test_features[
                    example_index.item()]
                all_features.append(eval_feature)
                unique_id = int(eval_feature['unique_id'])
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        predictions = self.metric.generate_predictions(
            all_examples,
            all_features,
            all_results,
            n_best_size=20,
            max_answer_length=512,
        )

        input_data_file = os.path.join(self.input_dir, "{}.json".format("test"))

        result = self.metric.get_eval(input_data_file, predictions)
        print("**************Test**********************")
        print("f1 score", result['F1'])
        print("EM", result["EM"])
        values = {'F1': result['F1'], 'EM': result["EM"], 'TOTAL': result['TOTAL'], 'SKIP': result["SKIP"]}
        self.log_dict(values)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def freeze_params(model):
        for par in model.parameters():
            par.requires_grad = False
