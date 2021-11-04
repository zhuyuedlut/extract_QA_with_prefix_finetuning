import collections
import os
import json

import pytorch_lightning as pl
import torch.optim
from tqdm import tqdm
from transformers import BartForQuestionAnswering

from src.metric import Metric

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


class BartQAModel(pl.LightningModule):
    def __init__(self, args):
        super(BartQAModel, self).__init__()
        self.learning_rate = args.learning_rate
        self.input_dir = args.data_dir
        self.output_dir = args.output_dir

        self.model = BartForQuestionAnswering.from_pretrained(args.model_name_or_path)
        self.metric = Metric(args)

    def forward(self, x):
        input_ids, attention_mask, start_positions, end_positions = x

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

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
        print(len(predictions))

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
        values = {'F1': float(result['F1']), 'EM': float(result["EM"]), 'TOTAL': int(result['TOTAL']), 'SKIP': int(result["SKIP"]) }
        self.log_dict(values)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
