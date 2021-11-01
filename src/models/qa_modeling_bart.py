import collections
import json
import logging

import pytorch_lightning as pl
import torch.optim
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BartForQuestionAnswering

from src.data_proccess.post_proccess import generate_predictions
from src.metric import evaluate
from src.models.prefix_modeling_bart import BartModel
from src.models.prefix_tuning import PrefixTuning
from src.utils import freeze_params

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


class PrefixBartQAModel(pl.LightningModule):
    def __init__(self, model_name_or_path, cache_dir, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

        self.config = AutoConfig.from_pretrained(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir
        )

        self.seq2seq_model = BartModel.from_pretrained(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir
        )

        self.prefix_model = PrefixTuning(self.config)

        freeze_params(self.seq2seq_model)
        print('Freezing entire seq2seq model')

    def forward(self, **inputs):
        pad_token_id = self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    # def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
    #
    # def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:


class BartQAModel(pl.LightningModule):
    def __init__(self, model_name_or_path, **kwargs):
        super(BartQAModel, self).__init__()
        self.model = BartForQuestionAnswering.from_pretrained(model_name_or_path)

    def forward(self, x):
        input_ids, attention_mask, start_positions, end_positions = x

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch

        outputs = self((input_ids, input_mask, start_positions, end_positions))
        loss = outputs[0]

        return {'loss': loss}

    def validation_step(self, batch):
        input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices = batch

        outputs = self((input_ids, input_mask, start_positions, end_positions))

        loss, start_logits, end_logits = outputs[:3]

        return {
            'loss': loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "example_indices": example_indices
        }

    def validation_epoch_end(self, outputs):
        all_results = []

        for output in tqdm(outputs, desc="Evaluating"):
            batch_start_logits = output['start_logits']
            batch_end_logits = output['end_logits']
            example_indices = output['example_indices']

            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = self.trainer.datamodule.val_features[example_index.item()]
                unique_id = int(eval_feature['unique_id'])
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        predictions = generate_predictions(
            self.trainer.datamodule.val_examples,
            self.trainer.datamodule.val_features,
            all_results,
            n_best_size=20,
            max_answer_length=512,
            do_lower_case=True
        )

        temp_result = self.get_eval(self.trainer.datamodule.val_file, predictions)
        print("f1 score", temp_result['F1'])
        print("EM", temp_result["EM"])

        self.log("f1", temp_result['F1'], prog_bar=True)

    def test_step(self, batch):
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch

        outputs = self((input_ids, input_mask, start_positions, end_positions))
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_eval(self, original_file, predictions):
        ground_truth_file = json.load(open(original_file, 'r'))
        F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, predictions)
        AVG = (EM + F1) * 0.5
        output_result = collections.OrderedDict()
        output_result['AVERAGE'] = '%.3f' % AVG
        output_result['F1'] = '%.3f' % F1
        output_result['EM'] = '%.3f' % EM
        output_result['TOTAL'] = TOTAL
        output_result['SKIP'] = SKIP

        return output_result
