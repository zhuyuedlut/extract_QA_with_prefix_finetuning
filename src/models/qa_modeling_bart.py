import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer

from src.models.prefix_modeling_bart import BartModel
from src.models.prefix_tuning import PrefixTuning
from src.utils import freeze_params

logger = logging.getLogger(__name__)


class BartForQuestionAnswering(pl.LightningModule):
    def __init__(self,
                 model_name_or_path,
                 cache_dir,
                 ):
        super().__init__()

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

    @staticmethod
    def add_model_specific_args(parent_parse):
        parser = ArgumentParser(parents=[parent_parse])
        parser.add_argument("--pre_len", default=200, type=int, help="The prefix embedding length")

        return parser
