import logging
import os
from argparse import ArgumentParser

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from model_prefix_bart import BartModel

logger = logging.getLogger(__name__)


class PrefixBart(pl.LightningModule):
    def __init__(self,
                 model_name_or_path,
                 config=None,
                 **config_kwargs,
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name is not None else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config = config

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.tokenizer is not None else self.hparams.model_name_or_path,
                cache_dir=cache_dir
            )




    def forward(self, **inputs):
        return self.model(inputs)

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
        parser.add_argument("--pre_len", default=200, type=int,
                            help="The prefix embedings length")
        return parser
