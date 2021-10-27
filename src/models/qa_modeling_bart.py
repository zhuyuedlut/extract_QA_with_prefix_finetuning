import logging

import pytorch_lightning as pl
import torch.optim
from transformers import AutoConfig, AutoTokenizer, BartForQuestionAnswering

from src.models.prefix_modeling_bart import BartModel
from src.models.prefix_tuning import PrefixTuning
from src.utils import freeze_params

logger = logging.getLogger(__name__)


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
    def __init__(self, model_name_or_path, cache_dir):
        super(BartQAModel, self).__init__()
        self.model = BartForQuestionAnswering.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        return { 'loss': loss, 'log': { 'train_loss': loss }}

    def validation_step(self, batch):
        outputs = self(batch)
        loss = outputs[0]
        return {'loss': loss, 'log': {'val_loss': loss}}

    def test_step(self, batch):
        outputs = self(batch)
        loss = outputs[0]
        return { 'loss': loss, 'log': {'test_loss': loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


