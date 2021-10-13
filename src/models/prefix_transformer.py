import pytorch_lightning as pl

from transformers import AutoConfig, AutoTokenizer

from model_bart import BartModel


class PrefixTransformer(pl.LightningModule):
    def __init__(self,
                 model_name_or_path,
                 num_labels
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = BartModel.from_pretrained(model_name_or_path, config=self.config)

    def forward(self, **inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

