import torch.nn as nn

from transformers import PretrainedBartModel


class BartModel(PretrainedBartModel):
    def __init__(self, config):
        super(BartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
