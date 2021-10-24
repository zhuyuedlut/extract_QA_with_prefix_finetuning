import torch
import torch.nn as nn

from transformers import PretrainedBartModel


class PrefixTuning(PretrainedBartModel):
    def __init__(self, config):
        super(PrefixTuning, self).__init__(config)
        self.preseqlen = config.preseqlen
        self.mid_dim = config.mid_dim
        self.n_embd = config.d_model

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.prefix_attn = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
        )