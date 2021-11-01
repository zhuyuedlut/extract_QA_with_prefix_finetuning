import torch
import torch.nn as nn

from transformers import BartPretrainedModel, BartConfig


class PrefixTuning(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super(PrefixTuning, self).__init__(config)
        self.preseqlen = config.preseqlen
        self.mid_dim = config.mid_dim

        self.n_embd = config.d_model
        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.match_n_embd = self.n_embd // self.match_n_head

        self.input_tokens = torch.arange(self.preseqlen).long()
        # prefix for attention compute
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.prefix_attn = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        # prefix for encoder compute
        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.prefix_encoder = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        # prefix for decoder compute
        self.wte_decoder = nn.Embedding(self.preseqlen, self.n_embd)
        self.prefix_decoder = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )
