import torch
import torch.nn as nn

from transformers import BartConfig


class PrefixTuning(nn.Module):
    def __init__(self, args, config: BartConfig):
        super(PrefixTuning, self).__init__()
        self.pre_len = args.pre_leh
        self.prefix_dropout = args.prefix_dropout
        self.mid_dim = config.mid_dim

        self.n_embd = config.d_model
        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.match_n_embd = self.n_embd // self.match_n_head

        self.input_tokens = torch.arange(self.preseqlen).long()

        # prefix for encoder compute
        self.wte_encoder = nn.Embedding(self.pre_len, self.n_embd)
        self.prefix_encoder = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        # prefix for cross compute
        self.wte_cross = nn.Embedding(self.pre_len, self.n_embd)
        self.prefix_cross = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        # prefix for decoder compute
        self.wte_decoder = nn.Embedding(self.pre_len, self.n_embd)
        self.prefix_decoder = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.dropout = nn.Dropout(self.prefix_dropout)

    def forward(self, input_id):
        encode_embeddings = self.wte(input_id)
        past_key_values = self.prefix_encoder(encode_embeddings)
        encoder_past_key_values = self.transform_key_values(past_key_values)

        cross_embedding = self.wte_cross(input_id)
        past_key_values = self.prefix_cross(cross_embedding)
        cross_past_key_values = self.transform_key_values(past_key_values)

        decoder_embedding = self.prefix_decoder(input_id)
        past_key_values = self.prefix_decoder(decoder_embedding)
        decoder_past_key_values = self.transform_key_values(past_key_values)

        return encoder_past_key_values, cross_past_key_values, decoder_past_key_values

    def transform_key_values(self, past_key_values):
        batch_size, seq_len, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, seq_len, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        return past_key_values.permute([2, 0, 3, 1, 4]).split(2)
