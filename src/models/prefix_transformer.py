import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedBartModel, BartConfig
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 encoder_decoder_attention=False, cache_key=None,
                 prefix_len=1, use_prefix=False):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert cache_key in ['self', 'encoder_decoder', 'encoder']
        if self.encoder_decoder_attention:
            assert cache_key == 'encoder_decoder'
        self.cache_key = cache_key

        self.use_prefix = use_prefix
        self.prefix_len = prefix_len

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads,
                                        self.head_dim).transpose(0, 1)

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        if "prev_key" in saved_state:
            temp_prev_key = saved_state["prev_key"]
            assert temp_prev_key is not None
            prev_key = temp_prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            # static_kv means it is cross attention and prev_key length longer than prefix length
            # set k to prev_key and will not use k
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" is saved_state:
            temp_prev_value = saved_state["prev_value"]
            assert temp_prev_value is not None
            prev_value = temp_prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                # k, v is only prev_key and prev_value
                new_key_padding_mask = prev_key_padding_mask
            else:
                if key_padding_mask is not None:
                    new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
                else:
                    new_key_padding_mask = None
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask

    def forward(self, query, key, key_padding_mask, layer_state, attn_mask, output_attentions):
        static_kv = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        no_extend = False
        # layer state prefix tuning matrix object
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            use_prefix = self.use_prefix
            prefix_len = self.prefix_len
            # static_kv is to compute cross attention
            if "prev_key" in saved_state and static_kv and use_prefix:
                computed_len = saved_state['prev_key'].size(2)
                if computed_len > prefix_len:
                    # key set to None, there will not compute k before use_save_state
                    key = None
                    no_extend = True
            elif "prev_key" in saved_state and static_kv and not use_prefix:
                key = None
                no_extend = True
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        # cross attention
        if static_kv:
            if key is None:
                k = v = None
            else:
                # if pass the key, so use key variable to compute k, v
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            # if there is not pass key, so use query variable to compute k, v
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        # saved_state is not none mean it needs to be processed to prefix tuning
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, no_extend, bsz)

        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask
        }
        assert k is not None
        src_len = k.size(1)
        # torch.bmm is batch matrix dot
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_weights is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_prob = F.dropout(attn_weights, p=self.dropout, training=self.training)

        assert v is not None
        attn_output = torch.bmm(attn_prob, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.tranpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights



class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super(EncoderLayer, self).__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, cache_key='encoder',
                                   use_prefix=True, prefix_len=config.prefix_len)
        self.normalize_before = config.normalize_before



class BartEncoder(nn.Module):
    def __init__(self, config: BartConfig,
                 embed_tokens: Optional[nn.Embedding] = None):
        super(BartEncoder, self).__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model

        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(
            embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim,
                                             self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            self.max_source_positions,
            self.embed_dim,
        )


class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # the embedding vector at padding_idx is not updated during training
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)

        self.init_weights()
