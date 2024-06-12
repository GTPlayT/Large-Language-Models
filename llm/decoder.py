import torch
from torch import nn

from llm.layers import Normalization, MultiLayerProtectron
from llm.attention import Attention
from llm.config import Config

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.self_attn = Attention (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim
        )

        self.mlp = MultiLayerProtectron (
            config.hidden_size,
            config.intermediate_size
        )

        self.input_layernorm = Normalization(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.post_attention_layernorm = Normalization(
            config.hidden_size,
            config.rms_norm_eps
        )

    def forward(self, x, freqs_cis, kv_write_indicies, kv_cache, mask):
        residue = x
        x = self.input_layernorm(x)
        x = self.self_attn (
            x,
            freqs_cis,
            kv_write_indicies,
            kv_cache,
            mask
        )
        
        x += residue

        residue = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x += residue

        return x


