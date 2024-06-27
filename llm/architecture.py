import torch
from torch import nn

from llm.layers import Normalization
from llm.decoder import Decoder
from llm.config import Config

class Architecture(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList()
        for _ in range(self.config.num_hidden_layers):
            self.layers.append(Decoder(self.config))
        self.norm = Normalization(self.config.hidden_size, self.config.rms_norm_eps)
    
    def forward(self, x, freqs_cis, kv_write_indices, kv_cache, mask):
        for i in range(len(self.layers)):
            self.layers[i].to(self.config.device)
            x = self.layers[i](
                x, 
                freqs_cis,
                kv_write_indices,
                kv_cache[i],
                mask
            )
            self.layers[i].to(self.config.idle_device)
        x = self.norm(x)
        return x
