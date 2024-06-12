import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim)),
            requires_grad=False,
        )

    def forward(self, x):
        output = nn.functional.embedding(x, self.weight)
        return output
    
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features)),
            requires_grad=False
        )
    
    def forward(self, x):
        output = nn.functional.linear(x, self.weight)
        return output
    
class Normalization(nn.Module):
    def __init__(self, dim, eps = 1e-6, add_offset = True):
        super().__init__()
        self.eps = eps
        self.add_offset = add_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def normalize(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        x = self.normalize(x.float()).type_as(x)
        output = x * self.weight
        if self.add_offset:
            output += x
        return output
    
class MultiLayerProtectron(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_gate = nn.functional.gelu(x_gate, approximate='tanh')
        x_up = self.up_proj(x)
        x_fuse = x_gate * x_up 
        output = self.down_proj(x_fuse)
        return output

