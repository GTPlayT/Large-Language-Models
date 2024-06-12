import torch
from torch import nn

from llm.layers import Linear
from llm.misc import rotary_emb

class Attention (nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        even_chunk_division = self.num_heads % self.num_kv_heads
        if even_chunk_division != 0:
            raise ValueError("The number of heads can not be distributed evenly among the number of kv heads.")
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = Linear (
            self.hidden_size, 
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        )
        self.o_proj = Linear (
            self.num_heads * self.head_dim,
            self.hidden_size
        )

    def forward(self, x, freqs_cis, kv_write_indices, kv_cache, mask):
        batch_size, seqeunce_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        xq = rotary_emb(xq, freqs_cis=freqs_cis)
        xk = rotary_emb(xk, freqs_cis=freqs_cis)

        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache

        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(key, self.num_heads // self.num_kv_heads, dim=2)
            value = torch.repeat_interleave(value, self.num_heads // self.num_kv_heads, dim=2)

        q = xq.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        weight = ((q @ k.transpose(2, 3)) * self.head_dim ** -0.5) + mask
        weight = nn.functional.softmax(weight.float(), dim=-1).type_as(q)
        weight @= v

        output = weight.transpose(1, 2).contiguous().view(batch_size, seqeunce_len, -1)
        output = self.o_proj(output)
        return output





