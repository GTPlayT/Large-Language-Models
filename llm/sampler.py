import torch
from torch import nn

from llm.config import Config

class Sampler(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @torch.inference_mode()
    def forward(self, x, emb, out_pos, temp, top_ps, top_ks, emb_bias = None):
        x = x.index_select(1, out_pos).squeeze(dim=1)
        
        logits = x @ emb.t()

        if emb_bias is not None:
            logits += emb_bias
        
        if temp is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)
        
        probs = torch.softmax(logits, dim=-1, dtype=self.config.dtype_float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)
        return next_token_ids, logits