import torch
from torch import nn

from llm.config import Config
from llm.tokenizer import Tokenizer
from llm.layers import Embedding
from llm.architecture import Architecture
from llm.sampler import Sampler
from llm.misc import compute_freqs_cis


class LanguageModel (nn.Module):
    def __init__(self, config = None):
        super().__init__()

        if config is None:
            self.config = Config()
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Language model will be using {self.config.device}.")
        else:
            self.config = config

        even_chunk_divison = self.config.hidden_size % self.config.num_attention_heads
        if even_chunk_divison != 0:
            raise ValueError(f"The number of hidden states are not properly distributed between the number of attention heads.")
        
        self.tokenizer = Tokenizer(self.config.tokenizer_model)
        self.embedder = Embedding(self.config.vocab_size, self.config.hidden_size)
        self.model = Architecture(self.config)
        self.sampler = Sampler(self.config)

        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = compute_freqs_cis(self.config.head_dim, self.config.max_position_embeddings * 2, theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)

        self.load_weights(self.config.llm_model)

        self.to(self.config.device)

    def forward(self, input_tokens, input_pos, kv_cache, mask, output_pos, temp, top_ps, top_ks):
        freqs_cis = self.freqs_cis.index_select(0, input_pos)
        x = self.embedder(input_tokens) * (self.config.hidden_size ** 0.5)
        x = self.model (
            x,
            freqs_cis,
            input_pos,
            kv_cache,
            mask
        )
        next_tokens, logits = self.sampler (
            x,
            self.embedder.weight,
            output_pos,
            temp,
            top_ps,
            top_ks
        )
        return next_tokens, logits
    
    def generate(self, prompts, output_len=100, temp=0.5, top_p=1, top_k=100):
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            dtype = self.config.dtype_float
            k_cache = torch.zeros(size=size, dtype=dtype, device=self.config.device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=self.config.device)
            kv_caches.append((k_cache, v_cache))

        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(self.config.device)
        input_token_ids_tensor = input_token_ids_tensor.to(self.config.device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(self.config.device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(self.config.device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            self.config.device)
        temperatures_tensor = None if not temp else torch.FloatTensor(
            [temp] * batch_size).to(self.config.device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(self.config.device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(self.config.device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            self.config.device)

        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids_tensor,
                input_positions_tensor,
                kv_caches,
                curr_mask_tensor,
                output_positions_tensor,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                self.config.device)
            output_index = output_index + 1

        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        return results[0] if is_str_prompt else results

    def load_weights(self, model_path):
        self.load_state_dict(
            torch.load(
                model_path, mmap=True, weights_only=True,
            )['model_state_dict'],
            strict=True,
        )

