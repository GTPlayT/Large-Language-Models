import torch
from torch import nn

from llm.config import Config
from llm.tokenizer import Tokenizer
from llm.layers import Embedding
from llm.architecture import Architecture
from llm.focal import FocalLoss
from llm.misc import compute_freqs_cis


class LanguageModel (nn.Module):
    def __init__(self, config = None):
        super().__init__()

        self.torch = torch

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

        # Rotatory Position Embeddings (RoPE)
        # Let X be the token embeddings
        # Let O be the theta at the position P
        # According to the formula:
        # X(new) = X * cos(O) + X(orthogonal) * sin(O)
        # X can look like [[123, 332, 543]]
        rope_theta = getattr(config, 'rope_theta', 10000)

        # This is the part where frequencies are used in cosine and sine function to positional encode information.
        # The formula looks like this:
        # Let PE mean the positional encoding.
        # Let p be the position and i be the dimension.
        # PE(p, 2i) = sin (p / rope_theta ^ (2i / head dimension size)
        # PE(p, 2i + 1) = cos (p / rope_theta ^ (2i / head dimension size)
        freqs_cis = compute_freqs_cis(self.config.head_dim, self.config.max_position_embeddings * 2, theta=rope_theta)
        
        # This is a buffer.
        # In the parent class, i.e., nn.Module, this is not taken into account while calculating back propagtion.
        self.register_buffer('freqs_cis', freqs_cis)

        # To load the model
        self.load_weights(self.config.llm_model)

        self.to(self.config.device)


    def forward_aux(self, input_tokens, input_pos, kv_cache, mask, output_pos):
        freqs_cis = self.freqs_cis.index_select(0, input_pos)
        x = self.embedder(input_tokens) * (self.config.hidden_size ** 0.5)
        x = self.model (
            x,
            freqs_cis,
            input_pos,
            kv_cache,
            mask
        )
        x = x.index_select(1,  output_pos).squeeze(dim=1)
        x = x @ self.embedder.weight.t()
        x = nn.functional.softmax(x, dim=-1)
        return (torch.argmax(x, dim=-1).squeeze(dim=-1), x)
    
    def create_KV_caches(self, batch_size, max_seq_len):
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            k_cache = torch.zeros(size=size, dtype=self.config.dtype_float, device=self.config.device)
            v_cache = torch.zeros(size=size, dtype=self.config.dtype_float, device=self.config.device)
            kv_caches.append((k_cache, v_cache))
        
        return kv_caches
    
    def causal_mask (self, max_seq_len):
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -torch.inf).to(torch.float32)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(self.config.device)
        return mask_tensor
    
    def sliding_window_masking (self, max_seq_len, window_size=50):
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -torch.inf).to(torch.float32)
        for i in range(max_seq_len):
            start = max(0, i - window_size)
            end = min(max_seq_len, i + window_size + 1)
            mask_tensor[0, 0, i, start:end] = 0
        return mask_tensor
    
    def encoder_masking (self, max_seq_len):
        mask_tensor = torch.ones(max_seq_len, max_seq_len)
        mask_tensor = torch.triu(mask_tensor, diagonal=1)
        mask_tensor = mask_tensor == 1
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1).to(torch.float32)
        return mask_tensor

    @torch.no_grad()
    def forward(self, prompts, output_len=100):
        # This part checks if the prompts input is a list or a string;
        # if string, it converts the to a list of reviews
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]
        batch_size = len(prompts)
        
        # Convert the prompts to tokens and then checking if the size of output prompt...
        # and input prompt exceede the maximum number of embeddings the model is able to process
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # KV caches are created here.
        # Intially they all are a list of zeros.
        # K and V look something like [0, 0, 0, 0, ...0] till the maximum sequence lenght of the input;
        # maximum sequence lenght of input = maximum of input prompt lenght from list + the desired output lenght.
        kv_caches = self.create_KV_caches(batch_size, max_seq_len)

        # For this part, we are creating 2 tensors, input tensor and output tensor.
        # The input tensor will have a size of the minimum prompt lenght, it will have the tokens in the given prompts.
        # The output tensor will have a size of the maximum sequence lenght defined by the user.
        token_ids_tensor = torch.full((batch_size, max_seq_len),self.tokenizer.pad_id, dtype=self.config.dtype_int)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=self.config.dtype_int)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(self.config.device)
        input_token_ids_tensor = input_token_ids_tensor.to(self.config.device)

        # Over here, we are comparing what elements are left unfilled and then it is being masked.
        # The final tensor looks like this.
        # Let the pad ID be 0 for this case. (It depends from tokenizer to tokenizer).
        # token_ids_tensor = [[112, 123, 125, 0, 0, 0]]
        # prompt_mask_tensor = [[True, True, True, False, False, False]]
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id

        # This only creates a simple tensor from 0 till the minimum prompt legth.
        # Which should like this as follows.
        # input_positions_tensor = [0, 1, 2, 3, ..., minimum prompt length]
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=self.config.dtype_int).to(self.config.device)

        # Decoder only transformers use causal masking.
        # This where causal masking is created and the tensor is selected.
        # A causal masking looks similar to as follows:
        # [0,   -inf,   -inf, ..., -inf]
        # [0,      0,   -inf, ..., -inf]
        # [0,      0,      0, ..., -inf]
        # ...    ...     ...         ...
        # [0,      0,      0, ...,    0]
        mask_tensor = self.causal_mask(max_seq_len)

        # Since, we are going to be predicting what is next.
        # For that we need to know what is what we currently know.
        # We need to have another variable that is keeping the track of what we currently know.
        # This variable is described as below.
        # It looks something like this.
        # [0,   -inf,   -inf, ..., -inf]
        # [0,      0,   -inf, ..., -inf]
        # ...    ...     ...         ...
        # [0,      0,      0, ..., -inf]
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)

        # This is where we are keeping track of where our output tensor is.
        # At the same time, where is our output index should lie.
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(self.config.device)
        output_index = torch.tensor(min_prompt_len, dtype=self.config.dtype_int).to(self.config.device)

        # This is where we will be generating prompts.
        # We are going to be iterating from how big the user wants the prompts to be.
        for i in range(max_seq_len - min_prompt_len):
            next_token_id, _ = self.forward_aux(
                input_token_ids_tensor,
                input_positions_tensor,
                kv_caches,
                curr_mask_tensor,
                output_positions_tensor,
            )

            # Again, this selects the output mask value.
            # Which looks similar to this.
            # curr_prompt_mask = [True] (if input index is at the prompt)
            # curr_prompt_mask = [False] (if the current masking is not in prompt and it is the generated value)
            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            
            # Like the above, this is where we are selecting the current token value.
            # It should as following:
            # curr_token_ids = [0] (considering the padding value to be 0, which means there was no prompt here)
            # curr_token_ids = [(some natural number)] (this means that there was a prompt here)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)

            # Since we don't want to mess with the original prompts. 
            # We can use PyTorch's where function which is defined as following.
            # torch.where(condition, x, y)
            # if condition is True then pick x else pick y
            # This will help us ensure that we are keeping the original prompts with us.
            # Assuming that there was no prompt here and curr_prompt_mask was False.
            # Take a token 223, for example.
            # Then the output token ID should look like.
            # output_token_ids = [[223]]        
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_id).unsqueeze(dim=1)

            # This is where we will copy the output_token_ids into our output tensor.
            # Why index_copy_ is being used here instead of index_copy is because index_copy_ is an inplace method.
            # It does not return a new tensor while index_copy does. 
            # This method can easily be replaced with index_copy.
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            # This part updates all the mask tensorts and the prepares the for the next loop.
            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=self.config.dtype_int, device=self.config.device)
            output_index = output_index + 1

        # This code converts all the tensors to a list.
        # This converted list is then sent to decoder of the tokenizer.
        # This is truncated output is sent back.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i]) + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        return results[0] if is_str_prompt else results
    

    def train_model_aux(self, prompts, targets, criteria, optimizer):
        self.train()

        batch_size = len(prompts)

        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        targets_tokens = [self.tokenizer.encode(target) for target in targets]

        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)

        max_target_len = max(len(p) for p in targets_tokens)
        
        max_seq_len = max_prompt_len + max_target_len

        if max_seq_len >= self.config.max_position_embeddings:
            raise ValueError("Cannot take these many embeddings in, please look into this issue.")
        
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            dtype = self.config.dtype_float
            k_cache = torch.zeros(size=size, dtype=dtype, device=self.config.device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=self.config.device)
            kv_caches.append((k_cache, v_cache))

        token_ids_tensor = torch.full((batch_size, max_seq_len), self.tokenizer.pad_id, dtype=self.config.dtype_int)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=self.config.dtype_int)
       
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])

        token_ids_tensor = token_ids_tensor.to(self.config.device)
        input_token_ids_tensor = input_token_ids_tensor.to(self.config.device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=self.config.dtype_int).to(self.config.device)

        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -torch.inf).to(torch.float32)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(self.config.device)

        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(self.config.device)
        output_index = torch.tensor(min_prompt_len, dtype=self.config.dtype_int).to(self.config.device)


        for i in range(max_seq_len - min_prompt_len):

            # if batch_size == 1:
            #     target_tokens = torch.tensor(targets_tokens[0][i])
            # else:
            #     target_tokens = torch.tensor([targets_[i] for targets_ in targets_tokens])

            if batch_size == 1:
                target_tokens = torch.tensor(targets_tokens[0][i], dtype=torch.long, device=self.config.device, requires_grad=False)  # Ensure the target is a long tensor
            else:
                target_tokens = torch.tensor([targets_[i] for targets_ in targets_tokens], dtype=torch.long).to(self.config.device)



            # input_token_ids_tensor = input_token_ids_tensor.clone().detach().float().requires_grad_(True)
            # output = input_token_ids_tensor * 2
            # output.sum().backward()

            next_token_id, logits = self.forward_aux(
                input_token_ids_tensor,
                input_positions_tensor,
                kv_caches,
                curr_mask_tensor,
                output_positions_tensor,
            )

            b_size, vocab = logits.shape

            target = torch.zeros((b_size, vocab))
            target[0][target_tokens] = 1
            print(target.shape)

            print(max(logits))

            if next_token_id.dim() == 0:
                # loss = nn.functional.cross_entropy(next_token_id.view(1, 1).to(torch.float), target_tokens.view(1, 1).to(torch.float))
                # loss = criteria(next_token_id.view(1, 1), target_tokens.view(1, 1))
                loss = criteria(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # try:
                #     loss.backward()
                # except:
                #     print(next_token_id, target_tokens)
                # optimizer.step()
            else:
                loss = criteria(next_token_id.view(-1, 1).float(), target_tokens.view(-1, 1).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, target_tokens).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=self.config.dtype_int).to(self.config.device)
            output_index = output_index + 1

        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i]) + max_target_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        return results[0] if is_str_prompt else results
    
    def train_model(self, prompts, targets, lr=0.001):
        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        is_str_target = isinstance(targets, str)
        if is_str_target:
            targets = [targets]

        self.train_model_aux(prompts, targets, criteria, optimizer)




    def load_weights(self, model_path):
        self.load_state_dict(
            torch.load(
                model_path, mmap=True, weights_only=True,
            )['model_state_dict'],
            strict=True,
        )

