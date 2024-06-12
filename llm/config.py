import torch

class Config:
    def __init__(self):
        self._vocab_size: int = 256000
        self._max_position_embeddings: int = 8192
        self._num_hidden_layers: int = 18
        self._num_attention_heads: int = 8
        self._num_key_value_heads: int = 1
        self._hidden_size: int = 2048
        self._intermediate_size: int = 16384
        self._head_dim: int = 256
        self._rms_norm_eps: float = 1e-6
        self._dtype_float: torch.dtype = torch.float
        self._dtype_int: torch.dtype = torch.int8
        self._quant: bool = False
        self._device: str = 'cpu'
        self._tokenizer_model: str = 'llm/models/tokenizer.model'
        self._llm_model: str = 'llm/models/gemma-2b-it.ckpt'
    
    # Getters and Setters for each attribute
    @property
    def vocab_size(self):
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    @property
    def max_position_embeddings(self):
        return self._max_position_embeddings

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        self._max_position_embeddings = value

    @property
    def num_hidden_layers(self):
        return self._num_hidden_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        self._num_hidden_layers = value

    @property
    def num_attention_heads(self):
        return self._num_attention_heads

    @num_attention_heads.setter
    def num_attention_heads(self, value):
        self._num_attention_heads = value

    @property
    def num_key_value_heads(self):
        return self._num_key_value_heads

    @num_key_value_heads.setter
    def num_key_value_heads(self, value):
        self._num_key_value_heads = value

    @property
    def hidden_size(self):
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, value):
        self._hidden_size = value

    @property
    def intermediate_size(self):
        return self._intermediate_size

    @intermediate_size.setter
    def intermediate_size(self, value):
        self._intermediate_size = value

    @property
    def head_dim(self):
        return self._head_dim

    @head_dim.setter
    def head_dim(self, value):
        self._head_dim = value

    @property
    def rms_norm_eps(self):
        return self._rms_norm_eps

    @rms_norm_eps.setter
    def rms_norm_eps(self, value):
        self._rms_norm_eps = value

    @property
    def dtype_float(self):
        return self._dtype_float

    @dtype_float.setter
    def dtype_float(self, value):
        if value not in [torch.float, torch.float16, torch.float32, torch.float64]:
            raise ValueError("dtype must be one of torch.float16, torch.float32, or torch.float64")
        self._dtype_float = value

    @property
    def dtype_int(self):
        return self._dtype_int

    @dtype_int.setter
    def dtype_int(self, value):
        if value not in [torch.float, torch.float16, torch.float32, torch.float64]:
            raise ValueError("dtype must be one of torch.float16, torch.float32, or torch.float64")
        self._dtype_int = value

    @property
    def quant(self):
        return self._quant

    @quant.setter
    def quant(self, value):
        self._quant = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value not in ['cpu', 'cuda', 'xpu']:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        self._device = value

    @property
    def tokenizer_model(self):
        return self._tokenizer_model

    @tokenizer_model.setter
    def tokenizer_model(self, value):
        self._tokenizer_model = value

    @property
    def llm_model(self):
        return self._llm_model

    @llm_model.setter
    def llm_model(self, value):
        self._llm_model = value

