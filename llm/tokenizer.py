import os
from typing import List

import sentencepiece


class Tokenizer:

    def __init__(self, model_path: str):
        # Reload tokenizer.
        assert os.path.isfile(model_path), model_path
        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.EncodeAsIds(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.DecodeIds(t)