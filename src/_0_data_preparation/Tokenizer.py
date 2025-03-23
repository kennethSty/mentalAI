import tiktoken
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode(self, text: str):
        pass
    @abstractmethod
    def get_pad_token_id(self):
        pass

class GPTTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(
            tokenizer=tiktoken.get_encoding("gpt2")
        )
    def encode(self, text: str):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: List[int]):
        return self.tokenizer.decode(token_ids)

    def get_pad_token_id(self):
        return self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


class PsychBertTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("mnaylor/psychbert-cased")
        )
    def encode(self, text: str):
        return self.tokenizer.encode(text)
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

class EmoBertTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("tae898/emoberta-base")
        )
    def encode(self, text: str):
        return self.tokenizer.encode(text)
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id