import csv
import ast
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset
import tiktoken

from src.utils.csv_utils import CSVUtils

class SuicideStreamDataset(IterableDataset):
    """
    Iterable finetuning dataset used for training with lazy loading.
    We extend IterableDataset because the normal Dataset requires to load the full Dataset into memory which is too large.
    """
    def __init__(self,
                 tokenizer: tiktoken.Encoding,
                 csv_file_path: str,
                 max_length=499,
                 padding_text="<|endoftext|>"):

        CSVUtils.increase_csv_maxsize()
        self.csv_file_path = csv_file_path
        self.tokenizer = tokenizer  #use same tokenizer that pretrained model used!
        self.max_length = max_length
        self.pad_token_id = tokenizer.encode(padding_text, allowed_special={"<|endoftext|>"})[0]

    def __iter__(self):
        with open(self.csv_file_path, "r") as input_csv:
            reader = csv.DictReader(input_csv)
            for i, row in enumerate(reader):
                encoded_text = ast.literal_eval(row["encoded_text"])
                label = int(row["class"])
                # Truncate & Pad
                encoded_text = encoded_text[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(encoded_text))
                # Use yield for efficient streaming (returns and remembers where left of in iteration)
                yield (
                    torch.tensor(encoded_text, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long)
                )

class SuicideDataset(Dataset):
    def __init__(self, csv_file_path: str,
                 tokenizer: tiktoken.Encoding,
                 max_length=499,
                 padding_text="<|endoftext|>"):

        self.max_length = max_length
        self.data = pd.read_csv(csv_file_path)
        self.data["encoded_text"] = self.data.apply(lambda row: ast.literal_eval(row["encoded_text"]), axis = 1)
        pad_token_id = tokenizer.encode(padding_text, allowed_special={"<|endoftext|>"})[0]
        self.data["encoded_text"] = self.data["encoded_text"].apply(
            lambda encoded_text:
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
        )

    def __getitem__(self, index):
        encoded = self.data.iloc[index]["encoded_text"]
        label = self.data.iloc[index]["class"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
