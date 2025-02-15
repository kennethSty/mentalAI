import csv

import torch
from torch.utils.data import IterableDataset
import tiktoken
from src.utils.csv_utils import CSVUtils
import ast

class SuicideDataset(IterableDataset):
    """
    Iterable finetuning dataset for classifying suicidal intent with lazy loading.
    We extend IterableDataset because the normal Dataset requires to load the full Dataset into memory which is too large.
    """
    def __init__(self,
                 tokenizer: tiktoken.Encoding,
                 max_length=800,
                 csv_file_path="../../data/01_preprocessed/encoded_suicide_detection.csv",
                 padding_text="<|endoftext|>"):

        self.csv_file_path = csv_file_path
        self.tokenizer = tokenizer  #use same tokenizer that pretrained model used!
        self.max_length = max_length
        self.pad_token_id =  tokenizer.encode(padding_text, allowed_special={"<|endoftext|>"})[0]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, step = 0, 1
        else:
            #each worker must read a unique part of the dataset to avoid duplicate rows.
            #=> Each worker processes every 4th row
            start, step = worker_info.id, worker_info.num_workers

        CSVUtils.increase_csv_maxsize()
        with open(self.csv_file_path, "r") as input_csv:
            reader = csv.DictReader(input_csv)
            for i, row in enumerate(reader):
                if i % step == start:
                    encoded_text = ast.literal_eval(row["encoded_text"])
                    label = int(row["class"])
                    # Truncate & Pad
                    encoded_text = encoded_text[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(encoded_text))
                    # Use yield for efficient streaming (returns and remembers where left of in iteration)
                    yield (
                        torch.tensor(encoded_text, dtype=torch.long),
                        torch.tensor(label, dtype=torch.long)
                    )