import csv
from torch.utils.data import IterableDataset, Dataset

from src.utils.csv_utils import CSVUtils
from src._0_data_preparation.Tokenizer import Tokenizer

class ConversationStreamDataset(IterableDataset):
    """
    Iterable finetuning dataset used for training with lazy loading.
    We extend IterableDataset because the normal Dataset requires to load the full Dataset into memory which is too large.
    """
    def __init__(self,
                 tokenizer: Tokenizer,
                 csv_file_path: str):

        CSVUtils.increase_csv_maxsize()
        self.csv_file_path = csv_file_path
        self.tokenizer = tokenizer  #use same tokenizer that pretrained model used!
        self.pad_token_id = tokenizer.get_pad_token_id()

    def __iter__(self):
        with open(self.csv_file_path, "r") as input_csv:
            reader = csv.DictReader(input_csv)
            for i, row in enumerate(reader):
                question = row["question(s)"]
                answer = row["answer(s)"]
                # # Truncate & Pad
                # encoded_text = encoded_text[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(encoded_text))
                # attention_mask = [1 for _ in range(len(encoded_text))] + [0] * max(0, self.max_length - len(encoded_text))

                # Use yield for efficient streaming (returns and remembers where left of in iteration)
                yield (
                    question,
                    answer
                )

# class ConversationDataset(Dataset):
#     def __init__(self, csv_file_path: str,
#                  tokenizer: Tokenizer,
#                  max_length=SUICIDE_DS_CONFIG["max_token_length"]):

#         self.max_length = max_length
#         self.data = pd.read_csv(csv_file_path)
#         self.data["encoded_text"] = self.data.apply(lambda row: ast.literal_eval(row["encoded_text"]), axis = 1)
#         self.tokenizer = tokenizer
#         self.pad_token_id = self.tokenizer.get_pad_token_id()
#         self.data["encoded_text"] = self.data["encoded_text"].apply(
#             lambda encoded_text:
#             encoded_text + [self.pad_token_id] * (self.max_length - len(encoded_text))
#         )
#         self.data["attention_mask"] = self.data["encoded_text"].apply(
#             lambda encoded_text:
#             [1 for _ in encoded_text] + [0] * (self.max_length - len(encoded_text))
#         )

#     def __getitem__(self, index):
#         encoded = self.data.iloc[index]["encoded_text"]
#         attention_mask = self.data.iloc[index]["attention_mask"]
#         label = self.data.iloc[index]["class"]
#         return (
#             torch.tensor(encoded, dtype=torch.long),
#             torch.tensor(label, dtype=torch.long),
#             torch.tensor(attention_mask, dtype=torch.long)
#         )

#     def __len__(self):
#         return len(self.data)
