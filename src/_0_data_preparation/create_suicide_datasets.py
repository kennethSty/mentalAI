import pandas as pd
from pathlib import Path
import csv
from typing import Dict, List
from src.utils.paths_utils import check_and_create_directories
from src._0_data_preparation.Tokenizer import Tokenizer
from src.config import SUICIDE_DS_CONFIG, TOKENIZER_CONFIG

def create_suicide_datasets(raw_data_path: str,
                            tokenizer,
                            encoded_data_path: str,
                            train_ds_path: str,
                            val_ds_path: str,
                            test_ds_path: str):
    """ Encodes, padds and splits the suicide dataset """
    encode_clean_texts(raw_data_path=raw_data_path, encoded_data_path=encoded_data_path, tokenizer=tokenizer)
    df = pd.read_csv(filepath_or_buffer=encoded_data_path)
    train_df, validation_df, test_df = random_split(df, train_frac=SUICIDE_DS_CONFIG["train_frac"], val_frac=SUICIDE_DS_CONFIG["val_frac"])
    train_df.to_csv(train_ds_path, index=None)
    validation_df.to_csv(val_ds_path, index=None)
    test_df.to_csv(test_ds_path, index=None)
    print("saved train, validation and text data")


def encode_clean_texts(raw_data_path:str, encoded_data_path:str, tokenizer: Tokenizer, batch_size=50):
    with open(raw_data_path, "r") as input_csv,\
        open(encoded_data_path, "w") as output_csv:
        reader = csv.DictReader(input_csv)
        writer = csv.DictWriter(output_csv, fieldnames=["encoded_text", "class"])
        writer.writeheader()

        encoded_texts = []
        labels = []
        max_encoding_length = 0

        for row in reader:
            encode_texts, labels, max_encoding_length = clean_text(
                row=row,
                encoded_texts=encoded_texts,
                labels=labels, tokenizer=tokenizer,
                max_encoding_length=max_encoding_length
            )

            if len(encoded_texts) > batch_size:
                rows_to_write = [
                    {"encoded_text": text, "class": label }
                    for text, label in zip(encoded_texts, labels)
                ]
                writer.writerows(rows_to_write)
                encoded_texts.clear()
                labels.clear()

        if encoded_texts:
            rows_to_write = [
                {"encoded_text": text, "class": label}
                for text, label in zip(encoded_texts, labels)
            ]
            writer.writerows(rows_to_write)
            encoded_texts.clear()
            labels.clear()
        print("Successfully encoded texts")
        print("Maximum encoded tokens per text: ", max_encoding_length)

def clean_text(row: Dict[str, str], encoded_texts: List, labels: List, tokenizer: Tokenizer, max_encoding_length: int):
    if (len(row["text"]) > 0) and (len(row["class"]) > 0): #both fields not empty
        encoded_text = tokenizer.encode(row["text"])
        if (len(encoded_text) <= SUICIDE_DS_CONFIG["max_token_length"]): #based on suicide_token_count_hist.png
            if len(encoded_text) > max_encoding_length:
                max_encoding_length = len(encoded_text)
            label = 1 if (row["class"].strip()=="suicide") else 0
            encoded_texts.append(encoded_text)
            labels.append(label)
    return encoded_texts, labels, max_encoding_length

def random_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    #Shuffle entire dataset
    df = df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    train_end_idx = int(len(df) * train_frac)
    val_end_idx = train_end_idx + int(len(df) * val_frac)
    train_df = df[:train_end_idx]
    validation_df = df[train_end_idx:val_end_idx]
    test_df = df[val_end_idx:]

    return train_df, validation_df, test_df

if __name__ == "__main__":
    model_flag = "psychbert" #or gpt2
    raw_data_path = Path(f"../../data/00_raw/suicide_detection.csv")
    encoded_data_path = Path(f"../../data/01_preprocessed/{model_flag}/{model_flag}_encoded_suicide_detection.csv")
    train_ds_path = Path(f"../../data/02_train_test_splits/train/{model_flag}/{model_flag}_suicide_train.csv")
    val_ds_path =  Path(f"../../data/02_train_test_splits/train/{model_flag}/{model_flag}_suicide_val.csv")
    test_ds_path = Path(f"../../data/02_train_test_splits/test/{model_flag}/{model_flag}_suicide_test.csv")
    paths = [encoded_data_path, train_ds_path, val_ds_path, test_ds_path]
    check_and_create_directories(paths)

    tokenizer = TOKENIZER_CONFIG[model_flag]

    create_suicide_datasets(raw_data_path=raw_data_path,
                            tokenizer=tokenizer,
                            encoded_data_path=encoded_data_path,
                            train_ds_path=train_ds_path,
                            val_ds_path=val_ds_path,
                            test_ds_path=test_ds_path)