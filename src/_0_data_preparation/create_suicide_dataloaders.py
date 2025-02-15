from src._0_data_preparation.SuicideDataset import SuicideDataset

from torch.utils.data import DataLoader
import tiktoken
from pathlib import Path
import os

def get_suicide_dataloaders(
        batch_size: int,
        tokenizer: tiktoken.Encoding,
        train_ds_path: str,
        test_ds_path: str,
        val_ds_path: str):

    num_workers = max(1, os.cpu_count() - 1)

    train_dataset = SuicideDataset(
        csv_file_path=train_ds_path,
        tokenizer=tokenizer
    )
    test_dataset = SuicideDataset(
        csv_file_path=test_ds_path,
        tokenizer=tokenizer
    )
    val_dataset = SuicideDataset(
        csv_file_path=val_ds_path,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False, #because IterableDataset does not support shuffling
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )

    print("Loaded finetuning datasets")

    return train_loader, test_loader, val_loader

# Teest the loading
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    batch_size = 8
    train_ds_path = Path("../../data/02_train_test_splits/train/suicide_train.csv")
    val_ds_path =  Path("../../data/02_train_test_splits/train/suicide_val.csv")
    test_ds_path = Path("../../data/02_train_test_splits/test/suicide_test.csv")

    train_loader, test_loader, val_loader = get_suicide_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        train_ds_path=train_ds_path,
        val_ds_path=val_ds_path,
        test_ds_path=test_ds_path,
    )

    print("Setting up loaders successful")
    print("Train batches", train_loader)
    print("Test batches", test_loader)
    print("Val batches", val_loader)