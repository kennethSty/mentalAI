from src._0_data_preparation.SuicideStreamDataset import SuicideDataset, SuicideStreamDataset

from torch.utils.data import DataLoader
import tiktoken
from pathlib import Path

from src._0_data_preparation.Tokenizer import Tokenizer
from src.config import TOKENIZER_CONFIG, FINETUNE_CONFIG
from src.utils.paths_utils import check_and_create_directories


def get_suicide_dataloaders(
        batch_size: int,
        tokenizer: Tokenizer,
        train_ds_path=None,
        test_ds_path=None,
        val_ds_path=None):

    num_workers = 0 #max(1, os.cpu_count() - 1)
    train_loader, test_loader, val_loader = None, None, None

    if train_ds_path is not None:
        train_dataset = SuicideStreamDataset(
            csv_file_path=train_ds_path,
            tokenizer=tokenizer
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # because IterableDataset does not support shuffling
            drop_last=True
        )
    if test_ds_path is not None:
        #the test set is small enough for a regular dataset
        test_dataset = SuicideDataset(
            csv_file_path=test_ds_path,
            tokenizer=tokenizer
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )

    if val_ds_path is not None:
        val_dataset = SuicideStreamDataset(
            csv_file_path=val_ds_path,
            tokenizer=tokenizer
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

# Test the loading
if __name__ == "__main__":

    model_flag = "psychbert"
    tokenizer = TOKENIZER_CONFIG[model_flag]

    train_ds_path = Path(f"../../data/02_train_test_splits/train/{model_flag}/{model_flag}_suicide_train.csv")
    val_ds_path =  Path(f"../../data/02_train_test_splits/train/{model_flag}/{model_flag}_suicide_val.csv")
    test_ds_path = Path(f"../../data/02_train_test_splits/test/{model_flag}/{model_flag}_suicide_test.csv")
    paths = [train_ds_path, val_ds_path, test_ds_path]
    check_and_create_directories(paths)

    train_loader, test_loader, val_loader = get_suicide_dataloaders(
        tokenizer=tokenizer,
        batch_size=FINETUNE_CONFIG["batch_size"],
        train_ds_path=train_ds_path,
        val_ds_path=val_ds_path,
        test_ds_path=test_ds_path,
    )

    print("Setting up loaders successful")
    print("Train batches", train_loader)
    print("Test batches", test_loader)
    print("Val batches", val_loader)