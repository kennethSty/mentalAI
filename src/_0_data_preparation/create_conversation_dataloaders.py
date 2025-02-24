from src._0_data_preparation.ConversationStreamDataset import ConversationStreamDataset
from torch.utils.data import DataLoader
from src._0_data_preparation.Tokenizer import Tokenizer


def get_conversation_dataloaders(
        batch_size: int,
        tokenizer: Tokenizer,
        train_ds_path=None,
        test_ds_path=None,
        val_ds_path=None):

    num_workers = 0 #max(1, os.cpu_count() - 1)
    train_loader, test_loader, val_loader = None, None, None

    if train_ds_path is not None:
        train_dataset = ConversationStreamDataset(
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
        test_dataset = ConversationStreamDataset(
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
        val_dataset = ConversationStreamDataset(
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

    print("Loaded datasets")

    return train_loader, test_loader, val_loader
