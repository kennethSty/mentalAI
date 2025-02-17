from pathlib import Path
import tiktoken
import torch
from torch.utils.data import DataLoader
import pandas as pd
from typing import List

from gpt2_finetuning_utils  import finetune_loop, calc_accuracy_loader
from src._3_model_preparation.gpt_architecture.gpt_utils.gpt_download import download_and_load_gpt2
from src._3_model_preparation.gpt_architecture.gpt_utils.load_model import load_weights_into_gpt
from src._3_model_preparation.gpt_architecture.GPTModel import GPTModel
from src._0_data_preparation.create_suicide_dataloaders import get_suicide_dataloaders


def finetune_classification_head(is_test = False):

    #setup paths
    train_ds_path = Path("../../../data/02_train_test_splits/train/gpt2_suicide_train.csv")
    val_ds_path = Path("../../../data/02_train_test_splits/train/gpt2_suicide_val.csv")
    test_ds_path = Path("../../../data/02_train_test_splits/test/gpt2_suicide_test.csv")
    models_dir = Path("../../../models/pretrained/gpt2")
    losses_tracker_path = Path("../../../logs/gpt2_finetuning_losses.csv")
    acc_tracker_path = Path("../../../logs/gpt2_finetuning_accs.csv")
    finetuned_model_path = Path('../../../models/finetuned/gpt2_classif_tuned.pth')

    paths_to_check = [
        train_ds_path,
        val_ds_path,
        test_ds_path,
        models_dir,
        losses_tracker_path.parent,  # Parent directory of the CSV file
        acc_tracker_path.parent,  # Parent directory of the CSV file
        finetuned_model_path.parent  # Parent directory of the model file
    ]
    check_and_create_directories(paths_to_check)

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }

    # Load OpenAI Model weights
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir=models_dir
    )

    # Initialize model
    device = "mps" if torch.mps.is_available() else "cpu"
    print('using device: ', device)
    model = GPTModel(BASE_CONFIG)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load weights into model, freeze params and add classification head
    # Unfreeze transformer block and layer_norm because it increases performance
    load_weights_into_gpt(model, params)
    for param in model.parameters():
        param.requires_grad == False
    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad == True
    for param in model.layer_norm.parameters():
        param.requires_grad == True
    model.out_layer = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=2
    )
    model.to(device)

    train_loader, test_loader, val_loader = get_suicide_dataloaders(
        batch_size=8,
        tokenizer=tokenizer,
        train_ds_path=train_ds_path,
        test_ds_path=test_ds_path,
        val_ds_path=val_ds_path
    )

    if is_test:
        assess_pretrain_accuracy(model=model,dataloader=train_loader,device=device, label="train")
        assess_pretrain_accuracy(model=model, dataloader=test_loader, device=device, label="test")
        assess_pretrain_accuracy(model=model, dataloader=val_loader, device=device, label="val")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=5e-5,
        weight_decay=0.1
    )

    train_losses, val_losses, train_accs, val_accs, examples_seen = finetune_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        eval_freq=100,
        checkpoint_freq=100,
        eval_iter=4
    )

    losses_data = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    losses_df = pd.DataFrame(losses_data)
    losses_df.to_csv(losses_tracker_path, index=False)

    accs_data = {
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    accs_df = pd.DataFrame(accs_data)
    accs_df.to_csv(acc_tracker_path, index=False)

    # Save the trained model and weights
    torch.save(model.state_dict(), finetuned_model_path)


def assess_pretrain_accuracy(model: GPTModel, dataloader: DataLoader, device: str, label: str):
    model.eval()
    with torch.inference_mode():
        accuracy = calc_accuracy_loader(
            dataloader, model, device, num_batches=4
        )
    print(f"{label}-accuracy before training:", accuracy)
    model.train()
    return

def check_and_create_directories(paths_to_check: List):
    for path in paths_to_check:
        # Create directories if they don't exist
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
    print("All paths correctly setup.")
    return


if __name__ == "__main__":
    finetune_classification_head(is_test=True)
