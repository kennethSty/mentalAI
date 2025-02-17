from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src._4_model_finetuning.finetuning_utils import finetune_loop, calc_accuracy_loader, assess_pretrain_accuracy
from src.utils.paths_utils import check_and_create_directories
from src._3_model_preparation.psychbert_architecture.PsychBertClassifier import PsychBertClassifier
from src._0_data_preparation.create_suicide_dataloaders import get_suicide_dataloaders
from src.config import FINETUNE_CONFIG, TOKENIZER_CONFIG
from src.utils.gpu_utils import DeviceManager

def finetune_classification_head(is_test = False):

    model_flag = "psychbert"

    #setup paths
    train_ds_path = Path(f"../../../data/02_train_test_splits/train/{model_flag}/{model_flag}_suicide_train.csv")
    val_ds_path = Path(f"../../../data/02_train_test_splits/train/{model_flag}/{model_flag}_suicide_val.csv")
    test_ds_path = Path(f"../../../data/02_train_test_splits/test/{model_flag}/{model_flag}_suicide_test.csv")
    losses_tracker_path = Path(f"../../../logs/{model_flag}_finetuning_losses.csv")
    acc_tracker_path = Path(f"../../../logs/{model_flag}_finetuning_accs.csv")
    finetuned_model_path = Path(f'../../../models/finetuned/{model_flag}_classif_tuned.pth')

    paths_to_check = [
        train_ds_path,
        val_ds_path,
        test_ds_path,
        losses_tracker_path.parent,  # Parent directory of the CSV file
        acc_tracker_path.parent,  # Parent directory of the CSV file
        finetuned_model_path.parent  # Parent directory of the model file
    ]
    check_and_create_directories(paths_to_check)

    # Initialize model
    device = DeviceManager().get_device()
    model = PsychBertClassifier()
    tokenizer = TOKENIZER_CONFIG[model_flag]
    model.to(device)

    train_loader, test_loader, val_loader = get_suicide_dataloaders(
        batch_size=FINETUNE_CONFIG["batch_size"],
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
        lr=FINETUNE_CONFIG["lr"],
        weight_decay=FINETUNE_CONFIG["weight_decay"]
    )

    train_losses, val_losses, train_accs, val_accs, examples_seen = finetune_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=FINETUNE_CONFIG["num_epochs"],
        eval_freq=FINETUNE_CONFIG["eval_freq"],
        checkpoint_freq=FINETUNE_CONFIG["checkpoint_freq"],
        eval_iter=FINETUNE_CONFIG["eval_iter"],
        checkpoint_dir=f'../../../{model_flag}_checkpoints'
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


if __name__ == "__main__":
    finetune_classification_head()
