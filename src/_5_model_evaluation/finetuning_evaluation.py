from src._3_model_preparation.gpt_architecture.GPTModel import GPTModel
from src._0_data_preparation.create_suicide_dataloaders import get_suicide_dataloaders
from src.config import FINETUNE_EVAL_CONFIG, GPT_BASEMODEL_CONFIG
from src.utils.gpu_utils import DeviceManager
from src._4_model_finetuning.finetuning_utils import evaluate_model_acc, evaluate_model_loss

from pathlib import Path
import tiktoken
import torch
import logging
from datetime import datetime

# Setup logging
def set_up_logging():
    log_dir = Path("../../logs")
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    log_filename = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_finetune_eval_log.txt"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    checkpoint_path = Path("../../models/finetuned/gpt2_checkpoints/checkpoint_step_1000.pth")
    test_ds_path = Path("../../data/02_train_test_splits/test/gpt2/gpt2_suicide_test.csv")
    checkpoint = torch.load(checkpoint_path)
    set_up_logging()

    # Initialize model
    device = DeviceManager().get_device()
    tokenizer = tiktoken.get_encoding("gpt2")

    _, test_loader, _ = get_suicide_dataloaders(
        batch_size= FINETUNE_EVAL_CONFIG["batch_size"],
        tokenizer=tokenizer,
        test_ds_path=test_ds_path,
    )

    model = GPTModel(GPT_BASEMODEL_CONFIG)
    model.out_layer = torch.nn.Linear(
        in_features=GPT_BASEMODEL_CONFIG["emb_dim"],
        out_features=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    eval_accuracy = evaluate_model_acc(
        model=model,
        test_loader=test_loader,
        device=device,
        num_batches=FINETUNE_EVAL_CONFIG["num_eval_batches"]
    )

    log_message = (f"Test accuracy on {FINETUNE_EVAL_CONFIG['num_eval_batches'] * FINETUNE_EVAL_CONFIG['batch_size']} instances: "
                   f"{eval_accuracy * 100:.2f}%")

    logging.info(log_message)
    print(log_message)

if __name__ == "__main__":
    main()