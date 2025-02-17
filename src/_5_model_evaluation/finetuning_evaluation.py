from src._3_model_preparation.gpt_architecture.GPTModel import GPTModel
from src._4_model_finetuning.gpt2_finetuning.gpt2_finetuning_utils import calc_accuracy_loader
from src._0_data_preparation.create_suicide_dataloaders import get_suicide_dataloaders

from torch.utils.data import DataLoader
from pathlib import Path
import tiktoken
import torch

def model_test_eval(model: GPTModel, test_loader: DataLoader, device: str, num_batches: int):
    model.eval()
    with torch.inference_mode():
        test_accuracy = calc_accuracy_loader(
            test_loader, model, device, num_batches = num_batches
        )
    print(test_accuracy)
    return test_accuracy

def main():
    checkpoint_path = Path("../../models/finetuned/gpt2_checkpoints/checkpoint_step_1000.pth")
    test_ds_path = Path("../../data/02_train_test_splits/test/gpt2_suicide_test.csv")
    checkpoint = torch.load(checkpoint_path)
    num_batches = 100
    batch_size = 8

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }

    # Initialize model
    device = "mps" if torch.mps.is_available() else "cpu"
    print('using device: ', device)
    tokenizer = tiktoken.get_encoding("gpt2")

    _, test_loader, _ = get_suicide_dataloaders(
        batch_size=batch_size,
        tokenizer=tokenizer,
        test_ds_path=test_ds_path,
    )

    model = GPTModel(BASE_CONFIG)
    model.out_layer = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    eval_accuracy = model_test_eval(
        model=model,
        test_loader=test_loader,
        device=device,
        num_batches=num_batches
    )
    print(f"Test accuracy on {num_batches * batch_size} instances: {eval_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()