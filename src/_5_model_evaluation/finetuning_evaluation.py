from src._0_data_preparation.create_suicide_dataloaders import get_suicide_dataloaders
from src._5_model_evaluation.evaluation_utils import load_finetuned_model
from src.config import FINETUNE_EVAL_CONFIG
from src.utils.gpu_utils import DeviceManager
from src._4_model_finetuning.finetuning_utils import evaluate_model_predscores
from src.utils.paths_utils import set_up_logging
from src.config import TOKENIZER_CONFIG

import torch.nn as nn
import torch
from typing import List, Dict
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import csv




def evaluate_model_on_testset(model: nn.Module, test_loader: DataLoader, device: str, model_flag: str):
    set_up_logging(log_file_name=f"{model_flag}_finetune_eval_log.txt")

    precision, recall, accuracy, f1, false_classified_inputs = evaluate_model_predscores(
        model=model,
        data_loader=test_loader,
        device=device,
        num_batches=FINETUNE_EVAL_CONFIG["num_eval_batches"]
    )

    save_missclassifications(
        false_classified_inputs=false_classified_inputs,
        model_flag=model_flag
    )

    if FINETUNE_EVAL_CONFIG['num_eval_batches'] is None:
        num_instances = FINETUNE_EVAL_CONFIG['batch_size'] * len(test_loader)
    else:
        num_instances = FINETUNE_EVAL_CONFIG['batch_size'] * FINETUNE_EVAL_CONFIG['num_eval_batches']

    log_message = (f"Evaluation on {num_instances} instances: "
                   f" Accuracy: {accuracy * 100:.2f}%"
                   f" Precision: {precision:.2f}"
                   f" Recall: {recall:.2f}"
                   f" F1: {f1:.2f} \n"
                   f" False classified inputs: {false_classified_inputs}"
                   )
    logging.info(log_message)
    print(log_message)

def save_missclassifications(false_classified_inputs: List[Dict[str, torch.Tensor]], model_flag: str):
    tokenizer = TOKENIZER_CONFIG[model_flag]
    save_path = Path(f"../../data/04_evaluation/misclassifications_{model_flag}.csv")

    with open(save_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=["pred_label", "input"])
        writer.writeheader()
        for row in false_classified_inputs:
            row["input"] = tokenizer.decode(row["input"].tolist())
            writer.writerow(row)
    return

if __name__ == "__main__":
    device = DeviceManager().get_device()
    assert device == "cuda" or device == "mps", "Evaluation has to be done on GPU"

    #Set model flag to choose the right tokenized test datasets
    model_flag = "gpt2"  # gpt2 #emobert, #psychbert
    model, tokenizer = load_finetuned_model(model_flag=model_flag, device=device)

    test_ds_path = Path(f"../../data/02_train_test_splits/test/{model_flag}/{model_flag}_suicide_test.csv")
    _, test_loader, _ = get_suicide_dataloaders(
        batch_size=FINETUNE_EVAL_CONFIG["batch_size"],
        tokenizer=tokenizer,
        test_ds_path=test_ds_path,
    )

    evaluate_model_on_testset(model, test_loader, device, model_flag=model_flag)