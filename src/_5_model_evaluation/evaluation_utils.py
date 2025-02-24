from pathlib import Path

import torch

from src._3_model_preparation.emobert_architecture.EmoBertClassifier import EmoBertClassifier
from src._3_model_preparation.gpt_architecture.GPTClassifier import GPTClassifier
from src._3_model_preparation.psychbert_architecture.PsychBertClassifier import PsychBertClassifier
from src.config import TOKENIZER_CONFIG


def load_finetuned_model(model_flag: str, device: str):
    checkpoint_path = Path(f"models/finetuned/{model_flag}_checkpoints/checkpoint_step_8000.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_flag == "gpt2":
        model = GPTClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        tokenizer = TOKENIZER_CONFIG[model_flag]
    elif model_flag == "psychbert":
        model = PsychBertClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        tokenizer = TOKENIZER_CONFIG[model_flag]
    else:
        assert model_flag == "emobert", "model_flag should be emobert"
        model = EmoBertClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        tokenizer = TOKENIZER_CONFIG[model_flag]

    return model, tokenizer