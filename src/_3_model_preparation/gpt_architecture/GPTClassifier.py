import torch.nn as nn
from pathlib import Path
import torch

from src.config import GPT_BASEMODEL_CONFIG
from src._3_model_preparation.gpt_architecture.gpt_utils.gpt_download import download_and_load_gpt2
from src._3_model_preparation.gpt_architecture.gpt_utils.load_model import load_weights_into_gpt
from src._3_model_preparation.gpt_architecture.GPTModel import GPTModel

class GPTClassifier(nn.Module):
    def __init__(self, models_dir = Path("../../../models/pretrained/gpt2")):
        super().__init__()
        self.settings, self.params = download_and_load_gpt2(
            model_size=GPT_BASEMODEL_CONFIG["model_size"], models_dir=models_dir
        )
        self.model = GPTModel(GPT_BASEMODEL_CONFIG)
        self.model = load_weights_into_gpt(self.model, self.params)
        for param in self.model.parameters():
            param.requires_grad == False
        for param in self.model.transformer_blocks[-1].parameters():
            param.requires_grad == True
        for param in self.model.layer_norm.parameters():
            param.requires_grad == True
        self.model.out_layer = torch.nn.Linear(
            in_features=GPT_BASEMODEL_CONFIG["emb_dim"],
            out_features=2
        )

    def forward(self, x):
        return self.model(x)[:, -1, :]