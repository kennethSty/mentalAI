import torch.nn as nn
from pathlib import Path
import torch

from src.config import GPT_BASEMODEL_CONFIG
from src._3_model_preparation.gpt_architecture.gpt_utils.gpt_download import download_and_load_gpt2
from src._3_model_preparation.gpt_architecture.gpt_utils.load_model import load_weights_into_gpt
from src._3_model_preparation.gpt_architecture.GPTModel import GPTModel
from src.utils.gpu_utils import DeviceManager
from src.config import TOKENIZER_CONFIG


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
        self.classif_config = {
            "id2label": {0: "not suicidal", 1: "suicidal"},
            "label2id": {"not suicidal": 0, "suicidal": 1},
            "num_labels": 2
        }
        self.device = DeviceManager().get_device()

    def forward(self, token_ids):
        return self.model(token_ids)[:, -1, :]

    def classify(self, x: str, max_length = 499):
        tokenizer = TOKENIZER_CONFIG["gpt2"]
        token_ids = tokenizer.encode(x)

        #truncate to supported context length
        supported_context_length = self.model.pos_embed_layer.weight.shape[1]
        token_ids = token_ids[:min(
            max_length, supported_context_length
        )]
        token_ids += [tokenizer.get_pad_token_id()] * (max_length - len(token_ids))

        input = torch.tensor(
            token_ids, device=self.device
        ).unsqueeze(0) #add batch dimension

        with torch.inference_mode():
            logits = self.model(input)[:, -1, :]
        prediced_label = torch.argmax(logits, dim=-1).item()
        return self.classif_config["id2label"][prediced_label]


    def __str__(self):
        return f"GPT2 Classifier with size {GPT_BASEMODEL_CONFIG['model_size']}"