from src._3_model_preparation.gpt_architecture.GeLU import GeLU

import torch.nn as nn
from typing import Dict

class FeedForward(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GeLU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"])
        )
        
    def forward(self, x):
        return self.layers(x)  
