from src._3_model_preparation.gpt_architecture.LayerNorm import LayerNorm
from src._3_model_preparation.gpt_architecture.ParMultiHeadAttention import ParMultiHeadAttention
from src._3_model_preparation.gpt_architecture.FeedForward import FeedForward

import torch.nn as nn
from typing import Dict

class TransformerBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.multi_head_attn = ParMultiHeadAttention(
            d_in = config["emb_dim"],
            d_out = config["emb_dim"],
            context_length = config["context_length"],
            dropout = config["drop_rate"],
            num_heads = config["n_heads"],
            qkv_bias = config["qkv_bias"]
            )
        self.layer_norm1 = LayerNorm(emb_dim = config["emb_dim"])
        self.layer_norm2 = LayerNorm(emb_dim = config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        batch_size, n_tokens, emb_dim = x.shape

        skip_connect = x
        x = self.layer_norm1(x)
        x = self.multi_head_attn(x)
        x = self.dropout(x)
        x = x + skip_connect 

        skip_connect = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + skip_connect
        
        return x
