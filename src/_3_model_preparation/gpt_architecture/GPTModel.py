from config import GPT_CONFIG_124M
from TransformerBlock import TransformerBlock
from LayerNorm import LayerNorm

import torch
import torch.nn as nn
from typing import Dict


class GPTModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.token_embed_layer = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed_layer = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout_layer = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.layer_norm = LayerNorm(config["emb_dim"])
        # stretches to vocab size(n_tok, emb_dim) * (emb_dim, vocab_size) = (n_tokens, vocab_dize)
        self.out_layer = nn.Linear(config["emb_dim"], config["vocab_size"]) 

    def forward(self, x_ids):
        batch_size, num_tokens = x_ids.shape
        token_embeddings = self.token_embed_layer(x_ids)
        pos_embeddings = self.pos_embed_layer(
            torch.arange(num_tokens, device = x_ids.device)
        )
        x = token_embeddings + pos_embeddings #shape: (batch, num_tok, emb_dim)
        x = self.dropout_layer(x)
        z = self.transformer_blocks(x)
        z = self.layer_norm(z)
        logits = self.out_layer(z)
        return logits
        

