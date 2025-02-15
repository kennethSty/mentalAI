import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        batch_size, n_tokens, dim_in = x.shape
        mean = x.mean(dim = -1, keepdim = True) #shape: (batch, n_tokens, 1)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps) #as shapes match rest is broadcasted for elementwise operation
        return self.scale * x_norm + self.shift #shift & scale are expanded (emb_dim,) -> (1, 1, emb_dim) and then broadcasted accordingly
