import torch
import torch.nn as nn

class ParMultiHeadAttention(nn.Module):

    def __init__(self, d_in:int , d_out: int, 
                 context_length: int, dropout:float, num_heads: int, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), f"d ({d_out}) must be devisible by num_heads({num_heads})"

        self.d_out = d_out
        self.d_in = d_in
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_out_dim = d_out // num_heads
    
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

        self.out_projection = nn.Linear(d_out, d_out) #mixes concatenated output of layers

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_out_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_out_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_out_dim)

        #swaps 2nd & 3rd dim -> new shape (batch_sizeds, num_heads, num_tokens, head_out_dim)
        #Done because of batch_broadcasting and we want to have num_tokens, Head_out_dim in last 2 dims for matwrix mult
        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2)

        #compute matrix product in each batch, in each head of num_tokens, head_out_dim matrices
        attention_scores = queries @ keys.transpose(2, 3) # will do matrix mult of dims: (n_tok, head out) (head_out, n_tok) 

        #Mask attention scores for causal attention
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] 
        attention_scores = attention_scores.masked_fill_(mask_bool, -torch.inf)

        #Compute weights
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim = -1
        )
        attention_weights = self.dropout(attention_weights)

        # compute context vector
        context_vectors = attention_weights @ values
        context_vectors = context_vectors.transpose(1,2) #switch num_heads and num_tokesn again, because we want to concat head_out_dims of heads
        context_vectors = context_vectors.contiguous().view(
            batch_size, num_tokens, self.d_out
        )

        #mix concatenated contexctvectors
        context_vectors = self.out_projection(context_vectors)

        return context_vectors
        
