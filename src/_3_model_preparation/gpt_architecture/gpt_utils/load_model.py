from finetuning.gpt_download import download_and_load_gpt2
from gpt_architecture.config import GPT_CONFIG_124M
from gpt_architecture.GPTModel import  GPTModel

import torch
from typing import Dict
import numpy as np


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, dtype=torch.float32))

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_embed_layer.weight = assign(gpt.pos_embed_layer.weight, params['wpe'])
    gpt.token_embed_layer.weight = assign(gpt.token_embed_layer.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].multi_head_attn.W_query.weight = assign(
            gpt.transformer_blocks[b].multi_head_attn.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].multi_head_attn.W_key.weight = assign(
            gpt.transformer_blocks[b].multi_head_attn.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].multi_head_attn.W_value.weight = assign(
            gpt.transformer_blocks[b].multi_head_attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].multi_head_attn.W_query.bias = assign(
            gpt.transformer_blocks[b].multi_head_attn.W_query.bias, q_b)
        gpt.transformer_blocks[b].multi_head_attn.W_key.bias = assign(
            gpt.transformer_blocks[b].multi_head_attn.W_key.bias, k_b)
        gpt.transformer_blocks[b].multi_head_attn.W_value.bias = assign(
            gpt.transformer_blocks[b].multi_head_attn.W_value.bias, v_b)

        gpt.transformer_blocks[b].multi_head_attn.out_projection.weight = assign(
            gpt.transformer_blocks[b].multi_head_attn.out_projection.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].multi_head_attn.out_projection.bias = assign(
            gpt.transformer_blocks[b].multi_head_attn.out_projection.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].layer_norm1.scale = assign(
            gpt.transformer_blocks[b].layer_norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].layer_norm1.shift = assign(
            gpt.transformer_blocks[b].layer_norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].layer_norm2.scale = assign(
            gpt.transformer_blocks[b].layer_norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].layer_norm2.shift = assign(
            gpt.transformer_blocks[b].layer_norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.layer_norm.scale = assign(gpt.layer_norm.scale, params["g"])
    gpt.layer_norm.shift = assign(gpt.layer_norm.shift, params["b"])
    gpt.out_layer.weight = assign(gpt.out_layer.weight, params["wte"])
                

    return gpt                        
        
